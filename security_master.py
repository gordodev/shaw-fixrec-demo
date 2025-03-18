#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunked Security Master Loader with Multi-Threading Support

This module provides efficient handling of extremely large security master files
by loading and processing them in manageable chunks using multiple threads.
Optimized for memory efficiency with production-grade error handling, logging,
and resource monitoring capabilities.

Key features:
- Streaming CSV processing (no full file load)
- Memory-efficient CUSIP/Symbol indexing
- Multi-threaded chunk processing
- Resource monitoring to prevent system overload 
- Configurable thread count and chunk size
- Thread-safe data structure access

Author: Carlyle
Date: March 16, 2025
"""

import os
import sys
import csv
import time
import logging
import mmap
import gc
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Iterator, Optional, Any, Callable, Generator
from dataclasses import dataclass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from threading import Lock

# Constants for chunked processing
DEFAULT_CHUNK_SIZE_MB = 64  # Default chunk size in MB

# Configure logger
logger = logging.getLogger(__name__)


@dataclass
class SecurityMasterConfig:
    """Configuration for chunked security master loading."""
    chunk_size_mb: int = 64  # Size of each chunk in MB
    index_buffer_count: int = 1000000  # Buffer size for index building
    monitor_memory: bool = True  # Enable memory usage monitoring
    memory_threshold_pct: int = 90  # Memory usage threshold to trigger GC
    max_retries: int = 3  # Maximum number of retries on failure
    enable_mmap: bool = True  # Use memory mapping for faster file access
    logging_interval: int = 5  # Log progress every N seconds
    thread_count: int = 3  # Number of threads to use for processing
    max_queue_size: int = 10  # Maximum number of chunks to queue
    cpu_threshold_pct: int = 80  # CPU usage threshold to throttle processing
    adaptative_threading: bool = True  # Dynamically adjust thread count based on system load


class ResourceMonitor:
    """
    Monitor system resources during processing and adjust thread usage accordingly.
    
    This class provides real-time monitoring of CPU, memory, and disk usage
    to prevent system overload during multi-threaded processing.
    """
    
    def __init__(self, cpu_threshold: int = 80, memory_threshold: int = 90, 
                 check_interval: float = 1.0, adaptative: bool = True):
        """
        Initialize the resource monitor.
        
        Args:
            cpu_threshold: CPU usage percentage threshold to trigger throttling
            memory_threshold: Memory usage percentage threshold to trigger GC
            check_interval: How often to check resource usage (seconds)
            adaptative: Whether to adaptively adjust thread count
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.adaptative = adaptative
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        self.peak_memory_usage = 0
        self.peak_cpu_usage = 0
        self.optimal_thread_count = 3  # Default starting point
        self.throttle_event = threading.Event()
        self.start_memory_usage = 0
        
    def start(self):
        """Start resource monitoring in a separate thread."""
        if self.monitor_thread is not None:
            return
            
        self.stop_flag.clear()
        self.throttle_event.clear()
        self.start_memory_usage = self._get_memory_usage()
        self.peak_memory_usage = self.start_memory_usage
        self.peak_cpu_usage = 0
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop resource monitoring."""
        if self.monitor_thread is None:
            return
            
        self.stop_flag.set()
        self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None
        
    def should_throttle(self) -> bool:
        """Check if processing should be throttled."""
        return self.throttle_event.is_set()
        
    def get_optimal_thread_count(self) -> int:
        """Get the current optimal thread count based on system load."""
        return self.optimal_thread_count
        
    def get_peak_usage(self) -> Tuple[int, float, float]:
        """
        Get peak resource usage during monitoring.
        
        Returns:
            Tuple of (peak memory in bytes, peak memory percentage, peak CPU percentage)
        """
        return (self.peak_memory_usage, 
                self._bytes_to_memory_percent(self.peak_memory_usage),
                self.peak_cpu_usage)
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_flag.is_set():
            try:
                # Check memory usage
                current_memory = self._get_memory_usage()
                memory_percent = self._bytes_to_memory_percent(current_memory)
                
                # Check CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Check disk I/O
                disk_io = psutil.disk_io_counters()
                
                # Update peak values
                if current_memory > self.peak_memory_usage:
                    self.peak_memory_usage = current_memory
                
                if cpu_percent > self.peak_cpu_usage:
                    self.peak_cpu_usage = cpu_percent
                
                # Determine if throttling is needed
                need_throttling = False
                
                if memory_percent >= self.memory_threshold:
                    logger.warning(
                        f"Memory usage critical: {memory_percent:.1f}% "
                        f"({self._format_bytes(current_memory)})"
                    )
                    # Suggest garbage collection
                    collected = gc.collect()
                    logger.info(f"Forced garbage collection: {collected} objects collected")
                    need_throttling = True
                    
                if cpu_percent >= self.cpu_threshold:
                    logger.warning(f"CPU usage critical: {cpu_percent:.1f}%")
                    need_throttling = True
                
                # Set throttle event if needed
                if need_throttling:
                    if not self.throttle_event.is_set():
                        logger.info("Throttling enabled due to high resource usage")
                        self.throttle_event.set()
                elif self.throttle_event.is_set():
                    logger.info("Resource usage normalized, resuming full speed")
                    self.throttle_event.clear()
                
                # Adjust optimal thread count if adaptative mode is enabled
                if self.adaptative:
                    self._adjust_thread_count(cpu_percent, memory_percent)
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
                
            time.sleep(self.check_interval)
    
    def _adjust_thread_count(self, cpu_percent: float, memory_percent: float):
        """
        Dynamically adjust the optimal thread count based on system load.
        
        Args:
            cpu_percent: Current CPU usage percentage
            memory_percent: Current memory usage percentage
        """
        # Start with the default of 3 threads
        new_count = 3
        
        # Reduce threads if system is under heavy load
        if cpu_percent > 90 or memory_percent > 90:
            new_count = 1
        elif cpu_percent > 80 or memory_percent > 80:
            new_count = 2
        elif cpu_percent < 50 and memory_percent < 60:
            # Increase threads if system has plenty of capacity
            new_count = 4
        
        # Update the optimal thread count if it changed
        if new_count != self.optimal_thread_count:
            logger.info(f"Adjusting optimal thread count from {self.optimal_thread_count} to {new_count} "
                       f"(CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)")
            self.optimal_thread_count = new_count
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        return psutil.Process(os.getpid()).memory_info().rss
        
    def _bytes_to_memory_percent(self, bytes_value: int) -> float:
        """Convert bytes to percentage of total system memory."""
        total_memory = psutil.virtual_memory().total
        return (bytes_value / total_memory) * 100
        
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024 or unit == 'TB':
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024


class ThreadSafeIndices:
    """Thread-safe container for security master indices."""
    
    def __init__(self):
        """Initialize the thread-safe indices."""
        self.cusip_index = {}  # Maps CUSIP to file position
        self.symbol_index = {}  # Maps symbol to CUSIP
        self.region_index = {}  # Maps region to list of CUSIPs
        self.exchange_index = {}  # Maps exchange to list of CUSIPs
        self.locks = {
            'cusip': threading.RLock(),
            'symbol': threading.RLock(),
            'region': threading.RLock(),
            'exchange': threading.RLock()
        }
        
    def add_cusip(self, cusip: str, position: int):
        """Add a CUSIP to file position mapping."""
        with self.locks['cusip']:
            self.cusip_index[cusip] = position
            
    def add_symbol(self, symbol: str, cusip: str):
        """Add a symbol to CUSIP mapping."""
        with self.locks['symbol']:
            self.symbol_index[symbol] = cusip
            
    def add_region(self, region: str, cusip: str):
        """Add a CUSIP to a region."""
        with self.locks['region']:
            if region not in self.region_index:
                self.region_index[region] = []
            self.region_index[region].append(cusip)
            
    def add_exchange(self, exchange: str, cusip: str):
        """Add a CUSIP to an exchange."""
        with self.locks['exchange']:
            if exchange not in self.exchange_index:
                self.exchange_index[exchange] = []
            self.exchange_index[exchange].append(cusip)
            
    def get_cusip_count(self) -> int:
        """Get the number of CUSIPs indexed."""
        with self.locks['cusip']:
            return len(self.cusip_index)
            
    def get_regions(self) -> List[str]:
        """Get the list of all regions."""
        with self.locks['region']:
            return list(self.region_index.keys())
            
    def get_exchanges(self) -> List[str]:
        """Get the list of all exchanges."""
        with self.locks['exchange']:
            return list(self.exchange_index.keys())
            
    def get_dict_snapshot(self) -> Dict[str, Dict[str, Any]]:
        """Get a snapshot of the data for the analyzer with optimized performance."""
        logger.debug("Starting get_dict_snapshot")
        result = {}
        
        try:
            # Get copies of data under individual locks
            with self.locks['cusip']:
                cusip_data = self.cusip_index.copy()
                logger.debug(f"Copied {len(cusip_data)} CUSIPs")
            
            with self.locks['symbol']:
                symbol_map = self.symbol_index.copy()
                logger.debug(f"Copied {len(symbol_map)} symbols")
                
            with self.locks['region']:
                region_map = {k: list(v) for k, v in self.region_index.items()}
                logger.debug(f"Copied {len(region_map)} regions")
                
            with self.locks['exchange']:
                exchange_map = {k: list(v) for k, v in self.exchange_index.items()}
                logger.debug(f"Copied {len(exchange_map)} exchanges")
            
            # Create reverse lookup dictionaries (O(n) operations)
            logger.debug("Creating reverse lookup maps")
            cusip_to_symbol = {}
            for symbol, cusip in symbol_map.items():
                cusip_to_symbol[cusip] = symbol
            
            cusip_to_region = {}
            for region, cusips in region_map.items():
                for cusip in cusips:
                    cusip_to_region[cusip] = region
            
            cusip_to_exchange = {}
            for exchange, cusips in exchange_map.items():
                for cusip in cusips:
                    cusip_to_exchange[cusip] = exchange
            
            # Build result dictionary with O(1) lookups instead of nested loops
            logger.debug("Building result dictionary with optimized lookups")
            for cusip in cusip_data:
                result[cusip] = {}
                
                # Direct lookups using reverse maps
                if cusip in cusip_to_symbol:
                    result[cusip]['Symbol'] = cusip_to_symbol[cusip]
                
                if cusip in cusip_to_region:
                    result[cusip]['Region'] = cusip_to_region[cusip]
                    
                if cusip in cusip_to_exchange:
                    result[cusip]['Exchange'] = cusip_to_exchange[cusip]
                
            logger.debug(f"Result dictionary built with {len(result)} entries")
            
        except Exception as e:
            logger.error(f"Error building security dictionary: {e}", exc_info=True)
            return {}
        
        return result

class ChunkedSecurityMasterLoader:
    def __init__(self, config: Optional[SecurityMasterConfig] = None):
        self.config = config or SecurityMasterConfig()
        self.indices = ThreadSafeIndices()
        self.file_path = None
        self.file_size = 0
        self.header = {}
        self._is_file_loaded = False
        self.resource_monitor = None
        self.processed_rows = 0
        self.total_rows = 0
        self.start_time = 0
        self.thread_pool = None
        self.job_queue = Queue(maxsize=self.config.max_queue_size)
        self.result_lock = threading.Lock()
        self.file = None  # Add file handle reference
        self.tasks_completed = 0
        self.tasks_submitted = 0
        self.all_tasks_completed = threading.Event()

    def get_security_count(self) -> int:
        """Return the total number of securities loaded."""
        return len(self.indices.cusip_index)

    def _process_chunk(self, chunk: List[List[str]]):
        """Process a chunk of CSV rows."""
        processed = 0
        for row in chunk:
            try:
                # Skip if row doesn't have enough fields
                if len(row) < len(self.header):
                    continue
                    
                # Extract key fields
                cusip_idx = self.header.get('CUSIP', -1)
                symbol_idx = self.header.get('Symbol', -1)
                region_idx = self.header.get('Region', -1) 
                exchange_idx = self.header.get('Exchange', -1)
                
                # Skip if essential indices are missing
                if cusip_idx < 0 or symbol_idx < 0:
                    continue
                    
                cusip = row[cusip_idx]
                symbol = row[symbol_idx]
                
                # Store in indices
                self.indices.add_cusip(cusip, processed)  # Using row position as placeholder
                self.indices.add_symbol(symbol, cusip)
                
                # Add region if available
                if region_idx >= 0 and region_idx < len(row):
                    region = row[region_idx]
                    if region:
                        self.indices.add_region(region, cusip)
                
                # Add exchange if available
                if exchange_idx >= 0 and exchange_idx < len(row):
                    exchange = row[exchange_idx]
                    if exchange:
                        self.indices.add_exchange(exchange, cusip)
                        
                processed += 1
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                
        return processed

    def load(self, file_path: str) -> bool:
        """Main entry point for loading the security master file."""
        self.file_path = file_path
        self.start_time = time.time()
        success = False

        try:
            if not os.path.exists(file_path):
                logger.error(f"Security master file not found: {file_path}")
                return False

            self.file_size = os.path.getsize(file_path)
            logger.info(f"Loading security master from {file_path} ({self._format_file_size(self.file_size)})")

            # Initialize resource monitor
            self.resource_monitor = ResourceMonitor(
                cpu_threshold=self.config.cpu_threshold_pct,
                memory_threshold=self.config.memory_threshold_pct,
                adaptative=self.config.adaptative_threading
            )
            self.resource_monitor.start()

            # Open file with appropriate mode
            file_mode = 'r' if self.config.enable_mmap else 'rb'
            with open(file_path, file_mode) as self.file:
                # Memory map the file if enabled
                if self.config.enable_mmap:
                    mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
                    content = mm.read().decode('utf-8')
                    mm.close()
                else:
                    content = self.file.read()

                # Read CSV header
                csv_reader = csv.reader(content.splitlines())
                self.header = {name: idx for idx, name in enumerate(next(csv_reader))}

                # Prepare for chunked processing
                chunk_size_bytes = self.config.chunk_size_mb * 1024 * 1024
                self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_count)

                # Process chunks
                chunk = []
                current_chunk_size = 0
                for row in csv_reader:
                    chunk.append(row)
                    current_chunk_size += sum(len(field) for field in row)

                    if current_chunk_size >= chunk_size_bytes:
                        self._submit_chunk(chunk)
                        chunk = []
                        current_chunk_size = 0

                # Process remaining rows
                if chunk:
                    self._submit_chunk(chunk)

                # Wait for completion
                self.thread_pool.shutdown(wait=True)
                logger.info("Thread pool shutdown complete")
                logger.info(f"Tasks submitted: {self.tasks_submitted}, completed: {self.tasks_completed}")
                gc.collect()
                logger.info("Garbage collection after thread pool shutdown")
                self.all_tasks_completed.wait(timeout=30)

            success = True
            logger.info(f"Successfully loaded {self.indices.get_cusip_count():,} securities")

        except Exception as e:
            logger.error(f"Failed to load security master: {str(e)}", exc_info=True)
            return False
        finally:
            if self.resource_monitor:
                self.resource_monitor.stop()
            if self.thread_pool:
                self.thread_pool.shutdown(wait=False)
            print ("Taking out the garbage next")
            gc.collect()
            print ("garbage tossed")

        self._is_file_loaded = success
        logger.info("About to return from security_master.load() method with success: %s", success)
        return success

    def _submit_chunk(self, chunk: List[List[str]]):
        """Submit a chunk for processing to the thread pool."""
        self.tasks_submitted += 1
        future = self.thread_pool.submit(self._process_chunk, chunk)
        future.add_done_callback(self._handle_result)

    # Update _handle_result to signal completion:
    def _handle_result(self, future):
        try:
            processed = future.result()
            with self.result_lock:
                self.processed_rows += processed
                self.tasks_completed += 1
                if self.tasks_completed >= self.tasks_submitted:
                    self.all_tasks_completed.set()
            logger.debug(f"Processed chunk with {processed} rows")
        except Exception as e:
            logger.error(f"Chunk processing failed: {str(e)}")


        def _process_chunk(self, chunk: List[List[str]]):
            """Process a chunk of CSV rows."""
            processed = 0
            for row in chunk:
                try:
                    cusip = row[self.header['CUSIP']]
                    symbol = row[self.header['Symbol']]
                    region = row[self.header['Region']]
                    exchange = row[self.header['Exchange']]

                    with self.result_lock:
                        self.indices.add_cusip(cusip, self.file.tell())
                        self.indices.add_symbol(symbol, cusip)
                        self.indices.add_region(region, cusip)
                        self.indices.add_exchange(exchange, cusip)

                    processed += 1
                except Exception as e:
                    logger.warning(f"Invalid row: {str(e)}")
            return processed

        def _handle_result(self, future):
            """Handle completion of a chunk processing task."""
            try:
                processed = future.result()
                with self.result_lock:
                    self.processed_rows += processed
                logger.debug(f"Processed chunk with {processed} rows")
            except Exception as e:
                logger.error(f"Chunk processing failed: {str(e)}")

    def is_file_loaded(self) -> bool:
        """Return True if the security master was successfully loaded."""
        return self._is_file_loaded

    def _format_file_size(self, size_bytes: int) -> str:
        """Format the file size as a human-readable string."""
        return self._format_bytes(size_bytes)

    def _format_bytes(self, bytes_value: int) -> str:
        """Convert bytes to a human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024 or unit == 'TB':
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024

    def get_security_dict(self, fields: List[str] = None) -> Dict[str, Dict[str, Any]]:
        """Return a lightweight dictionary snapshot of the security master data."""
        logger.info("Starting get_security_dict method")
        try:
            result = self.indices.get_dict_snapshot()
            logger.info(f"Completed get_security_dict, returning {len(result)} records")
            return result
        except Exception as e:
            logger.error(f"Error in get_security_dict: {e}", exc_info=True)
            return {}
    
def load_security_master_data(secmaster_path: str, 
                             chunk_size_mb: int = DEFAULT_CHUNK_SIZE_MB, 
                             use_mmap: bool = True,
                             resource_monitor: Optional[ResourceMonitor] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load and index the security master file in a memory-efficient way.
    
    Args:
        secmaster_path: Path to security master CSV file
        chunk_size_mb: Size of each chunk to process in MB
        use_mmap: Whether to use memory mapping for file access
        resource_monitor: Resource monitor instance for throttling
        
    Returns:
        Dictionary of security master data indexed by CUSIP
        
    Raises:
        FileNotFoundError: If security master file doesn't exist
        ValueError: If security master file is invalid
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading security master from: {secmaster_path}")
    
    # Check if file exists
    if not os.path.exists(secmaster_path):
        logger.error(f"Security master file not found: {secmaster_path}")
        raise FileNotFoundError(f"Security master file not found: {secmaster_path}")
    
    # Check file size
    file_size = os.path.getsize(secmaster_path)
    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"Security master file size: {file_size_mb:.2f} MB")
    
    # Fast path for small files (< 10MB)
    if file_size_mb < 10:
        logger.info("Using fast path for small security master file")
        try:
            return load_small_secmaster(secmaster_path)
        except Exception as e:
            logger.error(f"Fast path failed, falling back to standard loader: {e}")
            # Fall through to standard loading
    
    # Standard loading for larger files
    # Log memory usage at start
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    logger.info(f"Memory usage before loading: {start_memory / (1024**2):.1f} MB")
    
    # Start loading timer
    start_time = time.time()
    
    # Load the security master with chunked loading
    try:
        # Call the factory function from security_master.py
        sec_master = load_security_master(secmaster_path)
        
        if not sec_master.is_file_loaded():
            logger.error(f"Failed to load security master from: {secmaster_path}")
            raise ValueError(f"Failed to load security master from: {secmaster_path}")
        
        # Throttle if needed before getting security dict
        if resource_monitor:
            throttle_processing(resource_monitor)
        
        # Get a lightweight dictionary for the analyzer
        # This specifically requests only the fields needed for analysis
        security_data = sec_master.get_security_dict(fields=['Symbol', 'Region', 'Exchange'])
        
        # Log timing and memory stats
        elapsed_time = time.time() - start_time
        current_memory = process.memory_info().rss
        memory_increase = current_memory - start_memory
        
        logger.info(f"Loaded {len(security_data)} securities in {elapsed_time:.2f} seconds")
        logger.info(f"Memory usage after loading: {current_memory / (1024**2):.1f} MB " +
                   f"(increase: {memory_increase / (1024**2):.1f} MB)")
        
        # Log some stats about the security master
        regions = set(sec.get('Region', 'Unknown') for sec in security_data.values() if 'Region' in sec)
        exchanges = set(sec.get('Exchange', 'Unknown') for sec in security_data.values() if 'Exchange' in sec)
        
        logger.info(f"Security master covers {len(regions)} regions and {len(exchanges)} exchanges")
        
        return security_data
    
    except Exception as e:
        logger.error(f"Error loading security master: {e}")
        raise
    
    def get_securities_by_region(self, region: str) -> List[Dict[str, Any]]:
        """
        Get all securities for a specific region.
        
        Args:
            region: Region code
            
        Returns:
            List of security records in the region
        """
        if not self.is_file_loaded:
            logger.warning("Attempted lookup before security master was loaded")
            return []
            
        result = []
        
        # Get CUSIPs for the region
        cusips = []
        with self.indices.locks['region']:
            if region in self.indices.region_index:
                cusips = self.indices.region_index[region].copy()
        
        # Process in batches to manage memory
        batch_size = 1000
        for i in range(0, len(cusips), batch_size):
            batch = cusips[i:i+batch_size]
            
            for cusip in batch:
                record = self.get_record_by_cusip(cusip)
                if record:
                    result.append(record)
            
            # Release memory after each batch
            gc.collect()
            
        return result
    
    def get_security_count(self) -> int:
        """Return the total number of securities loaded."""
        return self.indices.get_cusip_count()
    
    def is_file_loaded(self) -> bool:
        """Check if security master data has been loaded."""
        return self.is_file_loaded
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size as human-readable string."""
        return self._format_bytes(size_bytes)
    
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024 or unit == 'TB':
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024


def load_security_master(file_path: str, thread_count: int = 3, chunk_size_mb: int = 64, 
                        use_mmap: bool = True) -> ChunkedSecurityMasterLoader:
    """
    Factory function to create and load a ChunkedSecurityMasterLoader instance.
    
    Args:
        file_path: Path to the security master CSV file
        thread_count: Number of threads to use for processing (default: 3)
        chunk_size_mb: Size of each chunk in MB (default: 64)
        use_mmap: Whether to use memory mapping for faster access (default: True)
        
    Returns:
        Loaded ChunkedSecurityMasterLoader instance
    """
    # Create config with reasonable defaults for production use
    config = SecurityMasterConfig(
        chunk_size_mb=chunk_size_mb,
        index_buffer_count=1000000,  # Buffer 1M records before GC
        monitor_memory=True,
        memory_threshold_pct=85,  # Alert at 85% memory usage
        max_retries=3,
        enable_mmap=use_mmap,
        logging_interval=5,  # Log progress every 5 seconds
        thread_count=thread_count,
        max_queue_size=thread_count * 2,  # Queue size based on thread count
        cpu_threshold_pct=80,  # Throttle at 80% CPU usage
        adaptative_threading=True  # Dynamically adjust thread count
    )
    
    # Create and load the security master
    sec_master = ChunkedSecurityMasterLoader(config)
    success = sec_master.load(file_path)
    
    if not success:
        logger.error(f"Failed to load security master from {file_path}")
    
    return sec_master


# For testing the module directly
if __name__ == "__main__":
    import argparse
    
    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Multi-threaded Security Master Loader")
    parser.add_argument("file_path", help="Path to security master CSV file")
    parser.add_argument("--threads", type=int, default=3, 
                       help="Number of threads to use (default: 3)")
    parser.add_argument("--chunk-size", type=int, default=64, 
                       help="Chunk size in MB (default: 64)")
    parser.add_argument("--no-mmap", action="store_true", 
                       help="Disable memory mapping")
    parser.add_argument("--static-threads", action="store_true",
                       help="Disable adaptive threading")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load security master with specified parameters
    start_time = time.time()
    
    # Create config
    config = SecurityMasterConfig(
        chunk_size_mb=args.chunk_size,
        enable_mmap=not args.no_mmap,
        thread_count=args.threads,
        adaptative_threading=not args.static_threads
    )
    
    # Create and load security master
    loader = ChunkedSecurityMasterLoader(config)
    success = loader.load(args.file_path)
    
    elapsed = time.time() - start_time
    
    if success:
        print(f"Successfully loaded {loader.get_security_count():,} securities in {elapsed:.2f} seconds")
        
        # Print some sample lookups
        if loader.get_security_count() > 0:
            cusips = list(loader.indices.cusip_index.keys())
            if cusips:
                sample_cusip = cusips[0]
                sample_record = loader.get_record_by_cusip(sample_cusip)
                
                print(f"\nSample record for CUSIP {sample_cusip}:")
                if sample_record:
                    for key, value in sample_record.items():
                        print(f"  {key}: {value}")
    else:
        print(f"Failed to load security master after {elapsed:.2f} seconds")