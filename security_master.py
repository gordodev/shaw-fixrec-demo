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
        """Get a snapshot of the data for the analyzer."""
        result = {}
        with self.locks['cusip'], self.locks['symbol'], self.locks['region'], self.locks['exchange']:
            for cusip in self.cusip_index.keys():
                result[cusip] = {}
                # Add symbol if available
                for symbol, c in self.symbol_index.items():
                    if c == cusip:
                        result[cusip]['Symbol'] = symbol
                        break
                # Add region if available
                for region, cusips in self.region_index.items():
                    if cusip in cusips:
                        result[cusip]['Region'] = region
                        break
                # Add exchange if available
                for exchange, cusips in self.exchange_index.items():
                    if cusip in cusips:
                        result[cusip]['Exchange'] = exchange
                        break
        return result


class ChunkedSecurityMasterLoader:
    """
    Memory-efficient security master loader that processes files in chunks
    using multiple threads for improved performance.
    
    This class allows processing extremely large security master files
    without loading the entire file into memory at once. It builds indexes
    incrementally and provides streaming access to the data.
    """
    
    def __init__(self, config: Optional[SecurityMasterConfig] = None):
        """
        Initialize the chunked loader.
        
        Args:
            config: Configuration options (or None for defaults)
        """
        self.config = config or SecurityMasterConfig()
        self.indices = ThreadSafeIndices()
        
        self.file_path = None
        self.file_size = 0
        self.header = {}
        self.is_loaded = False
        self.resource_monitor = None
        self.processed_rows = 0
        self.total_rows = 0
        self.start_time = 0
        self.thread_pool = None
        self.job_queue = Queue(maxsize=self.config.max_queue_size)
        self.result_lock = threading.Lock()
        
    def load(self, file_path: str) -> bool:
        """
        Load security master from file and build indexes using multiple threads.
        
        Args:
            file_path: Path to security master CSV file
            
        Returns:
            True if successful, False otherwise
        """
        self.file_path = file_path
        self.start_time = time.time()
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"Security master file not found: {file_path}")
                return False
                
            # Get file size
            self.file_size = os.path.getsize(file_path)
            logger.info(f"Loading security master from {file_path} ({self._format_file_size(self.file_size)})")
            
            # Initialize resource monitor
            self.resource_monitor = ResourceMonitor(
                cpu_threshold=self.config.cpu_threshold_pct,
                memory_threshold=self.config.memory_threshold_pct,
                adaptative=self.config.adaptative_threading
            )
            self.resource_monitor.start()
            
            # Process file in chunks using multiple threads
            if self.config.enable_mmap and self.file_size > 0:
                # Use memory mapping for faster access with chunking
                success = self._process_with_mmap_threaded()
            else:
                # Fall back to regular file reading
                success = self._process_with_reader_threaded()
                
            if success:
                self.is_loaded = True
                
                # Log processing stats
                elapsed = time.time() - self.start_time
                logger.info(f"Successfully indexed {self.indices.get_cusip_count():,} securities "
                           f"in {elapsed:.2f} seconds")
                logger.info(f"Found {len(self.indices.get_regions())} regions and "
                           f"{len(self.indices.get_exchanges())} exchanges")
                
                # Log peak resource usage
                peak_memory, peak_memory_pct, peak_cpu = self.resource_monitor.get_peak_usage()
                logger.info(f"Peak memory usage: {self._format_bytes(peak_memory)} ({peak_memory_pct:.1f}%)")
                logger.info(f"Peak CPU usage: {peak_cpu:.1f}%")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load security master: {e}")
            return False
        finally:
            # Clean up resources
            if self.resource_monitor:
                self.resource_monitor.stop()
                self.resource_monitor = None
                
            # Clean up thread pool if it exists
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.thread_pool = None
                
            # Force garbage collection
            gc.collect()
    
    def _process_chunk(self, chunk_data: Tuple[bytes, int, Dict[str, int]]) -> int:
        """
        Process a chunk of data from the security master file.
        
        Args:
            chunk_data: Tuple of (data, offset, header_indices)
            
        Returns:
            Number of rows processed
        """
        data, offset, header_indices = chunk_data
        rows_processed = 0
        
        # Get header indices for fields we care about
        cusip_idx = header_indices.get('CUSIP', -1)
        symbol_idx = header_indices.get('Symbol', -1)
        region_idx = header_indices.get('Region', -1)
        exchange_idx = header_indices.get('Exchange', -1)
        
        # Check if we have the minimum required indices
        if cusip_idx < 0 or symbol_idx < 0:
            logger.error("Missing required columns in header")
            return 0
        
        # Split data into lines
        lines = data.split(b'\n')
        line_offset = offset
        
        for line in lines:
            # Skip empty lines
            if not line.strip():
                line_offset += len(line) + 1
                continue
                
            try:
                # Check if we should throttle processing
                if self.resource_monitor and self.resource_monitor.should_throttle():
                    time.sleep(0.1)  # Brief pause to reduce resource usage
                
                # Decode and split the line
                decoded_line = line.decode('utf-8').strip()
                row = decoded_line.split(',')
                
                # Process the row if it has enough fields
                if len(row) > max(cusip_idx, symbol_idx):
                    cusip = row[cusip_idx]
                    symbol = row[symbol_idx]
                    
                    if cusip and symbol:
                        # Add to indices
                        self.indices.add_cusip(cusip, line_offset)
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
                        
                        rows_processed += 1
                        
                        # Update total processed rows
                        with self.result_lock:
                            self.processed_rows += 1
            except UnicodeDecodeError:
                logger.warning(f"Skipped invalid UTF-8 sequence at offset {line_offset}")
            except Exception as e:
                logger.warning(f"Error processing row at offset {line_offset}: {e}")
            
            # Update line offset for the next iteration
            line_offset += len(line) + 1
        
        return rows_processed
    
    def _process_with_mmap_threaded(self) -> bool:
        """Process security master using memory mapping with multiple threads."""
        try:
            with open(self.file_path, 'rb') as f:
                # Create memory-mapped file
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Find and parse header
                header_line = mm.readline().decode('utf-8').strip()
                header_fields = header_line.split(',')
                self.header = {field: i for i, field in enumerate(header_fields)}
                
                # Check required fields
                if 'CUSIP' not in self.header or 'Symbol' not in self.header:
                    logger.error("Security master file missing required columns: CUSIP, Symbol")
                    return False
                
                # Calculate chunk size in bytes from config
                chunk_size_bytes = self.config.chunk_size_mb * 1024 * 1024
                
                # Initialize progress tracking
                last_log_time = time.time()
                last_position = mm.tell()
                
                # Create a thread pool for processing chunks
                thread_count = self.config.thread_count
                logger.info(f"Processing with {thread_count} threads")
                self.thread_pool = ThreadPoolExecutor(max_workers=thread_count)
                
                # Start producer thread to enqueue chunks
                producer_thread = threading.Thread(
                    target=self._chunk_producer, 
                    args=(mm, chunk_size_bytes, self.header)
                )
                producer_thread.daemon = True
                producer_thread.start()
                
                # Start worker threads to process chunks
                futures = []
                for _ in range(thread_count):
                    futures.append(self.thread_pool.submit(self._chunk_consumer))
                
                # Monitor progress
                try:
                    while producer_thread.is_alive() or not self.job_queue.empty():
                        # Check if it's time to log progress
                        current_time = time.time()
                        if current_time - last_log_time > self.config.logging_interval:
                            position = mm.tell()
                            progress = (position / self.file_size) * 100
                            speed = (position - last_position) / (current_time - last_log_time) / (1024 * 1024)
                            
                            # Log progress
                            logger.info(f"Processing: {progress:.1f}% complete, "
                                       f"speed: {speed:.1f} MB/s, "
                                       f"rows: {self.processed_rows:,}")
                            
                            # Update values for next iteration
                            last_log_time = current_time
                            last_position = position
                            
                            # Check if we should adjust thread count
                            if self.resource_monitor and self.config.adaptative_threading:
                                optimal_count = self.resource_monitor.get_optimal_thread_count()
                                if optimal_count != thread_count:
                                    logger.info(f"Adjusting thread count from {thread_count} to {optimal_count}")
                                    # This doesn't change existing threads, but affects future ones
                                    if optimal_count > thread_count:
                                        # Add more worker threads
                                        for _ in range(optimal_count - thread_count):
                                            futures.append(self.thread_pool.submit(self._chunk_consumer))
                                    thread_count = optimal_count
                            
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    logger.warning("Interrupted by user, finishing current tasks...")
                
                # Wait for producer to finish
                producer_thread.join()
                
                # Shutdown the thread pool
                self.thread_pool.shutdown(wait=True)
                
                logger.info(f"Processed {self.processed_rows:,} rows from security master")
                return True
                
        except Exception as e:
            logger.error(f"Error in multi-threaded processing: {e}")
            return False
    
    def _chunk_producer(self, mm: mmap.mmap, chunk_size: int, header: Dict[str, int]):
        """
        Producer function that reads chunks from the file and puts them in the job queue.
        
        Args:
            mm: Memory-mapped file
            chunk_size: Size of each chunk in bytes
            header: Dictionary mapping field names to column indices
        """
        try:
            # Set initial offset after header
            current_offset = mm.tell()
            remainder = b""
            remainder_offset = current_offset
            
            while True:
                # Try to read a chunk
                chunk = mm.read(chunk_size)
                if not chunk:
                    break
                
                # Prepend remainder from previous chunk if any
                data = remainder + chunk
                data_offset = remainder_offset
                
                # Find the last newline in the chunk
                last_newline = data.rfind(b'\n')
                if last_newline == -1:
                    # No newline in chunk, unusual but possible
                    remainder = data
                    remainder_offset = data_offset
                    continue
                
                # Split data at the last newline
                usable_data = data[:last_newline+1]
                remainder = data[last_newline+1:]
                remainder_offset = data_offset + last_newline + 1
                
                # Put the chunk in the job queue
                while True:
                    # Check if we should throttle
                    if self.resource_monitor and self.resource_monitor.should_throttle():
                        time.sleep(0.1)
                        continue
                    
                    try:
                        self.job_queue.put((usable_data, data_offset, header), block=True, timeout=1)
                        break
                    except Exception:
                        # Queue is full, wait a bit
                        if self.resource_monitor and self.resource_monitor.should_throttle():
                            time.sleep(0.5)
                        else:
                            time.sleep(0.1)
            
            # Process any remaining data
            if remainder:
                self.job_queue.put((remainder, remainder_offset, header), block=True)
                
        except Exception as e:
            logger.error(f"Error in chunk producer: {e}")
            
        finally:
            # Signal that we're done by adding None to the queue for each worker
            for _ in range(self.config.thread_count * 2):  # Add extra to ensure all workers get the signal
                try:
                    self.job_queue.put(None, block=True, timeout=1)
                except Exception:
                    pass
    
    def _chunk_consumer(self):
        """
        Consumer function that processes chunks from the job queue.
        
        This function runs in a worker thread and processes chunks until
        it receives a None sentinel value.
        """
        while True:
            try:
                # Get a job from the queue
                job = self.job_queue.get(block=True)
                
                # Check if we're done
                if job is None:
                    break
                
                # Process the chunk
                self._process_chunk(job)
                
                # Mark the job as done
                self.job_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error in chunk consumer: {e}")
    
    def _process_with_reader_threaded(self) -> bool:
        """Process security master using standard file reader with multiple threads."""
        logger.info("Memory mapping disabled, using standard file reader")
        
        try:
            with open(self.file_path, 'r', newline='') as csvfile:
                # Parse header
                reader = csv.reader(csvfile)
                header_row = next(reader)
                self.header = {field: i for i, field in enumerate(header_row)}
                
                # Check required fields
                if 'CUSIP' not in self.header or 'Symbol' not in self.header:
                    logger.error("Security master file missing required columns: CUSIP, Symbol")
                    return False
                
                # Calculate chunk size - number of rows per chunk
                rows_per_chunk = 100000  # Adjust based on expected row size
                
                # Create a thread pool for processing
                self.thread_pool = ThreadPoolExecutor(max_workers=self.config.thread_count)
                
                # Initialize tracking
                chunk_rows = []
                chunks_submitted = 0
                last_log_time = time.time()
                
                # Process in chunks
                for row in reader:
                    chunk_rows.append(row)
                    
                    # When we have enough rows, submit the chunk for processing
                    if len(chunk_rows) >= rows_per_chunk:
                        self._submit_row_chunk(chunk_rows, self.header)
                        chunks_submitted += 1
                        chunk_rows = []
                        
                        # Check if it's time to log progress
                        current_time = time.time()
                        if current_time - last_log_time > self.config.logging_interval:
                            position = csvfile.tell()
                            progress = (position / self.file_size) * 100
                            logger.info(f"Processing: {progress:.1f}% complete, "
                                       f"rows: {self.processed_rows:,}, "
                                       f"chunks: {chunks_submitted}")
                            last_log_time = current_time
                
                # Process any remaining rows
                if chunk_rows:
                    self._submit_row_chunk(chunk_rows, self.header)
                    chunks_submitted += 1
                
                # Shutdown the thread pool
                self.thread_pool.shutdown(wait=True)
                
                logger.info(f"Processed {self.processed_rows:,} rows from security master "
                           f"in {chunks_submitted} chunks")
                return True
                
        except Exception as e:
            logger.error(f"Error in reader-based processing: {e}")
            return False
    
    def _submit_row_chunk(self, rows: List[List[str]], header: Dict[str, int]):
        """
        Submit a chunk of rows for processing.
        
        Args:
            rows: List of row data
            header: Dictionary mapping field names to column indices
        """
        # Wait until the thread pool is available
        while self.resource_monitor and self.resource_monitor.should_throttle():
            time.sleep(0.1)
            
        # Submit the chunk for processing
        self.thread_pool.submit(self._process_row_chunk, rows, header)
    
    def _process_row_chunk(self, rows: List[List[str]], header: Dict[str, int]):
        """
        Process a chunk of rows.
        
        Args:
            rows: List of row data
            header: Dictionary mapping field names to column indices
        """
        # Get indices for the fields we care about
        cusip_idx = header.get('CUSIP', -1)
        symbol_idx = header.get('Symbol', -1)
        region_idx = header.get('Region', -1)
        exchange_idx = header.get('Exchange', -1)
        
        # Check if we have the required indices
        if cusip_idx < 0 or symbol_idx < 0:
            logger.error("Missing required columns in header")
            return
        
        # Process each row
        for row in rows:
            try:
                # Check if we should throttle
                if self.resource_monitor and self.resource_monitor.should_throttle():
                    time.sleep(0.1)
                    
                # Make sure the row has enough fields
                if len(row) <= max(cusip_idx, symbol_idx):
                    continue
                    
                cusip = row[cusip_idx]
                symbol = row[symbol_idx]
                
                if cusip and symbol:
                    # Add to indices - use position 0 as a placeholder since we don't
                    # have the actual file position in this mode
                    self.indices.add_cusip(cusip, 0)
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
                    
                    # Update processed rows counter
                    with self.result_lock:
                        self.processed_rows += 1
                        
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
    
    def get_record_by_cusip(self, cusip: str) -> Optional[Dict[str, str]]:
        """
        Get a security record by CUSIP.
        
        Instead of keeping all records in memory, this fetches the specific
        record from disk when needed.
        
        Args:
            cusip: CUSIP identifier
            
        Returns:
            Security record dictionary or None if not found
        """
        if not self.is_loaded:
            return None
            
        try:
            position = self.indices.cusip_index.get(cusip)
            if position is None:
                return None
                
            # If position is 0, it means we don't have the actual file position
            # In that case, we need to search the file for the CUSIP
            if position == 0:
                return self._search_by_cusip(cusip)
            
            # Read the record from the file
            with open(self.file_path, 'r', newline='') as f:
                f.seek(position)
                line = f.readline().strip()
                
                # Parse the CSV line
                values = next(csv.reader([line]))
                
                # Create dictionary
                return {field: values[idx] for field, idx in self.header.items() 
                        if idx < len(values)}
                
        except Exception as e:
            logger.error(f"Error retrieving record by CUSIP: {e}")
            return None
    
    def _search_by_cusip(self, cusip: str) -> Optional[Dict[str, str]]:
        """
        Search for a record by CUSIP when the file position is not known.
        
        Args:
            cusip: CUSIP identifier
            
        Returns:
            Security record dictionary or None if not found
        """
        try:
            with open(self.file_path, 'r', newline='') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                
                cusip_idx = self.header.get('CUSIP')
                if cusip_idx is None:
                    return None
                    
                for row in reader:
                    if len(row) > cusip_idx and row[cusip_idx] == cusip:
                        return {field: row[idx] for field, idx in self.header.items() 
                                if idx < len(row)}
                        
            return None
            
        except Exception as e:
            logger.error(f"Error searching by CUSIP: {e}")
            return None
    
    def get_symbol_by_cusip(self, cusip: str) -> Optional[str]:
        """
        Get the symbol for a given CUSIP.
        
        Args:
            cusip: CUSIP identifier
            
        Returns:
            Symbol string or None if not found
        """
        if not self.is_loaded:
            return None
            
        try:
            record = self.get_record_by_cusip(cusip)
            return record.get('Symbol') if record else None
        except Exception as e:
            logger.error(f"Error retrieving symbol by CUSIP: {e}")
            return None
    
    def get_cusip_by_symbol(self, symbol: str) -> Optional[str]:
        """
        Get CUSIP by symbol.
        
        Args:
            symbol: Security symbol
            
        Returns:
            CUSIP string or None if not found
        """
        if not self.is_loaded:
            return None
            
        return self.indices.symbol_index.get(symbol)
    
    def validate_symbol_cusip_pair(self, symbol: str, cusip: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a symbol matches the expected symbol for a CUSIP.
        
        Args:
            symbol: Symbol to validate
            cusip: CUSIP to check against
            
        Returns:
            Tuple of (is_valid, expected_symbol)
            - is_valid: True if symbol matches CUSIP, False otherwise
            - expected_symbol: Expected symbol if different, None if match
        """
        if not self.is_loaded:
            logger.warning("Attempted validation before security master was loaded")
            return (False, None)
            
        expected_symbol = self.get_symbol_by_cusip(cusip)
        
        if not expected_symbol:
            logger.warning(f"CUSIP {cusip} not found in security master")
            return (False, None)
            
        if symbol == expected_symbol:
            return (True, None)
        else:
            return (False, expected_symbol)
    
    def get_security_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        Get a lightweight dictionary suitable for the analyzer.
        
        This creates a minimal representation needed by the analyzer
        without loading all record details into memory.
        
        Returns:
            Dictionary mapping CUSIPs to minimal security records
        """
        if not self.is_loaded:
            logger.warning("Attempted to get security dict before security master was loaded")
            return {}
            
        # Get a thread-safe snapshot of the indices
        return self.indices.get_dict_snapshot()
    
    def get_securities_by_region(self, region: str) -> List[Dict[str, Any]]:
        """
        Get all securities for a specific region.
        
        Args:
            region: Region code
            
        Returns:
            List of security records in the region
        """
        if not self.is_loaded:
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
        return self.is_loaded
    
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