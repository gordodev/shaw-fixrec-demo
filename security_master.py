#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chunked Security Master Loader

This module provides efficient handling of extremely large security master files
by loading and processing them in manageable chunks rather than all at once.
Optimized for memory efficiency with production-grade error handling, logging,
and monitoring capabilities.

Key features:
- Streaming CSV processing (no full file load)
- Memory-efficient CUSIP/Symbol indexing
- Configurable chunk size to balance performance & memory usage
- Detailed performance metrics and memory monitoring
- Production-grade error handling and reporting

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


class MemoryUsageMonitor:
    """Monitor memory usage during processing."""
    
    def __init__(self, threshold_percent: int = 90, check_interval: float = 1.0):
        """
        Initialize memory monitor.
        
        Args:
            threshold_percent: Percentage threshold to trigger warnings
            check_interval: How often to check memory usage (seconds)
        """
        self.threshold_percent = threshold_percent
        self.check_interval = check_interval
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        self.peak_memory_usage = 0
        self.start_memory_usage = 0
        
    def start(self):
        """Start memory usage monitoring."""
        if self.monitor_thread is not None:
            return
            
        self.stop_flag.clear()
        self.start_memory_usage = self._get_current_usage()
        self.peak_memory_usage = self.start_memory_usage
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop memory usage monitoring."""
        if self.monitor_thread is None:
            return
            
        self.stop_flag.set()
        self.monitor_thread.join(timeout=2.0)
        self.monitor_thread = None
        
    def get_peak_usage(self) -> Tuple[int, float]:
        """
        Get peak memory usage during monitoring.
        
        Returns:
            Tuple of (peak usage in bytes, peak percentage)
        """
        return (self.peak_memory_usage, self._bytes_to_percent(self.peak_memory_usage))
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_flag.is_set():
            try:
                current_usage = self._get_current_usage()
                current_percent = self._bytes_to_percent(current_usage)
                
                # Update peak memory usage
                if current_usage > self.peak_memory_usage:
                    self.peak_memory_usage = current_usage
                
                # Check threshold
                if current_percent >= self.threshold_percent:
                    logger.warning(
                        f"Memory usage critical: {current_percent:.1f}% "
                        f"({self._format_bytes(current_usage)})"
                    )
                    # Suggest garbage collection
                    collected = gc.collect()
                    logger.info(f"Forced garbage collection: {collected} objects collected")
                
            except Exception as e:
                logger.error(f"Error in memory monitor: {e}")
                
            time.sleep(self.check_interval)
    
    def _get_current_usage(self) -> int:
        """Get current memory usage in bytes."""
        return psutil.Process(os.getpid()).memory_info().rss
        
    def _bytes_to_percent(self, bytes_value: int) -> float:
        """Convert bytes to percentage of total system memory."""
        total_memory = psutil.virtual_memory().total
        return (bytes_value / total_memory) * 100
        
    def _format_bytes(self, bytes_value: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_value < 1024 or unit == 'TB':
                return f"{bytes_value:.2f} {unit}"
            bytes_value /= 1024


class ChunkedSecurityMasterLoader:
    """
    Memory-efficient security master loader that processes files in chunks.
    
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
        self.cusip_index = {}  # Maps CUSIP to file position
        self.symbol_index = {}  # Maps symbol to CUSIP
        self.region_index = {}  # Maps region to list of CUSIPs
        self.exchange_index = {}  # Maps exchange to list of CUSIPs
        
        self.file_path = None
        self.file_size = 0
        self.header = {}
        self.is_loaded = False
        self.memory_monitor = None
        self.processed_rows = 0
        self.total_rows = 0
        self.start_time = 0
        
    def load(self, file_path: str) -> bool:
        """
        Load security master from file and build indexes.
        
        This method processes the file in chunks to avoid memory issues.
        
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
            
            # Initialize memory monitor if enabled
            if self.config.monitor_memory:
                self.memory_monitor = MemoryUsageMonitor(self.config.memory_threshold_pct)
                self.memory_monitor.start()
            
            # Process file in chunks
            if self.config.enable_mmap and self.file_size > 0:
                # Use memory mapping for faster access
                success = self._process_with_mmap()
            else:
                # Fall back to regular file reading
                success = self._process_with_reader()
                
            if success:
                self.is_loaded = True
                
                # Log processing stats
                elapsed = time.time() - self.start_time
                logger.info(f"Successfully indexed {len(self.cusip_index):,} securities in {elapsed:.2f} seconds")
                logger.info(f"Found {len(self.region_index)} regions and {len(self.exchange_index)} exchanges")
                
                # Log memory usage if monitoring was enabled
                if self.memory_monitor:
                    peak_bytes, peak_percent = self.memory_monitor.get_peak_usage()
                    logger.info(f"Peak memory usage: {self._format_bytes(peak_bytes)} ({peak_percent:.1f}%)")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to load security master: {e}")
            return False
        finally:
            # Clean up memory monitor
            if self.memory_monitor:
                self.memory_monitor.stop()
                self.memory_monitor = None
                
            # Force garbage collection
            gc.collect()
    
    def _process_with_mmap(self) -> bool:
        """Process security master using memory mapping."""
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
                
                # Init progress logging
                last_log_time = time.time()
                last_position = mm.tell()
                region_idx = self.header.get('Region', -1)
                exchange_idx = self.header.get('Exchange', -1)
                
                # Process the file line by line
                line = mm.readline()
                self.processed_rows = 0
                
                while line:
                    try:
                        # Check if it's time to log progress
                        current_time = time.time()
                        if current_time - last_log_time > self.config.logging_interval:
                            position = mm.tell()
                            progress = (position / self.file_size) * 100
                            speed = (position - last_position) / (current_time - last_log_time) / (1024 * 1024)
                            logger.info(f"Processing: {progress:.1f}% complete, speed: {speed:.1f} MB/s, rows: {self.processed_rows:,}")
                            last_log_time = current_time
                            last_position = position
                        
                        # Process the line
                        row = line.decode('utf-8').strip().split(',')
                        if len(row) >= len(self.header):
                            cusip = row[self.header['CUSIP']]
                            symbol = row[self.header['Symbol']]
                            
                            # Store position
                            position = mm.tell() - len(line)
                            self.cusip_index[cusip] = position
                            self.symbol_index[symbol] = cusip
                            
                            # Add to region index if available
                            if region_idx >= 0 and region_idx < len(row):
                                region = row[region_idx]
                                if region:
                                    if region not in self.region_index:
                                        self.region_index[region] = []
                                    self.region_index[region].append(cusip)
                            
                            # Add to exchange index if available
                            if exchange_idx >= 0 and exchange_idx < len(row):
                                exchange = row[exchange_idx]
                                if exchange:
                                    if exchange not in self.exchange_index:
                                        self.exchange_index[exchange] = []
                                    self.exchange_index[exchange].append(cusip)
                            
                            self.processed_rows += 1
                    except UnicodeDecodeError:
                        # Skip invalid UTF-8 sequences
                        logger.warning(f"Skipped invalid UTF-8 sequence at position {mm.tell()}")
                    except Exception as e:
                        logger.warning(f"Error processing row: {e}")
                    
                    # Get next line
                    line = mm.readline()
                    
                    # Trigger garbage collection if needed
                    if self.processed_rows % self.config.index_buffer_count == 0:
                        gc.collect()
                
                logger.info(f"Processed {self.processed_rows:,} rows from security master")
                return True
        except Exception as e:
            logger.error(f"Error processing file with memory mapping: {e}")
            return False
    
    def _process_with_reader(self) -> bool:
        """Process security master using standard file reader."""
        try:
            with open(self.file_path, 'r', newline='') as csvfile:
                # Calculate chunk size in bytes
                chunk_size = self.config.chunk_size_mb * 1024 * 1024
                
                # Parse header
                reader = csv.reader(csvfile)
                header_row = next(reader)
                self.header = {field: i for i, field in enumerate(header_row)}
                
                # Check required fields
                if 'CUSIP' not in self.header or 'Symbol' not in self.header:
                    logger.error("Security master file missing required columns: CUSIP, Symbol")
                    return False
                
                # Remember column indices for efficiency
                cusip_idx = self.header['CUSIP']
                symbol_idx = self.header['Symbol']
                region_idx = self.header.get('Region', -1)
                exchange_idx = self.header.get('Exchange', -1)
                
                # Init progress logging
                last_log_time = time.time()
                last_position = csvfile.tell()
                
                # Process the remaining rows
                self.processed_rows = 0
                
                for row in reader:
                    try:
                        # Check if it's time to log progress
                        current_time = time.time()
                        if current_time - last_log_time > self.config.logging_interval:
                            position = csvfile.tell()
                            progress = (position / self.file_size) * 100
                            speed = (position - last_position) / (current_time - last_log_time) / (1024 * 1024)
                            logger.info(f"Processing: {progress:.1f}% complete, speed: {speed:.1f} MB/s, rows: {self.processed_rows:,}")
                            last_log_time = current_time
                            last_position = position
                        
                        # Process the row
                        if len(row) >= len(self.header):
                            cusip = row[cusip_idx]
                            symbol = row[symbol_idx]
                            
                            if cusip and symbol:
                                # Store position
                                position = csvfile.tell()
                                self.cusip_index[cusip] = position
                                self.symbol_index[symbol] = cusip
                                
                                # Add to region index if available
                                if region_idx >= 0 and region_idx < len(row):
                                    region = row[region_idx]
                                    if region:
                                        if region not in self.region_index:
                                            self.region_index[region] = []
                                        self.region_index[region].append(cusip)
                                
                                # Add to exchange index if available
                                if exchange_idx >= 0 and exchange_idx < len(row):
                                    exchange = row[exchange_idx]
                                    if exchange:
                                        if exchange not in self.exchange_index:
                                            self.exchange_index[exchange] = []
                                        self.exchange_index[exchange].append(cusip)
                            
                            self.processed_rows += 1
                    except Exception as e:
                        logger.warning(f"Error processing row: {e}")
                    
                    # Trigger garbage collection if needed
                    if self.processed_rows % self.config.index_buffer_count == 0:
                        gc.collect()
                
                logger.info(f"Processed {self.processed_rows:,} rows from security master")
                return True
        except Exception as e:
            logger.error(f"Error processing file with reader: {e}")
            return False
    
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
        if not self.is_loaded or cusip not in self.cusip_index:
            return None
            
        try:
            position = self.cusip_index[cusip]
            
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
    
    def get_symbol_by_cusip(self, cusip: str) -> Optional[str]:
        """
        Get the symbol for a given CUSIP.
        
        Args:
            cusip: CUSIP identifier
            
        Returns:
            Symbol string or None if not found
        """
        if not self.is_loaded or cusip not in self.cusip_index:
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
            
        return self.symbol_index.get(symbol)
    
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
            
        # Build a lightweight dictionary with just the Symbol (required) and any
        # other fields that might be useful for grouping or display
        security_dict = {}
        essential_fields = ['Symbol', 'Region', 'Exchange']
        
        # Process each CUSIP in batches to manage memory
        cusips = list(self.cusip_index.keys())
        batch_size = min(10000, len(cusips))
        
        for i in range(0, len(cusips), batch_size):
            batch = cusips[i:i+batch_size]
            
            for cusip in batch:
                try:
                    record = self.get_record_by_cusip(cusip)
                    if record:
                        # Extract just the essential fields
                        security_dict[cusip] = {
                            field: record.get(field, '') 
                            for field in essential_fields 
                            if field in self.header
                        }
                except Exception as e:
                    logger.warning(f"Error retrieving record for CUSIP {cusip}: {e}")
            
            # Release memory after each batch
            gc.collect()
            
        return security_dict
    
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
            
        cusips = self.region_index.get(region, [])
        result = []
        
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
        return len(self.cusip_index)
    
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


def load_security_master(file_path: str) -> ChunkedSecurityMasterLoader:
    """
    Factory function to create and load a ChunkedSecurityMasterLoader instance.
    
    Args:
        file_path: Path to the security master CSV file
        
    Returns:
        Loaded ChunkedSecurityMasterLoader instance
    """
    # Create config with reasonable defaults for production use
    config = SecurityMasterConfig(
        chunk_size_mb=64,  # Process in 64MB chunks
        index_buffer_count=1000000,  # Buffer 1M records before GC
        monitor_memory=True,  # Enable memory monitoring
        memory_threshold_pct=85,  # Alert at 85% memory usage
        max_retries=3,  # Retry up to 3 times on failure
        enable_mmap=True,  # Use memory mapping for faster access
        logging_interval=5  # Log progress every 5 seconds
    )
    
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
    parser = argparse.ArgumentParser(description="Chunked Security Master Loader")
    parser.add_argument("file_path", help="Path to security master CSV file")
    parser.add_argument("--chunk-size", type=int, default=64, 
                       help="Chunk size in MB (default: 64)")
    parser.add_argument("--no-mmap", action="store_true", 
                       help="Disable memory mapping")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create config
    config = SecurityMasterConfig(
        chunk_size_mb=args.chunk_size,
        enable_mmap=not args.no_mmap
    )
    
    # Create and load security master
    loader = ChunkedSecurityMasterLoader(config)
    success = loader.load(args.file_path)
    
    if success:
        print(f"Successfully loaded {loader.get_security_count():,} securities")
        
        # Print some sample lookups
        if loader.get_security_count() > 0:
            cusips = list(loader.cusip_index.keys())
            sample_cusip = cusips[0]
            sample_record = loader.get_record_by_cusip(sample_cusip)
            
            print(f"\nSample record for CUSIP {sample_cusip}:")
            for key, value in sample_record.items():
                print(f"  {key}: {value}")
    else:
        print("Failed to load security master")
