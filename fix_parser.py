#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX Protocol Parser for Security Master Reconciliation Tool

This module handles parsing of FIX 4.2 messages from log files, optimized for
memory efficiency and performance. It focuses only on extracting the minimal
set of required fields (Symbol, CUSIP, Quantity, Price) to minimize memory usage.

Features:
- Streaming line-by-line processing to minimize memory footprint
- Memory-efficient data types (float32 instead of float64)
- Selective field extraction (ignores irrelevant FIX fields)
- Resource monitoring with automatic throttling
- Optimized for handling extremely large files (60GB+)

Author: Carlyle
Date: March 17, 2025
"""

import re
import os
import gc
import time
import logging
import threading
import numpy as np
import psutil
from typing import Dict, List, Optional, Tuple, Any, Iterator, Generator
from dataclasses import dataclass, field
import mmap
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FIXMessage:
    """
    Data class representing essential fields of a FIX message for reconciliation purposes.
    Uses memory-efficient data types (float32) for numerical values.
    """
    msg_type: str
    symbol: str
    cusip: str
    quantity: np.float32
    price: np.float32
    message_id: str  # For audit/tracing
    order_id: Optional[str] = None
    raw_message: Optional[str] = None  # Stored only in debug mode for troubleshooting

    def calculate_exposure(self) -> np.float32:
        """Calculate the financial exposure (price * quantity)."""
        return self.price * self.quantity


# This will be imported from resource_monitor.py in the future
# For now, implementing a simplified placeholder version
class ResourceMonitor:
    """
    Simplified placeholder for the full resource_monitor.py implementation.
    
    This class will be replaced with an import from the dedicated module
    in Phase 2 of the performance improvement plan.
    """
    
    def __init__(self, cpu_threshold: float = 70.0, memory_threshold: float = 80.0):
        """
        Initialize basic resource monitoring capabilities.
        
        Args:
            cpu_threshold: CPU usage percentage that triggers throttling
            memory_threshold: Memory usage percentage that triggers throttling
        """
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
        self._throttle_check_count = 0
        
    def start(self) -> None:
        """Start basic resource monitoring."""
        self.peak_cpu = 0.0
        self.peak_memory = 0.0
        logger.debug("Basic resource monitoring started")
        
    def stop(self) -> None:
        """Stop resource monitoring."""
        logger.debug("Resource monitoring stopped")
        
    def should_throttle(self) -> bool:
        """
        Simple check if processing should be throttled based on resource usage.
        
        This simplified version only checks resources periodically to reduce overhead.
        The full implementation in resource_monitor.py will be more sophisticated.
        """
        # Only check every 10 calls to reduce overhead
        self._throttle_check_count += 1
        if self._throttle_check_count % 10 != 0:
            return False
            
        # Basic resource check
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent
        
        # Update peak values
        if cpu_percent > self.peak_cpu:
            self.peak_cpu = cpu_percent
        if memory_percent > self.peak_memory:
            self.peak_memory = memory_percent
            
        # Determine if we should throttle
        should_throttle = (cpu_percent > self.cpu_threshold or 
                           memory_percent > self.memory_threshold)
                           
        if should_throttle:
            # Trigger garbage collection when throttling
            gc.collect()
            logger.debug(f"Resource check: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}% - throttling")
            
        return should_throttle
        
    def get_stats(self) -> Dict[str, float]:
        """Get basic resource usage statistics."""
        return {
            'peak_cpu': self.peak_cpu,
            'peak_memory': self.peak_memory,
            'current_cpu': psutil.cpu_percent(),
            'current_memory': psutil.virtual_memory().percent
        }


class FIXParser:
    """
    Optimized parser for FIX 4.2 protocol messages.
    
    Extracts only required fields (Symbol, CUSIP, Quantity, Price) from FIX
    messages for memory efficiency. Implements streaming processing with
    resource monitoring to prevent system overload.
    """
    
    # FIX message types we care about
    NEW_ORDER_SINGLE = "D"
    EXECUTION_REPORT = "8"
    
    # Required tags for our analysis (minimized for efficiency)
    REQUIRED_TAGS = {
        "35": None,  # MsgType
        "55": None,  # Symbol
        "48": None,  # SecurityID (CUSIP)
        "38": None,  # OrderQty
        "44": None,  # Price
        "11": None,  # ClOrdID (for Order ID)
    }
    
    def __init__(self, delimiter: str = "|", chunk_size_mb: int = 64, enable_mmap: bool = True, debug_mode: bool = False):
        """
        Initialize FIX parser with the specified configuration.
        
        Args:
            delimiter: Character separating FIX fields, typically '|' in log files
            chunk_size_mb: Size of each processing chunk in MB
            enable_mmap: Whether to use memory mapping for file access
            debug_mode: Whether to store raw messages for debugging
        """
        self.delimiter = delimiter
        self.chunk_size_mb = chunk_size_mb
        self.enable_mmap = enable_mmap
        self.debug_mode = debug_mode
        
        # Statistics
        self.message_count = 0
        self.valid_message_count = 0
        self.invalid_message_count = 0
        self.start_time = 0
        
        # Basic resource monitoring (Phase 1)
        # In Phase 2, this will be replaced with an imported module
        self.resource_monitor = ResourceMonitor()
        
        # Precompile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Precompile regex patterns for parsing FIX messages."""
        # Pattern to identify FIX version
        self.fix_version_pattern = re.compile(r"8=FIX\.4\.2")
        
        # Pattern to extract tag-value pairs
        # Using a more efficient pattern that only extracts tags we care about
        tag_pattern = '|'.join(self.REQUIRED_TAGS.keys())
        self.tag_value_pattern = re.compile(f"({tag_pattern})=([^{self.delimiter}]+)")
    
    def parse_file(self, file_path: str) -> List[FIXMessage]:
        """
        Parse a FIX log file and extract relevant messages.
        
        Args:
            file_path: Path to the FIX log file
            
        Returns:
            List of parsed FIX messages
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            PermissionError: If the file cannot be accessed due to permissions
        """
        logger.info(f"Parsing FIX messages from {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"FIX log file not found: {file_path}")
        
        # Record start time
        self.start_time = time.time()
        
        # Start resource monitoring
        self.resource_monitor.start()
        
        # Reset counters
        self.message_count = 0
        self.valid_message_count = 0
        self.invalid_message_count = 0
        
        messages = []
        
        try:
            # Determine parsing method based on configuration
            if self.enable_mmap and os.path.getsize(file_path) > 0:
                messages = self._parse_with_mmap(file_path)
            else:
                messages = self._parse_with_streaming(file_path)
            
            # Calculate performance metrics
            elapsed_time = time.time() - self.start_time
            messages_per_second = self.message_count / elapsed_time if elapsed_time > 0 else 0
            
            # Log summary
            logger.info(
                f"Parsed {self.valid_message_count:,} valid messages "
                f"({self.invalid_message_count:,} invalid) "
                f"in {elapsed_time:.2f} seconds "
                f"({messages_per_second:.1f} msgs/sec)"
            )
            
            # Log resource usage
            stats = self.resource_monitor.get_stats()
            logger.info(
                f"Resource usage - Peak CPU: {stats['peak_cpu']:.1f}%, "
                f"Peak Memory: {stats['peak_memory']:.1f}%"
            )
            
            return messages
            
        except Exception as e:
            logger.error(f"Error parsing FIX log file: {e}")
            raise
        finally:
            # Stop resource monitoring
            self.resource_monitor.stop()
            
            # Force garbage collection
            gc.collect()
    
    def _parse_with_mmap(self, file_path: str) -> List[FIXMessage]:
        """
        Parse FIX messages using memory mapping for efficiency.
        
        Args:
            file_path: Path to the FIX log file
            
        Returns:
            List of parsed FIX messages
        """
        messages = []
        
        try:
            file_size = os.path.getsize(file_path)
            chunk_size = self.chunk_size_mb * 1024 * 1024
            
            with open(file_path, 'rb') as f:
                # Create memory-mapped file object
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                
                # Process file in chunks
                offset = 0
                last_log_time = time.time()
                last_progress = 0
                last_throttle_check = time.time()
                
                try:
                    while offset < file_size:
                        # Check for resource throttling
                        if time.time() - last_throttle_check > 1.0:  # Check every second
                            last_throttle_check = time.time()
                            if self.resource_monitor.should_throttle():
                                # Sleep to allow system to recover
                                time.sleep(0.5)
                        
                        # Read a chunk from the current offset
                        mm.seek(offset)
                        chunk = mm.read(chunk_size)
                        if not chunk:
                            break
                        
                        # Process messages in the chunk
                        lines = chunk.split(b'\n')
                        
                        # Handle potential partial line at the end
                        incomplete_line = lines.pop() if chunk[-1:] != b'\n' else b''
                        
                        for line in lines:
                            self.message_count += 1
                            decoded_line = line.decode('utf-8', errors='ignore')
                            
                            # Only process FIX 4.2 messages
                            if self.fix_version_pattern.search(decoded_line):
                                message = self.parse_message(decoded_line)
                                if message:
                                    messages.append(message)
                                    self.valid_message_count += 1
                                else:
                                    self.invalid_message_count += 1
                            
                            # Periodically log progress and free memory
                            if self.message_count % 100000 == 0:
                                gc.collect()
                        
                        # Move offset forward
                        offset += len(chunk) - len(incomplete_line)
                        
                        # Log progress periodically
                        current_time = time.time()
                        if current_time - last_log_time > 5.0:  # Log every 5 seconds
                            progress = (offset / file_size) * 100
                            msgs_per_sec = (self.message_count - last_progress) / (current_time - last_log_time)
                            logger.info(f"Parsing progress: {progress:.1f}%, {msgs_per_sec:.1f} msgs/sec")
                            last_log_time = current_time
                            last_progress = self.message_count
                        
                finally:
                    # Always close the memory map
                    mm.close()
        
        except Exception as e:
            logger.error(f"Error in memory-mapped parsing: {e}")
            # Fall back to streaming if memory mapping fails
            logger.info("Falling back to streaming parser")
            return self._parse_with_streaming(file_path)
        
        return messages
    
    def _parse_with_streaming(self, file_path: str) -> List[FIXMessage]:
        """
        Parse FIX messages using streaming approach for memory efficiency.
        
        Args:
            file_path: Path to the FIX log file
            
        Returns:
            List of parsed FIX messages
        """
        messages = []
        
        try:
            file_size = os.path.getsize(file_path)
            
            with open(file_path, 'r', errors='ignore') as f:
                last_log_time = time.time()
                last_position = 0
                last_throttle_check = time.time()
                
                for line_num, line in enumerate(f, 1):
                    # Check for resource throttling
                    if time.time() - last_throttle_check > 1.0:  # Check every second
                        last_throttle_check = time.time()
                        if self.resource_monitor.should_throttle():
                            # Sleep to allow system to recover
                            time.sleep(0.5)
                    
                    self.message_count += 1
                    
                    # Only process FIX 4.2 messages
                    if self.fix_version_pattern.search(line):
                        message = self.parse_message(line)
                        if message:
                            messages.append(message)
                            self.valid_message_count += 1
                        else:
                            self.invalid_message_count += 1
                    
                    # Periodically log progress and free memory
                    if self.message_count % 100000 == 0:
                        gc.collect()
                    
                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_log_time > 5.0:  # Log every 5 seconds
                        current_position = f.tell()
                        progress = (current_position / file_size) * 100 if file_size > 0 else 0
                        msgs_per_sec = self.message_count / (current_time - self.start_time)
                        logger.info(f"Parsing progress: {progress:.1f}%, {msgs_per_sec:.1f} msgs/sec")
                        last_log_time = current_time
                        last_position = current_position
                        
        except Exception as e:
            logger.error(f"Error in streaming parsing: {e}")
            raise
        
        return messages
    
    def parse_message(self, message: str) -> Optional[FIXMessage]:
        """
        Parse a single FIX message and extract required fields only.
        
        Args:
            message: Raw FIX message string
            
        Returns:
            FIXMessage object if successfully parsed, None otherwise
        """
        # Extract all tags we care about
        tag_values = dict(self.tag_value_pattern.findall(message))
        
        # Check if this is a message type we care about
        msg_type = tag_values.get("35")
        if msg_type not in (self.NEW_ORDER_SINGLE, self.EXECUTION_REPORT):
            return None
        
        # Check for required fields
        for required_tag in self.REQUIRED_TAGS:
            if required_tag not in tag_values:
                return None
        
        try:
                                            # Create FIXMessage with extracted fields - using float32 for memory efficiency
            try:
                price_value = np.float32(tag_values["44"])
                quantity_value = np.float32(tag_values["38"])
            except ValueError:
                # Handle potential formatting issues in FIX messages
                # Some FIX implementations might use different decimal formats
                price_str = tag_values["44"].replace(',', '.')
                quantity_str = tag_values["38"].replace(',', '.')
                price_value = np.float32(price_str)
                quantity_value = np.float32(quantity_str)
                
            return FIXMessage(
                msg_type=tag_values["35"],
                symbol=tag_values["55"],
                cusip=tag_values["48"],
                quantity=quantity_value,
                price=price_value,
                message_id=f"MSG-{self.message_count}",  # Simple ID for tracking
                order_id=tag_values.get("11"),
                # Store raw message only in debug mode
                raw_message=message if self.debug_mode else None
            )
        
        except (KeyError, ValueError) as e:
            logger.debug(f"Error extracting fields from message: {e}")
            return None
    
    def get_statistics(self, messages: List[FIXMessage]) -> Dict[str, Any]:
        """
        Generate statistics about the parsed messages.
        
        Args:
            messages: List of parsed FIX messages
            
        Returns:
            Dictionary containing statistics about the messages
        """
        if not messages:
            return {"total_messages": 0}
        
        new_order_count = sum(1 for m in messages if m.msg_type == self.NEW_ORDER_SINGLE)
        exec_report_count = sum(1 for m in messages if m.msg_type == self.EXECUTION_REPORT)
        
        # Calculate total exposure (with proper type)
        total_exposure = np.float32(0.0)
        for message in messages:
            total_exposure += message.calculate_exposure()
        
        # Get unique symbols and CUSIPs
        unique_symbols = set(message.symbol for message in messages)
        unique_cusips = set(message.cusip for message in messages)
        
        return {
            "total_messages": len(messages),
            "new_order_singles": new_order_count,
            "execution_reports": exec_report_count,
            "total_exposure": float(total_exposure),  # Convert from np.float32 for compatibility
            "unique_symbols": len(unique_symbols),
            "unique_cusips": len(unique_cusips),
            "parsing_time_seconds": time.time() - self.start_time,
            "peak_cpu": self.resource_monitor.peak_cpu,
            "peak_memory": self.resource_monitor.peak_memory
        }


def process_fix_messages_parallel(file_path: str, num_workers: int = 4, chunk_size_mb: int = 64) -> List[FIXMessage]:
    """
    Process FIX messages in parallel using multiple workers.
    
    This is a placeholder for future enhancement to handle very large FIX logs
    as part of Phase 3 of the performance improvement plan.
    
    The implementation would involve splitting the file into chunks and processing
    each chunk in a separate process, then merging the results.
    
    Args:
        file_path: Path to the FIX log file
        num_workers: Number of parallel workers to use
        chunk_size_mb: Size of each chunk in MB
        
    Returns:
        List of parsed FIX messages
    """
    logger.info(f"Parallel processing scheduled for Phase 3. Using single-threaded parser.")
    parser = FIXParser(chunk_size_mb=chunk_size_mb)
    return parser.parse_file(file_path)


# Command line testing capability
if __name__ == "__main__":
    import argparse
    import sys
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Parse FIX messages from a log file")
    parser.add_argument("file_path", help="Path to the FIX log file")
    parser.add_argument("--delimiter", default="|", help="Delimiter for FIX messages (default: |)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--chunk-size", type=int, default=64, help="Chunk size in MB (default: 64)")
    parser.add_argument("--no-mmap", action="store_true", help="Disable memory mapping")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Parse the file
        fix_parser = FIXParser(
            delimiter=args.delimiter, 
            chunk_size_mb=args.chunk_size,
            enable_mmap=not args.no_mmap,
            debug_mode=args.debug
        )
        
        messages = fix_parser.parse_file(args.file_path)
        
        # Display statistics
        stats = fix_parser.get_statistics(messages)
        print("\nMessage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Success exit
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)