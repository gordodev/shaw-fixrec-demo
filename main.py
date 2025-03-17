#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX Symbol Discrepancy Checker - Performance Optimized

This tool identifies discrepancies between FIX message symbols and the security master
database, with focus on detecting corporate action-related symbol changes that weren't
properly updated in trading systems.

Pushing the performance envelope a bit more before doing rapid QA, then UAT deployment stage
NOTE: This code has been built to be PROD friendly, not actually PROD ready. Trying to see how far we can push it, 
      in a short period of time strictly as a demonstration of my skills and abilities. PROD version of this would 
      take a lot more time and be much more complex, and would need EXTENSIVE testing and QA before deployment. 
      PROD must be done with caution and care, and with a lot of testing. Financial risks are too high to do otherwise.

Key features:
- Memory-efficient processing of extremely large security master files (tested up to 200GB)
- Streaming FIX log parsing to minimize memory usage
- Dynamic resource management to prevent system overload
- Progressive throttling based on system load
- Pre-execution validation to prevent resource exhaustion

Usage:
    python main.py [--secmaster SECMASTER_PATH] [--fix-log FIX_LOG_PATH] 
                  [--output OUTPUT_PATH] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Author: Carlyle
Date: March 17, 2025
"""

import os
import sys
import time
import argparse
import logging
import gc
import psutil
import threading
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union

# Internal modules
from config import load_config, setup_logging
from security_master import load_security_master
from fix_parser import FIXParser
from analyzer import DiscrepancyAnalyzer
from reporter import DiscrepancyReporter


# Constants for resource management
MAX_CPU_PERCENT = 85
MAX_MEMORY_PERCENT = 90
THROTTLE_CPU_PERCENT = 70
THROTTLE_SLEEP_BASE = 0.05  # Base sleep time in seconds
FILE_SIZE_WARNING_GB = 50
CRITICAL_FILE_SIZE_GB = 150
MIN_FREE_DISK_SPACE_GB = 5
DEFAULT_CHUNK_SIZE_MB = 64

def load_small_secmaster(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Simple and direct loader for small security master files.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Dictionary mapping CUSIPs to security details
    """
    import csv
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"Using fast path loader for small file: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"Security master file not found: {file_path}")
        raise FileNotFoundError(f"Security master file not found: {file_path}")
    
    # Check file size
    file_size = os.path.getsize(file_path)
    file_size_mb = file_size / (1024 * 1024)
    logger.info(f"Security master file size: {file_size_mb:.2f} MB")
    
    # Direct CSV parsing for small files
    security_data = {}
    
    try:
        with open(file_path, 'r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            
            # Check that required columns exist
            if 'CUSIP' not in reader.fieldnames or 'Symbol' not in reader.fieldnames:
                logger.error("Security master CSV is missing required columns (CUSIP, Symbol)")
                raise ValueError("Security master CSV is missing required columns")
                
            for row in reader:
                cusip = row['CUSIP']
                if cusip:
                    # Only keep essential fields for analysis
                    security_data[cusip] = {
                        'Symbol': row['Symbol'],
                        'Region': row.get('Region', ''),
                        'Exchange': row.get('Exchange', '')
                    }
        
        logger.info(f"Successfully loaded {len(security_data)} securities")
        return security_data
        
    except Exception as e:
        logger.error(f"Error loading security master file: {e}")
        raise

class ResourceMonitor:
    """Monitor system resources during processing."""
    
    def __init__(self, 
                max_cpu_percent: int = MAX_CPU_PERCENT,
                max_memory_percent: int = MAX_MEMORY_PERCENT,
                check_interval: float = 1.0):
        """
        Initialize resource monitor.
        
        Args:
            max_cpu_percent: Maximum allowed CPU usage percentage
            max_memory_percent: Maximum allowed memory usage percentage
            check_interval: How often to check resources (seconds)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.stop_flag = threading.Event()
        self.monitor_thread = None
        self.critical_resources = False
        self.cpu_throttle_needed = False
        self.peak_cpu_percent = 0
        self.peak_memory_percent = 0
        
    def start(self):
        """Start resource monitoring."""
        if self.monitor_thread is not None:
            return
            
        self.stop_flag.clear()
        self.critical_resources = False
        self.cpu_throttle_needed = False
        self.peak_cpu_percent = 0
        self.peak_memory_percent = 0
        
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
        return self.cpu_throttle_needed
        
    def has_critical_resources(self) -> bool:
        """Check if resources have reached critical levels."""
        return self.critical_resources
        
    def get_peak_usage(self) -> Tuple[float, float]:
        """
        Get peak resource usage during monitoring.
        
        Returns:
            Tuple of (peak CPU percentage, peak memory percentage)
        """
        return (self.peak_cpu_percent, self.peak_memory_percent)
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while not self.stop_flag.is_set():
            try:
                # Get CPU usage (average across all cores)
                cpu_percent = psutil.cpu_percent(interval=0.1)
                
                # Get memory usage
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                
                # Update peak values
                self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
                self.peak_memory_percent = max(self.peak_memory_percent, memory_percent)
                
                # Check thresholds
                if cpu_percent >= self.max_cpu_percent or memory_percent >= self.max_memory_percent:
                    self.critical_resources = True
                    logging.warning(
                        f"Critical resource usage: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
                    )
                    # Force garbage collection
                    gc.collect()
                
                # Check if throttling is needed
                self.cpu_throttle_needed = cpu_percent >= THROTTLE_CPU_PERCENT
                
                if self.cpu_throttle_needed and cpu_percent >= THROTTLE_CPU_PERCENT + 10:
                    logging.warning(f"CPU usage high: {cpu_percent:.1f}%, throttling enabled")
                
            except Exception as e:
                logging.error(f"Error in resource monitor: {e}")
                
            time.sleep(self.check_interval)


def throttle_processing(monitor: ResourceMonitor, base_delay: float = THROTTLE_SLEEP_BASE):
    """
    Dynamically throttle processing based on resource usage.
    
    Args:
        monitor: Resource monitor instance
        base_delay: Base delay time in seconds
    """
    if monitor.should_throttle():
        # Apply progressive backoff - higher CPU usage means longer sleep
        cpu_percent = psutil.cpu_percent(interval=0.05)
        
        # Calculate sleep time based on CPU usage
        # More aggressive throttling as CPU usage increases
        cpu_factor = max(0, (cpu_percent - THROTTLE_CPU_PERCENT) / (100 - THROTTLE_CPU_PERCENT))
        sleep_time = base_delay * (1 + 3 * cpu_factor)  # Up to 4x base delay at 100% CPU
        
        # Add some jitter to prevent synchronized access patterns
        jitter = random.uniform(0, 0.5 * sleep_time)
        
        time.sleep(sleep_time + jitter)


def check_disk_space(path: str, min_free_gb: float = MIN_FREE_DISK_SPACE_GB) -> Tuple[bool, float]:
    """
    Check if there's enough free disk space.
    
    Args:
        path: Path to check
        min_free_gb: Minimum required free space in GB
        
    Returns:
        Tuple of (has_enough_space, free_space_gb)
    """
    try:
        # Ensure we're checking a directory path, not a file path
        if os.path.isfile(path):
            path_dir = os.path.dirname(path)
            # If dirname returns empty string, use current directory
            if not path_dir:
                path_dir = '.'
        else:
            path_dir = path
            
        disk_usage = psutil.disk_usage(path_dir)
        free_space_gb = disk_usage.free / (1024**3)
        
        return (free_space_gb >= min_free_gb, free_space_gb)
    except Exception as e:
        logging.error(f"Error checking disk space: {e}")
        # Return True as a fallback to prevent blocking execution incorrectly
        # In production, you might want different behavior
        return (True, min_free_gb + 1)


def validate_file_size(file_path: str) -> Tuple[bool, float, str]:
    """
    Validate file size and provide appropriate warnings.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (is_size_reasonable, size_gb, message)
    """
    try:
        if not os.path.exists(file_path):
            return (False, 0, f"File not found: {file_path}")
            
        # Get file size
        file_size = os.path.getsize(file_path)
        size_gb = file_size / (1024**3)
        
        # Check free disk space
        # Get the directory containing the file
        file_dir = os.path.dirname(file_path)
        if not file_dir:
            file_dir = '.'
            
        has_space, free_space_gb = check_disk_space(file_dir)
        if not has_space:
            return (False, size_gb, f"Not enough disk space: {free_space_gb:.2f}GB free, need at least {MIN_FREE_DISK_SPACE_GB}GB")
        
        # Check system memory
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Determine if file size is reasonable
        if size_gb > CRITICAL_FILE_SIZE_GB:
            return (False, size_gb, f"File size ({size_gb:.2f}GB) exceeds critical threshold ({CRITICAL_FILE_SIZE_GB}GB)")
        elif size_gb > FILE_SIZE_WARNING_GB:
            message = (f"Warning: Large file detected ({size_gb:.2f}GB). "
                      f"This may require significant system resources. "
                      f"System has {system_memory_gb:.2f}GB RAM, {free_space_gb:.2f}GB free disk space.")
            return (True, size_gb, message)
        
        return (True, size_gb, "")
    except Exception as e:
        logging.error(f"Error validating file size: {e}")
        # Return True as a fallback to prevent blocking execution incorrectly
        return (True, 0, f"Error checking file size, proceeding anyway: {e}")


def dynamic_chunk_size(file_size_gb: float, system_memory_gb: float) -> int:
    """
    Calculate optimal chunk size based on file size and available memory.
    
    Args:
        file_size_gb: File size in GB
        system_memory_gb: System memory in GB
        
    Returns:
        Chunk size in MB
    """
    # Base chunk size
    chunk_size_mb = DEFAULT_CHUNK_SIZE_MB
    
    # Adjust based on file size relative to system memory
    if file_size_gb > system_memory_gb * 2:
        # Very large file compared to memory - use smaller chunks
        chunk_size_mb = max(16, int(chunk_size_mb / 2))
    elif file_size_gb < system_memory_gb / 4:
        # Small file compared to memory - use larger chunks
        chunk_size_mb = min(256, int(chunk_size_mb * 2))
    
    # Ensure chunk size is reasonable
    return max(16, min(chunk_size_mb, 256))


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FIX Symbol Discrepancy Checker")
    
    parser.add_argument("--secmaster", 
                       help="Path to security master CSV file")
    
    parser.add_argument("--fix-log", 
                       help="Path to FIX message log file")
    
    parser.add_argument("--output", 
                       help="Path to output report file")
    
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    parser.add_argument("--delimiter", 
                       help="FIX message field delimiter (default: |)")
    
    # Advanced memory management options
    parser.add_argument("--chunk-size", 
                       type=int, default=0,  # 0 means auto-calculate
                       help="Security master chunk size in MB (default: auto)")
    
    parser.add_argument("--disable-mmap", 
                       action="store_true",
                       help="Disable memory mapping for security master loading")
    
    parser.add_argument("--max-cpu", 
                       type=int, default=MAX_CPU_PERCENT,
                       help=f"Maximum CPU usage percentage (default: {MAX_CPU_PERCENT})")
    
    parser.add_argument("--max-memory", 
                       type=int, default=MAX_MEMORY_PERCENT,
                       help=f"Maximum memory usage percentage (default: {MAX_MEMORY_PERCENT})")
    
    parser.add_argument("--force", 
                       action="store_true",
                       help="Force execution even if file size exceeds critical threshold")
    
    return parser.parse_args()


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


def parse_fix_log(fix_log_path: str, 
                 delimiter: str = "|",
                 resource_monitor: Optional[ResourceMonitor] = None) -> List[Dict[str, Any]]:
    """
    Parse the FIX message log file with resource-aware processing.
    
    Args:
        fix_log_path: Path to FIX log file
        delimiter: Character used to separate FIX fields (default: |)
        resource_monitor: Resource monitor instance for throttling
        
    Returns:
        List of parsed FIX messages
        
    Raises:
        FileNotFoundError: If FIX log file doesn't exist
    """
    print ("Starting to load FIX")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing FIX messages from: {fix_log_path}")
    
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss
    
    # Create parser with standard settings
    fix_parser = FIXParser(delimiter=delimiter)
    
    # Parse with resource monitoring
    messages = []
    try:
        # Check if file exists and get size
        if not os.path.exists(fix_log_path):
            raise FileNotFoundError(f"FIX log file not found: {fix_log_path}")
            
        file_size = os.path.getsize(fix_log_path)
        logger.info(f"FIX log size: {file_size / (1024**2):.2f} MB")
        
        # Parse messages in a resource-aware manner
        messages = fix_parser.parse_file(fix_log_path)
        
        # Check if resource monitor indicates we should abort
        if resource_monitor and resource_monitor.has_critical_resources():
            logger.warning("Aborting FIX parsing due to critical resource usage")
            raise RuntimeError("Critical resource usage during FIX parsing")
    
    except Exception as e:
        if not isinstance(e, FileNotFoundError):
            logger.error(f"Error parsing FIX log: {e}")
        raise
    
    elapsed_time = time.time() - start_time
    current_memory = psutil.Process(os.getpid()).memory_info().rss
    memory_increase = current_memory - start_memory
    
    stats = fix_parser.get_statistics(messages)
    
    logger.info(f"Parsed {stats['total_messages']} messages in {elapsed_time:.2f} seconds: "
               f"{stats['new_order_singles']} new orders, "
               f"{stats['execution_reports']} execution reports")
    logger.info(f"Memory usage for FIX parsing: {memory_increase / (1024**2):.1f} MB")
    
    # Convert to dicts for analyzer
    message_dicts = []
    for msg in messages:
        msg_dict = {
            'msg_type': msg.msg_type,
            'Symbol': msg.symbol,
            'CUSIP': msg.cusip,
            'Quantity': msg.quantity,
            'Price': msg.price,
            'message_id': msg.message_id,
            'order_id': msg.order_id
        }
        message_dicts.append(msg_dict)
    
    # Clear the original messages list to free memory
    messages.clear()
    gc.collect()
    
    return message_dicts


def analyze_fix_messages(fix_messages: List[Dict[str, Any]], 
                        security_master: Dict[str, Dict[str, Any]],
                        resource_monitor: Optional[ResourceMonitor] = None,
                        batch_size: int = 10000) -> Tuple[List[Dict[str, Any]], float]:
    """
    Analyze FIX messages to identify discrepancies with security master.
    
    Args:
        fix_messages: List of parsed FIX messages
        security_master: Security master data indexed by CUSIP
        resource_monitor: Resource monitor instance for throttling
        batch_size: Number of messages to process per batch
        
    Returns:
        Tuple of (discrepancies, total_exposure)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Analyzing {len(fix_messages):,} FIX messages in batches of {batch_size:,}")
    
    start_time = time.time()
    
    # Process in batches to manage resource usage
    analyzer = DiscrepancyAnalyzer(security_master)
    
    # If fewer than batch_size messages, just analyze directly
    if len(fix_messages) <= batch_size:
        discrepancies, total_exposure = analyzer.analyze_messages(fix_messages)
    else:
        # Process in batches
        all_discrepancies = []
        total_exposure = 0.0
        
        for i in range(0, len(fix_messages), batch_size):
            # Check if we should abort due to resource constraints
            if resource_monitor and resource_monitor.has_critical_resources():
                logger.warning("Aborting analysis due to critical resource usage")
                raise RuntimeError("Critical resource usage during analysis")
            
            # Apply throttling if needed
            if resource_monitor and resource_monitor.should_throttle():
                throttle_processing(resource_monitor)
            
            # Process batch
            batch = fix_messages[i:i+batch_size]
            batch_discrepancies, batch_exposure = analyzer.analyze_messages(batch)
            
            # Accumulate results
            all_discrepancies.extend(batch_discrepancies)
            total_exposure += batch_exposure
            
            # Log progress
            if (i + batch_size) % (batch_size * 10) == 0 or (i + batch_size) >= len(fix_messages):
                progress = min(100, 100 * (i + batch_size) / len(fix_messages))
                logger.info(f"Analysis progress: {progress:.1f}%, found {len(all_discrepancies)} discrepancies so far")
                
            # Trigger garbage collection after each batch
            gc.collect()
        
        # Sort combined discrepancies by exposure (highest first)
        discrepancies = sorted(
            all_discrepancies, 
            key=lambda x: x.get('Exposure', 0), 
            reverse=True
        )
    
    elapsed_time = time.time() - start_time
    
    logger.info(f"Analysis complete in {elapsed_time:.2f} seconds. Found {len(discrepancies)} discrepancies "
               f"with total exposure of ${total_exposure:,.2f}")
    
    # Log top discrepancies by exposure
    if discrepancies:
        top_disc = discrepancies[0]
        logger.info(f"Highest exposure discrepancy: "
                   f"CUSIP={top_disc['CUSIP']}, "
                   f"FIX Symbol={top_disc['FIXSymbol']}, "
                   f"Master Symbol={top_disc['MasterSymbol']}, "
                   f"Exposure=${top_disc['Exposure']:,.2f}")
    
    # Get region metrics if available
    region_metrics = analyzer.get_region_metrics()
    if region_metrics:
        for region, metrics in region_metrics.items():
            logger.debug(f"Region {region}: {metrics['count']} discrepancies, "
                        f"${metrics['exposure']:,.2f} exposure")
    
    return discrepancies, total_exposure


def generate_report(discrepancies: List[Dict[str, Any]], 
                   total_exposure: float,
                   output_path: str,
                   resource_monitor: Optional[ResourceMonitor] = None) -> None:
    """
    Generate a report of the identified discrepancies.
    
    Args:
        discrepancies: List of discrepancies
        total_exposure: Total financial exposure
        output_path: Path to output report file
        resource_monitor: Resource monitor instance for throttling
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating discrepancy report at: {output_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            logger.debug(f"Created output directory: {output_dir}")
        except Exception as e:
            logger.error(f"Error creating output directory: {e}")
            raise
    
    # Check disk space before writing
    has_space, free_space_gb = check_disk_space(output_path)
    if not has_space:
        logger.error(f"Not enough disk space to write report: {free_space_gb:.2f}GB free")
        raise RuntimeError(f"Not enough disk space to write report: {free_space_gb:.2f}GB free")
    
    # Apply throttling if needed
    if resource_monitor and resource_monitor.should_throttle():
        throttle_processing(resource_monitor)
    
    reporter = DiscrepancyReporter(output_path)
    reporter.generate_report(discrepancies, total_exposure)
    
    logger.info(f"Report generated successfully: {output_path}")


def pre_execution_validation(config: Dict[str, Any]) -> bool:
    """
    Perform pre-execution validation to ensure we can safely proceed.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if validation passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    logger.info("Performing pre-execution validation")
    
    # Check disk space for output directory
    output_dir = os.path.dirname(config['output_path'])
    if not output_dir:
        output_dir = '.'
    
    has_space, free_space_gb = check_disk_space(output_dir)
    if not has_space:
        logger.error(f"Not enough disk space for output: {free_space_gb:.2f}GB free")
        return False
    
    # Validate security master file size
    secmaster_valid, secmaster_size_gb, secmaster_message = validate_file_size(config['secmaster_path'])
    
    if not secmaster_valid and not config.get('force', False):
        logger.error(f"Security master validation failed: {secmaster_message}")
        return False
    elif secmaster_message:
        logger.warning(secmaster_message)
    
    # Validate FIX log file size
    fixlog_valid, fixlog_size_gb, fixlog_message = validate_file_size(config['fix_log_path'])
    
    if not fixlog_valid and not config.get('force', False):
        logger.error(f"FIX log validation failed: {fixlog_message}")
        return False
    elif fixlog_message:
        logger.warning(fixlog_message)
    
    # Check system memory
    system_memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Show combined resource requirements
    total_size_gb = secmaster_size_gb + fixlog_size_gb
    logger.info(f"Total input size: {total_size_gb:.2f}GB")
    logger.info(f"System memory: {system_memory_gb:.2f}GB")
    logger.info(f"Free disk space: {free_space_gb:.2f}GB")
    
    # Warn if combined size is large relative to system memory
    if total_size_gb > system_memory_gb and total_size_gb > 10:
        logger.warning(
            f"Combined input size ({total_size_gb:.2f}GB) exceeds system memory ({system_memory_gb:.2f}GB). "
            f"Processing may be slow and resource-intensive."
        )
    
    # Calculate optimal chunk size if not specified
    if config.get('chunk_size', 0) == 0:
        chunk_size = dynamic_chunk_size(secmaster_size_gb, system_memory_gb)
        logger.info(f"Auto-calculated chunk size: {chunk_size}MB")
        config['chunk_size'] = chunk_size
    
    return True


def run(config: Dict[str, Any]) -> int:
    """
    Main execution flow of the reconciliation tool.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    logger = logging.getLogger(__name__)
    start_time = time.time()
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(
        max_cpu_percent=config.get('max_cpu', MAX_CPU_PERCENT),
        max_memory_percent=config.get('max_memory', MAX_MEMORY_PERCENT)
    )
    resource_monitor.start()
    
    try:
        # Perform pre-execution validation
        if not pre_execution_validation(config):
            logger.error("Pre-execution validation failed. Aborting.")
            return 4  # Validation error
        
        # Add a small delay to stagger startup (helps prevent immediate overload)
        time.sleep(0.5)
        
        # Step 1: Load security master with memory-efficient chunked loading
        security_master = load_security_master_data(
            config['secmaster_path'],
            chunk_size_mb=config.get('chunk_size', DEFAULT_CHUNK_SIZE_MB),
            use_mmap=not config.get('disable_mmap', False),
            resource_monitor=resource_monitor
        )
        
        # Force garbage collection after loading security master
        gc.collect()
        
        # Step 2: Parse FIX log
        print ("Starting to parse FIX")
        fix_messages = parse_fix_log(
            config['fix_log_path'], 
            config['fix_delimiter'],
            resource_monitor=resource_monitor
        )
        print ("resource_monitor next")

        # Step 3: Analyze messages for discrepancies
        discrepancies, total_exposure = analyze_fix_messages(
            fix_messages, 
            security_master,
            resource_monitor=resource_monitor
        )
        
        # Step 4: Generate report
        generate_report(
            discrepancies, 
            total_exposure, 
            config['output_path'],
            resource_monitor=resource_monitor
        )
        
        # Log summary
        elapsed_time = time.time() - start_time
        logger.info(f"Completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Found {len(discrepancies)} discrepancies with total exposure of ${total_exposure:,.2f}")
        
        # Log resource usage
        peak_cpu, peak_memory = resource_monitor.get_peak_usage()
        logger.info(f"Peak resource usage: CPU {peak_cpu:.1f}%, Memory {peak_memory:.1f}%")
        
        # Log final memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        logger.info(f"Final memory usage: {memory_usage / (1024**2):.1f} MB")
        
        return 0  # Success
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2  # File access error
        
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return 3  # Processing error
        
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return 5  # Runtime error
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1  # General error
    
    finally:
        # Stop resource monitor
        if resource_monitor:
            resource_monitor.stop()
        
        # Force garbage collection at the end
        gc.collect()
        logger.info("FIX Symbol Discrepancy Checker completed")


def configure() -> Dict[str, Any]:
    """
    Set up configuration from command line args and config file.
    
    Returns:
        Configuration dictionary
    """
    # Load default configuration
    config = load_config()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Override config with command line args if provided
    if args.secmaster:
        config['secmaster_path'] = args.secmaster
    
    if args.fix_log:
        config['fix_log_path'] = args.fix_log
    
    if args.output:
        config['output_path'] = args.output
    
    if args.log_level:
        config['log_level'] = args.log_level
    
    if args.delimiter:
        config['fix_delimiter'] = args.delimiter
    
    # Additional memory management options
    if args.chunk_size:
        config['chunk_size'] = args.chunk_size
    
    if args.disable_mmap:
        config['disable_mmap'] = True
    
    # Resource management options
    if args.max_cpu:
        config['max_cpu'] = args.max_cpu
    
    if args.max_memory:
        config['max_memory'] = args.max_memory
    
    if args.force:
        config['force'] = True
    
    # Set up logging with configured level
    setup_logging(config['log_level'])
    
    return config


def main() -> int:
    """
    Entry point for the FIX Symbol Discrepancy Checker.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    print(f"FIX Symbol Discrepancy Checker - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Configure the application
        config = configure()
        
        # Validate required configuration settings
        for required in ['secmaster_path', 'fix_log_path', 'output_path']:
            if not config.get(required):
                print(f"Error: Missing required configuration: {required}")
                return 1
        
        # Display configuration for audit
        print(f"Security Master: {config['secmaster_path']}")
        print(f"FIX Log File: {config['fix_log_path']}")
        print(f"Output Report: {config['output_path']}")
        print(f"Log Level: {config['log_level']}")
        
        if 'chunk_size' in config and config['chunk_size'] > 0:
            print(f"Security Master Chunk Size: {config['chunk_size']} MB")
        else:
            print("Security Master Chunk Size: Auto")
        
        if config.get('disable_mmap', False):
            print("Memory Mapping: Disabled")
        else:
            print("Memory Mapping: Enabled")
            
        print(f"Max CPU Usage: {config.get('max_cpu', MAX_CPU_PERCENT)}%")
        print(f"Max Memory Usage: {config.get('max_memory', MAX_MEMORY_PERCENT)}%")
        
        if config.get('force', False):
            print("Warning: Force option enabled - will attempt to process regardless of file size")
        
        print("-" * 50)
        
        # Run the main process
        return run(config)
        
    except Exception as e:
        print(f"Error in configuration: {e}")
        return 1


if __name__ == "__main__":
    # Set the exit code from the main function
    sys.exit(main())