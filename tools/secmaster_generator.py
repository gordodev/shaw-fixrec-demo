#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Master File Generator for Testing

This script generates large security master files for testing file handling,
performance, and error recovery in trading applications. It can create files
of various sizes from 1GB to 200GB to stress test system capabilities.

The generator focuses on I/O efficiency and memory management to produce
large files quickly without exhausting system resources.

Features:
- Continuous disk space monitoring during generation
- Graceful error handling for "out of disk space" scenarios
- Size presets for common testing needs
- Detailed progress reporting
- Resource usage monitoring

Author: Carlyle
Date: March 15, 2025
"""

import os
import sys
import time
import argparse
import psutil
import threading
import random
import signal
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import multiprocessing as mp
from datetime import datetime


# Predefined sizes for testing
SIZE_PRESETS = {
    "small": 1,       # 1GB
    "medium": 5,      # 5GB
    "large": 10,      # 10GB
    "xlarge": 20,     # 20GB
    "extreme": 200,   # 200GB
}

# Template data for efficient generation
REGIONS = ["US", "EUROPE", "ASIA", "LATAM", "MENA", "CANADA"]
EXCHANGES = ["NYSE", "NASDAQ", "LSE", "TSE", "HKEX", "EURONEXT", "SSE", "BSE"]
ASSET_CLASSES = ["Equity", "Bond", "Option", "Future", "ETF", "Index", "Commodity"]
CURRENCIES = ["USD", "EUR", "JPY", "GBP", "CHF", "CAD", "AUD", "CNY"]
COUNTRIES = ["USA", "UK", "Japan", "Germany", "France", "Canada", "Australia", "China", "Brazil"]

# Constants for disk space monitoring
DISK_CHECK_INTERVAL = 5  # seconds
MIN_REQUIRED_SPACE_GB = 1.0  # Minimum space required to continue
DISK_SPACE_BUFFER_PERCENT = 10  # Keep extra 10% space free


@dataclass
class GenerationConfig:
    """Configuration for file generation."""
    output_file: str
    target_size_gb: float
    template_count: int = 10
    batch_size_mb: int = 500
    use_unique_values: bool = False
    show_progress: bool = True
    buffer_margin_gb: float = 2.0
    row_size_bytes: int = 150
    cpu_cores: int = 1
    safety_margin_percent: int = 10  # % of free space to maintain
    

def human_readable_size(size_bytes: int) -> str:
    """Convert bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024 or unit == 'TB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024


def get_free_disk_space(path: str) -> int:
    """
    Get free disk space in bytes for the given path.
    
    Args:
        path: Directory path to check
        
    Returns:
        Free space in bytes
    """
    try:
        if not os.path.exists(path):
            path = os.path.dirname(path)
        if not path:
            path = '.'
            
        disk_usage = psutil.disk_usage(path)
        return disk_usage.free
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return 0


def check_disk_space(file_path: str, required_gb: float, buffer_gb: float = 5.0) -> bool:
    """
    Check if there's enough disk space for the operation.
    
    Args:
        file_path: Path where the file will be written
        required_gb: Minimum required space in GB
        buffer_gb: Additional buffer space in GB
        
    Returns:
        True if there's enough space, False otherwise
    """
    required_bytes = int((required_gb + buffer_gb) * 1024 * 1024 * 1024)
    
    try:
        path = os.path.dirname(os.path.abspath(file_path))
        if not path:
            path = '.'
            
        disk_usage = psutil.disk_usage(path)
        
        if disk_usage.free < required_bytes:
            print(f"WARNING: Insufficient disk space! Need {required_gb + buffer_gb:.1f}GB, "
                  f"but only {disk_usage.free / (1024**3):.1f}GB available.")
            return False
            
        # Also check if the target file is too large relative to total disk size
        total_disk_gb = disk_usage.total / (1024**3)
        safe_limit_gb = total_disk_gb * 0.85  # Don't use more than 85% of total disk
        
        if required_gb > safe_limit_gb:
            print(f"WARNING: Target file size ({required_gb:.1f}GB) is too large for this disk "
                  f"({total_disk_gb:.1f}GB). Maximum recommended size: {safe_limit_gb:.1f}GB.")
            return False
            
        return True
    except Exception as e:
        print(f"Error checking disk space: {e}")
        return False


def check_free_space_percentage(path: str, min_percent: int = 10) -> bool:
    """
    Check if disk has at least the minimum percentage of free space.
    
    Args:
        path: Path to check
        min_percent: Minimum percentage of free space required
        
    Returns:
        True if there's enough space, False otherwise
    """
    try:
        if not os.path.exists(path):
            path = os.path.dirname(path)
        if not path:
            path = '.'
            
        disk_usage = psutil.disk_usage(path)
        free_percent = 100 - disk_usage.percent
        
        return free_percent >= min_percent
    except Exception as e:
        print(f"Error checking disk space percentage: {e}")
        return False


def generate_symbols(count: int) -> List[str]:
    """Generate a list of unique stock symbols."""
    symbols = []
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Start with single letter symbols
    for c in chars:
        symbols.append(c)
    
    # Add two letter symbols
    for c1 in chars:
        for c2 in chars:
            symbols.append(c1 + c2)
    
    # Add three letter symbols if needed
    if count > len(symbols):
        for c1 in chars:
            for c2 in chars:
                for c3 in chars[:10]:  # Limit to first 10 letters for third position
                    symbols.append(c1 + c2 + c3)
                    if len(symbols) >= count:
                        return symbols[:count]
    
    return symbols[:count]


def generate_cusips(count: int) -> List[str]:
    """Generate a list of plausible CUSIP identifiers."""
    cusips = []
    # Real CUSIPs have a check digit algorithm, but for testing we'll use a simplified approach
    for i in range(count):
        # Format: NNNNNNNN + check digit (9 chars total)
        base = f"{i:08d}"
        check_digit = str(sum(int(d) for d in base) % 10)  # Simple check digit
        cusips.append(base + check_digit)
    
    return cusips


def create_row_templates(count: int, use_unique: bool = False) -> List[str]:
    """Create a set of CSV row templates for security master records."""
    templates = []
    
    for i in range(count):
        if use_unique:
            # Create slightly varied templates
            region = random.choice(REGIONS)
            exchange = random.choice(EXCHANGES)
            asset_class = random.choice(ASSET_CLASSES)
            currency = random.choice(CURRENCIES)
            country = random.choice(COUNTRIES)
            description = f"Security {i+1}"
        else:
            # Use fixed values for faster generation
            region = REGIONS[i % len(REGIONS)]
            exchange = EXCHANGES[i % len(EXCHANGES)]
            asset_class = ASSET_CLASSES[i % len(ASSET_CLASSES)]
            currency = CURRENCIES[i % len(CURRENCIES)]
            country = COUNTRIES[i % len(COUNTRIES)]
            description = f"Security Template {i+1}"
        
        # Create template with placeholders for symbol and CUSIP
        template = f"{{symbol}},{{cusip}},{description},{region},{exchange},{asset_class},{currency},{country}\n"
        templates.append(template)
    
    return templates


class DiskSpaceMonitor:
    """
    Monitor available disk space during file generation.
    
    This class runs in a separate thread and periodically checks if there's
    enough disk space to continue the operation.
    """
    
    def __init__(self, file_path: str, min_free_gb: float = 1.0, min_free_percent: int = 10):
        """
        Initialize the disk space monitor.
        
        Args:
            file_path: Path to the file being generated
            min_free_gb: Minimum free space required in GB
            min_free_percent: Minimum percentage of free space required
        """
        self.file_path = file_path
        self.min_free_bytes = int(min_free_gb * 1024 * 1024 * 1024)
        self.min_free_percent = min_free_percent
        self.stop_event = threading.Event()
        self.disk_space_critical = threading.Event()
        self._thread = None
        
    def start(self):
        """Start monitoring disk space in a separate thread."""
        self._thread = threading.Thread(target=self._monitor_loop)
        self._thread.daemon = True
        self._thread.start()
        
    def stop(self):
        """Stop the monitoring thread."""
        self.stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        
    def has_critical_space_issue(self) -> bool:
        """Check if a critical disk space issue has been detected."""
        return self.disk_space_critical.is_set()
        
    def _monitor_loop(self):
        """Main monitoring loop that runs in a separate thread."""
        while not self.stop_event.is_set():
            try:
                # Check free space
                free_bytes = get_free_disk_space(self.file_path)
                
                if free_bytes < self.min_free_bytes:
                    print(f"\nWARNING: Low disk space! Only {human_readable_size(free_bytes)} available.")
                    self.disk_space_critical.set()
                
                # Also check percentage
                if not check_free_space_percentage(self.file_path, self.min_free_percent):
                    print(f"\nWARNING: Low disk space percentage! Less than {self.min_free_percent}% free.")
                    self.disk_space_critical.set()
                    
            except Exception as e:
                print(f"\nError in disk space monitor: {e}")
                
            # Sleep for a bit before checking again
            time.sleep(DISK_CHECK_INTERVAL)


def monitor_resources(stop_event, file_path, target_size_gb, disk_monitor, interval=1.0):
    """Monitor and display resource usage during file generation."""
    target_bytes = int(target_size_gb * 1024 * 1024 * 1024)
    start_time = time.time()
    
    while not stop_event.is_set():
        try:
            # Check if disk space is critical
            if disk_monitor.has_critical_space_issue():
                print("\nOperation paused due to disk space issues!")
                stop_event.set()
                break
                
            # RAM usage
            mem = psutil.virtual_memory()
            used_ram = mem.total - mem.available
            ram_percent = mem.percent
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk space
            free_space = get_free_disk_space(file_path)
            free_space_gb = free_space / (1024**3)
            
            # Disk usage
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                progress = (file_size / target_bytes) * 100
                
                # Estimate time remaining
                elapsed = time.time() - start_time
                if progress > 0:
                    total_estimated = elapsed * (100 / progress)
                    remaining = total_estimated - elapsed
                    eta_str = f"ETA: {int(remaining/60)}m {int(remaining%60)}s"
                else:
                    eta_str = "ETA: calculating..."
                
                # Calculate write speed
                if elapsed > 0:
                    write_speed = file_size / elapsed / (1024 * 1024)  # MB/s
                    speed_str = f"{write_speed:.1f} MB/s"
                else:
                    speed_str = "calculating..."
                
                print(f"\rProgress: {progress:.1f}% | File: {human_readable_size(file_size)} | "
                      f"Free: {free_space_gb:.1f}GB | RAM: {ram_percent:.1f}% | CPU: {cpu_percent:.1f}% | "
                      f"Speed: {speed_str} | {eta_str}", end="")
            else:
                print(f"\rPreparing to write... | Free: {free_space_gb:.1f}GB | "
                      f"RAM: {ram_percent:.1f}% | CPU: {cpu_percent:.1f}%", end="")
        except Exception as e:
            print(f"\rMonitoring error: {e}", end="")
        
        time.sleep(interval)
    
    print()  # Final newline


def generate_batch(config: GenerationConfig, batch_index: int) -> bytes:
    """Generate a batch of security master rows."""
    row_count = int((config.batch_size_mb * 1024 * 1024) / config.row_size_bytes)
    
    # Get or create templates
    templates = create_row_templates(config.template_count, config.use_unique_values)
    
    # Generate unique identifiers if needed
    base_index = batch_index * row_count
    if config.use_unique_values:
        symbols = generate_symbols(config.template_count)
        cusips = generate_cusips(config.template_count)
        
        # Generate batch data
        batch_data = b""
        for i in range(row_count):
            template_idx = (base_index + i) % len(templates)
            symbol_idx = (base_index + i) % len(symbols)
            cusip_idx = (base_index + i) % len(cusips)
            
            row = templates[template_idx].format(
                symbol=symbols[symbol_idx], 
                cusip=cusips[cusip_idx]
            )
            batch_data += row.encode('utf-8')
    else:
        # Use a repeated template for speed
        template = templates[0].format(symbol="AAPL", cusip="037833100")
        batch_data = template.encode('utf-8') * row_count
    
    return batch_data


class DiskSpaceError(Exception):
    """Exception raised when disk space becomes critically low during operation."""
    pass


def generate_in_chunks(config: GenerationConfig):
    """Generate the security master file in manageable chunks."""
    target_bytes = int(config.target_size_gb * 1024 * 1024 * 1024)
    
    # Create header row
    header = "Symbol,CUSIP,Description,Region,Exchange,AssetClass,Currency,Country\n"
    
    # Calculate number of batches
    batch_size_bytes = config.batch_size_mb * 1024 * 1024
    batch_count = (target_bytes - len(header.encode('utf-8'))) // batch_size_bytes
    if batch_count <= 0:
        batch_count = 1
    
    # Set up disk space monitoring
    disk_monitor = DiskSpaceMonitor(
        config.output_file, 
        min_free_gb=MIN_REQUIRED_SPACE_GB,
        min_free_percent=config.safety_margin_percent
    )
    disk_monitor.start()
    
    # Start monitoring in a separate thread
    stop_monitor = threading.Event()
    monitor_thread = threading.Thread(
        target=monitor_resources, 
        args=(stop_monitor, config.output_file, config.target_size_gb, disk_monitor)
    )
    monitor_thread.daemon = True
    
    try:
        # Create output file and write header
        with open(config.output_file, 'wb') as f:
            f.write(header.encode('utf-8'))
        
        # Only start monitoring thread if progress display is enabled
        if config.show_progress:
            monitor_thread.start()
        
        # Generate and write data in batches
        for i in range(batch_count):
            # Check if disk space monitor has detected a critical issue
            if disk_monitor.has_critical_space_issue():
                raise DiskSpaceError("Operation aborted due to critical disk space issues")
                
            # Check disk space before generating each batch
            if i % 5 == 0:  # Only check every 5 batches for performance
                free_space = get_free_disk_space(config.output_file)
                free_space_gb = free_space / (1024**3)
                
                if free_space_gb < MIN_REQUIRED_SPACE_GB:
                    raise DiskSpaceError(
                        f"Disk space critically low: {free_space_gb:.2f}GB free. "
                        f"Minimum required: {MIN_REQUIRED_SPACE_GB}GB."
                    )
            
            batch_data = generate_batch(config, i)
            
            # Check disk space again before writing
            free_space = get_free_disk_space(config.output_file)
            if len(batch_data) > free_space * (1 - config.safety_margin_percent/100):
                raise DiskSpaceError(
                    f"Not enough disk space to write batch. Batch size: {len(batch_data)}, "
                    f"Free space: {free_space} bytes."
                )
            
            # Write batch to file
            try:
                with open(config.output_file, 'ab') as f:
                    f.write(batch_data)
            except OSError as e:
                if e.errno == 28:  # No space left on device
                    raise DiskSpaceError(f"No space left on device: {e}")
                raise
            
            # Manually control progress updates if monitoring is disabled
            if not config.show_progress and i % 5 == 0:
                progress = (i + 1) / batch_count * 100
                print(f"Progress: {progress:.1f}% complete")
        
        # Final adjustment to exactly hit target size
        current_size = os.path.getsize(config.output_file)
        if current_size < target_bytes:
            remaining = target_bytes - current_size
            if remaining > 0:
                # Check disk space before final adjustment
                free_space = get_free_disk_space(config.output_file)
                if remaining > free_space * (1 - config.safety_margin_percent/100):
                    raise DiskSpaceError(
                        f"Not enough disk space for final adjustment. "
                        f"Required: {human_readable_size(remaining)}, "
                        f"Free: {human_readable_size(free_space)}"
                    )
                
                with open(config.output_file, 'ab') as f:
                    # Generate just enough data to reach the target
                    template = "AAPL,037833100,Apple Inc.,US,NASDAQ,Equity,USD,USA\n"
                    template_bytes = template.encode('utf-8')
                    repeats = remaining // len(template_bytes)
                    remainder = remaining % len(template_bytes)
                    
                    if repeats > 0:
                        f.write(template_bytes * repeats)
                    if remainder > 0:
                        f.write(template_bytes[:remainder])
        
    except DiskSpaceError as e:
        print(f"\n\nERROR: {e}")
        print("Generation aborted due to disk space constraints.")
        print("Partial file may have been created. Consider removing it to free up space.")
        return False
    except KeyboardInterrupt:
        print("\n\nOperation interrupted by user.")
        return False
    except Exception as e:
        print(f"\n\nError during generation: {e}")
        return False
    finally:
        # Ensure monitoring threads stop
        stop_monitor.set()
        disk_monitor.stop()
        if config.show_progress and monitor_thread.is_alive():
            monitor_thread.join(timeout=1.0)
    
    # Final confirmation
    final_size = os.path.getsize(config.output_file)
    print(f"\nGeneration complete: {config.output_file}")
    print(f"Final file size: {human_readable_size(final_size)} "
          f"({(final_size/target_bytes)*100:.2f}% of target)")
    return True


def generate_parallel(config: GenerationConfig):
    """Generate the file using multiple processes for improved performance."""
    # This is a placeholder for future implementation
    # Multi-process generation would be more complex but faster
    print("Parallel generation not implemented yet. Using single-process method.")
    return generate_in_chunks(config)


def cleanup_partial_file(file_path: str):
    """
    Remove partial file if generation failed.
    
    Args:
        file_path: Path to the file to remove
    """
    try:
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            print(f"Cleaning up partial file: {file_path} ({human_readable_size(file_size)})")
            os.remove(file_path)
            print("Cleanup successful.")
    except Exception as e:
        print(f"Error during cleanup: {e}")


def validate_size_safety(output_path: str, target_gb: float):
    """
    Validate that the requested file size can be safely generated.
    
    This function performs checks to ensure that the request is reasonable:
    1. The file size doesn't exceed a safe percentage of total disk space
    2. There's enough free space to generate the file
    3. The target disk has sufficient performance characteristics
    
    Args:
        output_path: Target file path
        target_gb: Requested file size in GB
        
    Returns:
        Tuple of (is_safe, message)
    """
    try:
        # Get disk information
        path_dir = os.path.dirname(os.path.abspath(output_path))
        if not path_dir:
            path_dir = '.'
            
        disk_usage = psutil.disk_usage(path_dir)
        
        # Calculate values
        total_disk_gb = disk_usage.total / (1024**3)
        free_disk_gb = disk_usage.free / (1024**3)
        percent_used = disk_usage.percent
        
        # Check if file size is too large relative to total disk size
        max_safe_percent = 80
        max_safe_gb = total_disk_gb * (max_safe_percent / 100)
        
        if target_gb > max_safe_gb:
            return (False, f"Target file size ({target_gb:.1f}GB) exceeds {max_safe_percent}% of total disk size ({total_disk_gb:.1f}GB). Maximum safe size: {max_safe_gb:.1f}GB.")
        
        # Check if there's enough free space (with buffer)
        buffer_gb = max(2.0, target_gb * 0.05)  # 5% of target size or at least 2GB
        required_gb = target_gb + buffer_gb
        
        if required_gb > free_disk_gb:
            return (False, f"Not enough free space. Required: {required_gb:.1f}GB (target + buffer), Available: {free_disk_gb:.1f}GB.")
        
        # Check if disk is already heavily utilized
        if percent_used > 85:
            return (False, f"Disk is already at {percent_used}% capacity. It's unsafe to generate large files when disk usage is above 85%.")
        
        # All checks passed
        return (True, f"Size validation passed. Creating {target_gb:.1f}GB file (requires {required_gb:.1f}GB with buffer) on disk with {free_disk_gb:.1f}GB available.")
        
    except Exception as e:
        return (False, f"Error validating file size safety: {e}")


def signal_handler(sig, frame):
    """Handle interrupt signals gracefully."""
    print("\n\nReceived interrupt signal. Cleaning up and exiting...")
    sys.exit(130)  # 130 is the standard exit code for SIGINT


def register_signal_handlers():
    """Register signal handlers for graceful termination."""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Entry point for the security master file generator."""
    # Register signal handlers
    register_signal_handlers()
    
    parser = argparse.ArgumentParser(
        description="Generate large security master files for testing"
    )
    
    # File configuration
    parser.add_argument("--output", "-o", type=str, default="secmaster_test.csv",
                        help="Output file path")
    
    # Size options
    size_group = parser.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--preset", "-p", type=str, choices=list(SIZE_PRESETS.keys()),
                          help="Predefined size preset")
    size_group.add_argument("--size", "-s", type=float,
                          help="Custom size in GB")
    
    # Performance options
    parser.add_argument("--batch-size", "-b", type=int, default=500,
                       help="Batch size in MB (default: 500)")
    parser.add_argument("--cores", "-c", type=int, default=1,
                       help="Number of CPU cores to use (default: 1)")
    
    # Content options
    parser.add_argument("--unique", "-u", action="store_true",
                       help="Generate unique values (slower but more realistic)")
    parser.add_argument("--templates", "-t", type=int, default=10,
                       help="Number of row templates to use (default: 10)")
    
    # Safety options
    parser.add_argument("--safety-margin", "-m", type=int, default=10,
                       help="Safety margin percentage of free space (default: 10%%)")
    parser.add_argument("--force", "-f", action="store_true",
                       help="Force generation even if safety checks fail")
    
    # Display options
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress display")
    parser.add_argument("--no-cleanup", action="store_true",
                       help="Don't clean up partial files on failure")
    
    args = parser.parse_args()
    
    # Determine target size
    if args.preset:
        target_gb = SIZE_PRESETS[args.preset]
    else:
        target_gb = args.size
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(f"Error creating output directory: {e}")
            return 1
    
    # Run additional size safety validation
    is_safe, message = validate_size_safety(args.output, target_gb)
    print(message)
    
    if not is_safe and not args.force:
        print("Use --force to override safety checks and generate anyway.")
        return 1
    
    # Configure generation
    config = GenerationConfig(
        output_file=args.output,
        target_size_gb=target_gb,
        template_count=args.templates,
        batch_size_mb=args.batch_size,
        use_unique_values=args.unique,
        show_progress=not args.no_progress,
        cpu_cores=args.cores,
        safety_margin_percent=args.safety_margin
    )
    
    print(f"Generating {target_gb}GB security master file: {args.output}")
    print(f"Using {'unique' if args.unique else 'repeated'} values")
    print(f"Safety margin: {args.safety_margin}% of free space")
    
    start_time = time.time()
    
    try:
        success = False
        if config.cpu_cores > 1:
            success = generate_parallel(config)
        else:
            success = generate_in_chunks(config)
            
        elapsed = time.time() - start_time
        
        if success:
            print(f"Generation completed in {elapsed:.1f} seconds "
                  f"({elapsed/60:.1f} minutes)")
            return 0
        else:
            print(f"Generation failed after {elapsed:.1f} seconds")
            # Clean up partial file if generation failed
            if not args.no_cleanup and os.path.exists(config.output_file):
                cleanup_partial_file(config.output_file)
            return 1
        
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")
        elapsed = time.time() - start_time
        print(f"Interrupted after {elapsed:.1f} seconds")
        
        # Clean up partial file if interrupted
        if not args.no_cleanup and os.path.exists(config.output_file):
            cleanup_partial_file(config.output_file)
            
        return 130
    except Exception as e:
        print(f"\nError during generation: {e}")
        elapsed = time.time() - start_time
        print(f"Failed after {elapsed:.1f} seconds")
        
        # Clean up partial file if failed
        if not args.no_cleanup and os.path.exists(config.output_file):
            cleanup_partial_file(config.output_file)
            
        return 1


if __name__ == "__main__":
    sys.exit(main())