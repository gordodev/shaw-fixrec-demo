#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX Symbol Discrepancy Checker

This tool identifies discrepancies between FIX message symbols and the security master
database, with focus on detecting corporate action-related symbol changes that weren't
properly updated in trading systems.

Key features:
- Memory-efficient processing of extremely large security master files (I tested up to 200GB security master file)
- Streaming FIX log parsing to minimize memory usage
- Progress reporting and resource monitoring
- logging

Usage:
    python main.py [--secmaster SECMASTER_PATH] [--fix-log FIX_LOG_PATH] 
                  [--output OUTPUT_PATH] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Author: Carlyle
Date: March 16, 2025
"""

import os
import sys
import time
import argparse
import logging
import gc
import psutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Internal modules
from config import load_config, setup_logging
from security_master import load_security_master
from fix_parser import FIXParser
from analyzer import DiscrepancyAnalyzer
from reporter import DiscrepancyReporter


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
                       type=int, default=64,
                       help="Security master chunk size in MB (default: 64)")
    
    parser.add_argument("--disable-mmap", 
                       action="store_true",
                       help="Disable memory mapping for security master loading")
    
    return parser.parse_args()


def load_security_master_data(secmaster_path: str, chunk_size_mb: int = 64, use_mmap: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Load and index the security master file in a memory-efficient way.
    
    Args:
        secmaster_path: Path to security master CSV file
        chunk_size_mb: Size of each chunk to process in MB
        use_mmap: Whether to use memory mapping for file access
        
    Returns:
        Dictionary of security master data indexed by CUSIP
        
    Raises:
        FileNotFoundError: If security master file doesn't exist
        ValueError: If security master file is invalid
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading security master from: {secmaster_path}")
    
    # Log memory usage at start
    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss
    logger.info(f"Memory usage before loading: {start_memory / (1024**2):.1f} MB")
    
    # Start loading timer
    start_time = time.time()
    
    # Load the security master with chunked loading
    try:
        # Call the factory function from chunked_secmaster_loader
        sec_master = load_security_master(secmaster_path)
        
        if not sec_master.is_file_loaded():
            logger.error(f"Failed to load security master from: {secmaster_path}")
            raise ValueError(f"Failed to load security master from: {secmaster_path}")
        
        # Get a lightweight dictionary for the analyzer
        security_data = sec_master.get_security_dict()
        
        # Log timing and memory stats
        elapsed_time = time.time() - start_time
        current_memory = process.memory_info().rss
        memory_increase = current_memory - start_memory
        
        logger.info(f"Loaded {sec_master.get_security_count():,} securities in {elapsed_time:.2f} seconds")
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


def parse_fix_log(fix_log_path: str, delimiter: str = "|") -> List[Dict[str, Any]]:
    """Parse the FIX message log file."""
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing FIX messages from: {fix_log_path}")
    
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss
    
    fix_parser = FIXParser(delimiter=delimiter)
    messages = fix_parser.parse_file(fix_log_path)
    
    elapsed_time = time.time() - start_time
    current_memory = psutil.Process(os.getpid()).memory_info().rss
    memory_increase = current_memory - start_memory
    
    stats = fix_parser.get_statistics(messages)
    
    # Safe logging that handles empty statistics
    logger.info(f"Parsed {stats.get('total_messages', 0)} messages in {elapsed_time:.2f} seconds: "
               f"{stats.get('new_order_singles', 0)} new orders, "
               f"{stats.get('execution_reports', 0)} execution reports")
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
    
    # Return empty list if no valid messages rather than crashing
    return message_dicts


def analyze_fix_messages(fix_messages: List[Dict[str, Any]], 
                        security_master: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """
    Analyze FIX messages to identify discrepancies with security master.
    
    Args:
        fix_messages: List of parsed FIX messages
        security_master: Security master data indexed by CUSIP
        
    Returns:
        Tuple of (discrepancies, total_exposure)
    """
    logger = logging.getLogger(__name__)
    logger.info("Analyzing FIX messages for discrepancies against security master")
    
    start_time = time.time()
    
    analyzer = DiscrepancyAnalyzer(security_master)
    discrepancies, total_exposure = analyzer.analyze_messages(fix_messages)
    
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
                   output_path: str) -> None:
    """
    Generate a report of the identified discrepancies.
    
    Args:
        discrepancies: List of discrepancies
        total_exposure: Total financial exposure
        output_path: Path to output report file
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Generating discrepancy report at: {output_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.debug(f"Created output directory: {output_dir}")
    
    reporter = DiscrepancyReporter(output_path)
    reporter.generate_report(discrepancies, total_exposure)
    
    logger.info(f"Report generated successfully: {output_path}")


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
    
    try:
        # Step 1: Load security master with memory-efficient chunked loading
        security_master = load_security_master_data(
            config['secmaster_path'],
            chunk_size_mb=config.get('chunk_size', 64),
            use_mmap=not config.get('disable_mmap', False)
        )
        
        # Step 2: Parse FIX log
        fix_messages = parse_fix_log(config['fix_log_path'], config['fix_delimiter'])
        
        # Step 3: Analyze messages for discrepancies
        discrepancies, total_exposure = analyze_fix_messages(fix_messages, security_master)
        
        # Step 4: Generate report
        generate_report(discrepancies, total_exposure, config['output_path'])
        
        # Log summary
        elapsed_time = time.time() - start_time
        logger.info(f"Completed successfully in {elapsed_time:.2f} seconds")
        logger.info(f"Found {len(discrepancies)} discrepancies with total exposure of ${total_exposure:,.2f}")
        
        # Log memory usage
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss
        logger.info(f"Peak memory usage: {memory_usage / (1024**2):.1f} MB")
        
        return 0  # Success
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2  # File access error
        
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        return 3  # Processing error
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1  # General error
    
    finally:
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
        
        if 'chunk_size' in config:
            print(f"Security Master Chunk Size: {config['chunk_size']} MB")
        
        if config.get('disable_mmap', False):
            print("Memory Mapping: Disabled")
        
        print("-" * 50)
        
        # Run the main process
        return run(config)
        
    except Exception as e:
        print(f"Error in configuration: {e}")
        return 1


if __name__ == "__main__":
    # Set the exit code from the main function
    sys.exit(main())