#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX Symbol Discrepancy Checker

This tool identifies discrepancies between FIX message symbols and the security master
database, with focus on detecting corporate action-related symbol changes that weren't
properly updated in trading systems.

Usage:
    python main.py [--secmaster SECMASTER_PATH] [--fix-log FIX_LOG_PATH] 
                   [--output OUTPUT_PATH] [--log-level {DEBUG,INFO,WARNING,ERROR}]

Author: Carlyle
Date: March 15, 2025
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

# Internal modules
from config import load_config, setup_logging
from fix_parser import FIXParser
from analyzer import DiscrepancyAnalyzer
from reporter import DiscrepancyReporter
from security_master import load_security_master


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="FIX Symbol Discrepancy Checker")
    
    parser.add_argument("--secmaster", 
                       help="Path to security master CSV file")
    
    parser.add_argument("--fix-log", 
                       help="Path to FIX log file")
    
    parser.add_argument("--output", 
                       help="Path to output report file")
    
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    parser.add_argument("--delimiter", 
                       help="FIX message field delimiter (default: |)")
    
    return parser.parse_args()


def load_security_master_data(secmaster_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load and index the security master file.
    
    Args:
        secmaster_path: Path to security master CSV file
        
    Returns:
        Dictionary of security master data indexed by CUSIP
        
    Raises:
        FileNotFoundError: If security master file doesn't exist
        ValueError: If security master file is invalid
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading security master from: {secmaster_path}")
    
    # Use the load_security_master function from security_master.py
    sec_master = load_security_master(secmaster_path)
    
    if not sec_master.is_loaded():
        logger.error(f"Failed to load security master from: {secmaster_path}")
        raise ValueError(f"Failed to load security master from: {secmaster_path}")
    
    # Convert SecurityMaster object's data to the format needed by the analyzer
    security_data = {}
    for cusip, security in sec_master.securities.items():
        security_data[cusip] = security
    
    logger.info(f"Loaded {len(security_data)} securities from master file")
    
    # Log some stats about the security master
    regions = set(sec.get('Region', 'Unknown') for sec in security_data.values() if 'Region' in sec)
    exchanges = set(sec.get('Exchange', 'Unknown') for sec in security_data.values() if 'Exchange' in sec)
    
    logger.info(f"Security master covers {len(regions)} regions and {len(exchanges)} exchanges")
    
    return security_data


def parse_fix_log(fix_log_path: str, delimiter: str = "|") -> List[Dict[str, Any]]:
    """
    Parse the FIX message log file.
    
    Args:
        fix_log_path: Path to FIX log file
        delimiter: Character used to separate FIX fields (default: |)
        
    Returns:
        List of parsed FIX messages
        
    Raises:
        FileNotFoundError: If FIX log file doesn't exist
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Parsing FIX messages from: {fix_log_path}")
    
    fix_parser = FIXParser(delimiter=delimiter)
    messages = fix_parser.parse_file(fix_log_path)
    
    stats = fix_parser.get_statistics(messages)
    
    logger.info(f"Parsed {stats['total_messages']} messages: "
               f"{stats['new_order_singles']} new orders, "
               f"{stats['execution_reports']} execution reports")
    
    # Convert FIXMessage objects to dictionaries for analyzer
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
    
    analyzer = DiscrepancyAnalyzer(security_master)
    discrepancies, total_exposure = analyzer.analyze_messages(fix_messages)
    
    logger.info(f"Analysis complete. Found {len(discrepancies)} discrepancies "
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
        # Step 1: Load security master
        security_master = load_security_master_data(config['secmaster_path'])
        
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
        print("-" * 50)
        
        # Run the main process
        return run(config)
        
    except Exception as e:
        print(f"Error in configuration: {e}")
        return 1


if __name__ == "__main__":
    # Set the exit code from the main function
    sys.exit(main())