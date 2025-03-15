#!/usr/bin/env python3
"""
Configuration loader for FIX Symbol Reconciliation Tool.

This module loads configuration from the config.ini file and sets up logging.
Future enhancement: Convert to fully dynamic configuration with command-line args.
"""

import os
import logging
import configparser
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure the logging system with the specified level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_file: str = 'config.ini') -> Dict[str, Any]:
    """
    Load configuration from the INI file.
    
    Args:
        config_file: Path to config file (default: config.ini in current dir)
        
    Returns:
        Dictionary containing configuration settings
    """
    config = configparser.ConfigParser()
    
    if not os.path.exists(config_file):
        logging.warning(f"Config file {config_file} not found, using defaults")
        return _get_default_config()
    
    config.read(config_file)
    
    # Create a dictionary from the config sections
    result = {}
    
    # Paths section
    result['secmaster_path'] = config.get('Paths', 'security_master', 
                                         fallback='./data/sample/secmaster.csv')
    result['fix_log_path'] = config.get('Paths', 'fix_log', 
                                       fallback='./data/sample/fix.log')
    result['output_dir'] = config.get('Paths', 'output_dir', 
                                     fallback='./reports')
    
    # FIX section
    result['fix_delimiter'] = config.get('FIX', 'delimiter', fallback='|')
    message_types = config.get('FIX', 'target_message_types', fallback='D,8')
    result['target_message_types'] = [t.strip() for t in message_types.split(',')]
    
    # Logging section
    result['log_level'] = config.get('Logging', 'level', fallback='INFO')
    
    # Processing section
    result['batch_size'] = config.getint('Processing', 'batch_size', fallback=1000)
    
    # Create output directory if it doesn't exist
    Path(result['output_dir']).mkdir(parents=True, exist_ok=True)
    
    # Generate output file path
    report_filename = f"discrepancy_report_{Path(result['fix_log_path']).stem}.csv"
    result['output_path'] = os.path.join(result['output_dir'], report_filename)
    
    # Setup logging
    setup_logging(result['log_level'])
    
    logging.info(f"Configuration loaded from {config_file}")
    
    return result


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration when config file is not available."""
    return {
        'secmaster_path': './data/sample/secmaster.csv',
        'fix_log_path': './data/sample/fix.log',
        'output_dir': './reports',
        'fix_delimiter': '|',
        'target_message_types': ['D', '8'],
        'log_level': 'INFO',
        'batch_size': 1000
    }


if __name__ == "__main__":
    # Test the config loading
    config = load_config()
    print("Configuration settings:")
    for key, value in config.items():
        print(f"  {key}: {value}")