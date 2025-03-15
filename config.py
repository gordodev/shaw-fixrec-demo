#!/usr/bin/env python3
"""
Configuration loader for FIX Symbol Reconciliation Tool.

This module loads configuration from the config.ini file and sets up logging.
Future enhancement: Convert to fully dynamic configuration with command-line args.
"""

import os
import sys
import logging
import configparser
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_level: str = 'INFO') -> None:
    """Configure the logging system with the specified level."""
    try:
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Warning: Invalid log level '{log_level}', using INFO")
            numeric_level = logging.INFO
        
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fall back to basic logging
        logging.basicConfig(level=logging.INFO)


def load_config(config_file: str = 'config.ini') -> Dict[str, Any]:
    """
    Load configuration from the INI file and environment variables.
    
    Args:
        config_file: Path to config file (default: config.ini in current dir)
        
    Returns:
        Dictionary containing configuration settings
    
    Note: 
        Security-sensitive settings like connection details should be provided
        via environment variables rather than in the config file.
        Example: FIX_SECMASTER_HOST, FIX_SECMASTER_PORT
    """
    # Start with defaults in case anything fails
    result = _get_default_config()
    
    try:
        config = configparser.ConfigParser()
        
        if not os.path.exists(config_file):
            logging.warning(f"Config file {config_file} not found, using defaults")
            return _apply_environment_overrides(result)
        
        # This could raise various exceptions if file is corrupt or inaccessible
        config.read(config_file)
        
        # Paths section
        if config.has_section('Paths'):
            result['secmaster_path'] = config.get('Paths', 'security_master', 
                                                fallback=result['secmaster_path'])
            result['fix_log_path'] = config.get('Paths', 'fix_log', 
                                               fallback=result['fix_log_path'])
            result['output_dir'] = config.get('Paths', 'output_dir', 
                                            fallback=result['output_dir'])
        
        # FIX section
        if config.has_section('FIX'):
            result['fix_delimiter'] = config.get('FIX', 'delimiter', 
                                               fallback=result['fix_delimiter'])
            
            try:
                message_types = config.get('FIX', 'target_message_types', 
                                          fallback='D,8')
                result['target_message_types'] = [t.strip() for t in message_types.split(',')]
            except Exception as e:
                logging.warning(f"Error parsing target_message_types, using defaults: {e}")
        
        # Logging section
        if config.has_section('Logging'):
            result['log_level'] = config.get('Logging', 'level', 
                                           fallback=result['log_level'])
        
        # Processing section
        if config.has_section('Processing'):
            try:
                result['batch_size'] = config.getint('Processing', 'batch_size', 
                                                   fallback=result['batch_size'])
            except ValueError as e:
                logging.warning(f"Invalid batch_size in config, using default: {e}")
        
        logging.info(f"Configuration loaded from {config_file}")
        
    except configparser.Error as e:
        logging.error(f"Error parsing config file {config_file}: {e}")
        logging.info("Proceeding with default configuration")
    except PermissionError as e:
        logging.error(f"Permission denied accessing config file {config_file}: {e}")
        logging.info("Proceeding with default configuration")
    except Exception as e:
        logging.error(f"Unexpected error loading config: {e}")
        logging.info("Proceeding with default configuration")
    
    # Apply environment variable overrides (higher precedence than file)
    result = _apply_environment_overrides(result)
    
    # Create output directory if it doesn't exist
    try:
        Path(result['output_dir']).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error(f"Failed to create output directory: {e}")
        # Fallback to current directory
        result['output_dir'] = '.'
        
    # Generate output file path
    report_filename = f"discrepancy_report_{Path(result['fix_log_path']).stem}.csv"
    result['output_path'] = os.path.join(result['output_dir'], report_filename)
    
    # Setup logging with configured level
    setup_logging(result['log_level'])
    
    return result


def _apply_environment_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Override configuration with values from environment variables.
    
    This allows for secure handling of sensitive information such as
    host addresses, credentials, and ports without storing in config files.
    
    Args:
        config: Current configuration dictionary
        
    Returns:
        Updated configuration dictionary
    """
    # Map of environment variables to config keys
    env_mapping = {
        'FIX_SECMASTER_PATH': 'secmaster_path',
        'FIX_LOG_PATH': 'fix_log_path',
        'FIX_OUTPUT_DIR': 'output_dir',
        'FIX_DELIMITER': 'fix_delimiter',
        'FIX_LOG_LEVEL': 'log_level',
        
        # Security-sensitive settings that should only come from env vars
        'FIX_SECMASTER_HOST': 'secmaster_host',
        'FIX_SECMASTER_PORT': 'secmaster_port',
        'FIX_SECMASTER_USER': 'secmaster_user',
        'FIX_SECMASTER_PASSWORD': 'secmaster_password'
    }
    
    # Apply overrides from environment
    for env_var, config_key in env_mapping.items():
        env_value = os.environ.get(env_var)
        if env_value is not None:
            config[config_key] = env_value
            # Don't log sensitive values
            if 'password' not in config_key and 'key' not in config_key:
                logging.debug(f"Applied environment override for {config_key}")
    
    return config


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
        # Don't display sensitive values
        if 'password' not in key and 'key' not in key:
            print(f"  {key}: {value}")