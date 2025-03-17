#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration loader for FIX Symbol Reconciliation Tool.

This module loads configuration from the config.ini file and sets up logging.
Includes advanced performance tuning parameters for handling extremely large
security master files (up to 200GB) without impacting production server resources.

Pushing the performance envelope a bit more before doing rapid QA, then UAT deployment stage.
NOTE: This code has been built to be PROD friendly, not actually PROD ready. Trying to see how far 
      we can push it in a short period of time strictly as a demonstration of skills and abilities. 
      PROD version would require much more extensive testing, QA, and security reviews before deployment.
      Financial risks are too high to cut corners in production systems. This is meant as a technical
      showcase that balances performance with safety mechanisms to protect production environments.

Author: Carlyle
Date: March 17, 2025
"""

import os
import sys
import logging
import configparser
import multiprocessing
import platform
import json
import socket
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List


# Force file and console logging -----------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("security_master_debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Make sure all loggers show DEBUG -----------------------
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).setLevel(logging.DEBUG)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None) -> None:
    """
    Configure the logging system with the specified level and optional file output.
    
    Args:
        log_level: Desired logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to write logs to a file
    """
    try:
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            print(f"Warning: Invalid log level '{log_level}', using INFO")
            numeric_level = logging.INFO
        
        # Basic configuration
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        log_date_format = '%Y-%m-%d %H:%M:%S'
        
        handlers = []
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format, log_date_format))
        handlers.append(console_handler)
        
        # File handler (if specified)
        if log_file:
            # Create directory for log file if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter(log_format, log_date_format))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            handlers=handlers,
            format=log_format,
            datefmt=log_date_format
        )
        
        # Log system information for debugging
        if log_level.upper() == 'DEBUG':
            logger = logging.getLogger(__name__)
            logger.debug(f"Python version: {platform.python_version()}")
            logger.debug(f"Host: {socket.gethostname()}")
            logger.debug(f"CPU cores: {multiprocessing.cpu_count()}")
            logger.debug(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f} GB")
            
    except Exception as e:
        print(f"Error setting up logging: {e}")
        # Fall back to basic logging
        logging.basicConfig(level=logging.INFO)


def detect_system_resources() -> Dict[str, Any]:
    """
    Detect available system resources to automatically configure performance settings.
    
    Returns:
        Dictionary of resource information
    """
    try:
        # Detect CPU cores
        cpu_count = multiprocessing.cpu_count()
        
        # Memory information
        mem_info = psutil.virtual_memory()
        total_mem_gb = mem_info.total / (1024**3)
        available_mem_gb = mem_info.available / (1024**3)
        
        # Disk information (for the current directory)
        disk_info = psutil.disk_usage('.')
        disk_free_gb = disk_info.free / (1024**3)
        
        # Calculate recommended chunk size based on available memory
        # Using 10% of available memory or 64MB, whichever is larger
        recommended_chunk_mb = max(64, int(available_mem_gb * 100))
        
        # Calculate recommended parallelism
        # Use 75% of CPU cores to leave room for other processes
        recommended_threads = max(1, int(cpu_count * 0.75))
        
        return {
            'cpu_count': cpu_count,
            'total_memory_gb': total_mem_gb,
            'available_memory_gb': available_mem_gb,
            'disk_free_gb': disk_free_gb,
            'recommended_chunk_mb': recommended_chunk_mb,
            'recommended_threads': recommended_threads
        }
    except Exception as e:
        logging.warning(f"Error detecting system resources: {e}")
        # Return conservative defaults
        return {
            'cpu_count': 2,
            'total_memory_gb': 8.0,
            'available_memory_gb': 4.0,
            'disk_free_gb': 10.0,
            'recommended_chunk_mb': 64,
            'recommended_threads': 1
        }


def load_config(config_file: str = 'config.ini', auto_tune: bool = True) -> Dict[str, Any]:
    """
    Load configuration from the INI file, environment variables, and auto-detection.
    
    Args:
        config_file: Path to config file (default: config.ini in current dir)
        auto_tune: Whether to automatically tune parameters based on system resources
        
    Returns:
        Dictionary containing configuration settings
    
    Note: 
        Security-sensitive settings like connection details should be provided
        via environment variables rather than in the config file.
        Example: FIX_SECMASTER_HOST, FIX_SECMASTER_PORT
    """
    # Start with defaults in case anything fails
    result = _get_default_config()
    
    # Detect system resources if auto-tuning is enabled
    if auto_tune:
        resources = detect_system_resources()
        
        # Apply recommended values based on system resources
        result['chunk_size_mb'] = resources['recommended_chunk_mb']
        result['max_threads'] = resources['recommended_threads']
        result['memory_limit_pct'] = min(80, int(resources['available_memory_gb'] / resources['total_memory_gb'] * 70))
        result['cpu_limit_pct'] = 75
        
        logging.info(f"Auto-tuned performance settings based on {resources['cpu_count']} cores "
                    f"and {resources['available_memory_gb']:.1f}GB available memory")
    
    try:
        config = configparser.ConfigParser()
        
        if not os.path.exists(config_file):
            logging.warning(f"Config file {config_file} not found, using defaults and auto-detection")
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
            result['log_file'] = config.get('Paths', 'log_file',
                                          fallback=result['log_file'])
        
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
        
        # Performance section
        if config.has_section('Performance'):
            try:
                # Only override auto-tuned values if explicitly set
                if config.has_option('Performance', 'chunk_size_mb'):
                    result['chunk_size_mb'] = config.getint('Performance', 'chunk_size_mb')
                
                if config.has_option('Performance', 'max_threads'):
                    result['max_threads'] = config.getint('Performance', 'max_threads')
                    
                if config.has_option('Performance', 'use_mmap'):
                    result['use_mmap'] = config.getboolean('Performance', 'use_mmap')
                    
                if config.has_option('Performance', 'memory_limit_pct'):
                    result['memory_limit_pct'] = config.getint('Performance', 'memory_limit_pct')
                    
                if config.has_option('Performance', 'cpu_limit_pct'):
                    result['cpu_limit_pct'] = config.getint('Performance', 'cpu_limit_pct')
                    
                if config.has_option('Performance', 'throttling_enabled'):
                    result['throttling_enabled'] = config.getboolean('Performance', 'throttling_enabled')
                    
            except ValueError as e:
                logging.warning(f"Invalid performance setting in config, using auto-tuned values: {e}")
        
        # Resource protection section
        if config.has_section('ResourceProtection'):
            try:
                if config.has_option('ResourceProtection', 'enabled'):
                    result['resource_protection'] = config.getboolean('ResourceProtection', 'enabled')
                    
                if config.has_option('ResourceProtection', 'monitoring_interval_sec'):
                    result['monitoring_interval_sec'] = config.getfloat('ResourceProtection', 'monitoring_interval_sec')
                    
                if config.has_option('ResourceProtection', 'backoff_factor'):
                    result['backoff_factor'] = config.getfloat('ResourceProtection', 'backoff_factor')
                    
                if config.has_option('ResourceProtection', 'critical_memory_pct'):
                    result['critical_memory_pct'] = config.getint('ResourceProtection', 'critical_memory_pct')
                    
            except ValueError as e:
                logging.warning(f"Invalid resource protection setting in config: {e}")
        
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
        
    # Generate output file path if not explicitly set
    if not result.get('output_path'):
        report_filename = f"discrepancy_report_{Path(result['fix_log_path']).stem}.csv"
        result['output_path'] = os.path.join(result['output_dir'], report_filename)
    
    # Setup logging with configured level
    setup_logging(result['log_level'], result.get('log_file'))
    
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
        'FIX_OUTPUT_PATH': 'output_path',
        'FIX_DELIMITER': 'fix_delimiter',
        'FIX_LOG_LEVEL': 'log_level',
        'FIX_LOG_FILE': 'log_file',
        'FIX_CHUNK_SIZE': 'chunk_size_mb',
        'FIX_MAX_THREADS': 'max_threads',
        'FIX_USE_MMAP': 'use_mmap',
        'FIX_MEMORY_LIMIT': 'memory_limit_pct',
        'FIX_CPU_LIMIT': 'cpu_limit_pct',
        'FIX_RESOURCE_PROTECTION': 'resource_protection',
        
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
            # Handle boolean values
            if env_value.lower() in ('true', 'yes', '1'):
                config[config_key] = True
            elif env_value.lower() in ('false', 'no', '0'):
                config[config_key] = False
            # Handle numeric values
            elif config_key in ('chunk_size_mb', 'max_threads', 'memory_limit_pct', 'cpu_limit_pct'):
                try:
                    config[config_key] = int(env_value)
                except ValueError:
                    logging.warning(f"Invalid numeric value for {env_var}: {env_value}")
            else:
                config[config_key] = env_value
                
            # Don't log sensitive values
            if 'password' not in config_key and 'key' not in config_key:
                logging.debug(f"Applied environment override for {config_key}")
    
    return config


def _get_default_config() -> Dict[str, Any]:
    """Return default configuration when config file is not available."""
    return {
        # File paths
        'secmaster_path': './data/sample/secmaster.csv',
        'fix_log_path': './data/sample/fix.log',
        'output_dir': './reports',
        'output_path': None,  # Will be generated based on fix_log_path
        'log_file': None,  # Default to console logging only
        
        # FIX protocol settings
        'fix_delimiter': '|',
        'target_message_types': ['D', '8'],
        
        # Logging
        'log_level': 'INFO',
        
        # Performance settings (will be auto-tuned if enabled)
        'chunk_size_mb': 64,  # Default chunk size for processing
        'max_threads': 1,     # Default to single thread
        'use_mmap': True,     # Use memory mapping for large files
        'batch_size': 1000,   # Records to process in a batch
        
        # Resource protection
        'resource_protection': True,  # Enable resource protection features
        'memory_limit_pct': 80,       # Memory usage threshold
        'cpu_limit_pct': 75,          # CPU usage threshold
        'throttling_enabled': True,   # Enable throttling when resource limits approached
        'monitoring_interval_sec': 1.0, # Resource check interval
        'backoff_factor': 1.5,        # Backoff factor when limits approached
        'critical_memory_pct': 90,    # Critical memory threshold for emergency exit
        
        # Advanced settings
        'enable_parallel_processing': False,  # Experimental feature flag
    }


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate configuration settings to ensure they're reasonable.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []
    
    # Check required file paths
    for path_key in ['secmaster_path', 'fix_log_path']:
        if not config.get(path_key):
            errors.append(f"Missing required configuration: {path_key}")
        elif not os.path.exists(config[path_key]):
            errors.append(f"File not found: {config[path_key]}")
    
    # Validate numeric ranges
    if not (1 <= config.get('max_threads', 1) <= multiprocessing.cpu_count() * 2):
        errors.append(f"Invalid max_threads value: {config.get('max_threads')}. Must be 1-{multiprocessing.cpu_count() * 2}")
        
    if not (16 <= config.get('chunk_size_mb', 64) <= 1024):
        errors.append(f"Invalid chunk_size_mb value: {config.get('chunk_size_mb')}. Must be 16-1024")
        
    if not (50 <= config.get('memory_limit_pct', 80) <= 95):
        errors.append(f"Invalid memory_limit_pct value: {config.get('memory_limit_pct')}. Must be 50-95")
        
    if not (50 <= config.get('cpu_limit_pct', 75) <= 99):
        errors.append(f"Invalid cpu_limit_pct value: {config.get('cpu_limit_pct')}. Must be 50-99")
    
    return (len(errors) == 0, errors)


def get_available_resources() -> Dict[str, Any]:
    """
    Get current available system resources for runtime monitoring.
    
    Returns:
        Dictionary with current resource availability information
    """
    try:
        mem = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)
        disk = psutil.disk_usage('.')
        
        return {
            'memory_used_pct': mem.percent,
            'memory_available_gb': mem.available / (1024**3),
            'cpu_used_pct': cpu_percent,
            'disk_free_gb': disk.free / (1024**3),
            'disk_used_pct': disk.percent
        }
    except Exception as e:
        logging.error(f"Error getting system resources: {e}")
        return {
            'memory_used_pct': 50,
            'memory_available_gb': 8.0,
            'cpu_used_pct': 50,
            'disk_free_gb': 10.0,
            'disk_used_pct': 50
        }


def estimate_memory_requirement(file_size_gb: float, safety_factor: float = 1.5) -> float:
    """
    Estimate memory requirements for processing a file of a given size.
    
    Args:
        file_size_gb: Size of the file in GB
        safety_factor: Multiplier for safety margin
        
    Returns:
        Estimated memory requirement in GB
    """
    # For security master files, we typically need 15-20% of file size for indexing
    # For FIX logs, we typically need 30-40% of file size for parsed messages
    # Default conservative estimate based on empirical testing
    return file_size_gb * 0.4 * safety_factor


def check_file_requirements(config: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Check if system meets the requirements to process the configured files.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (requirements_met, message)
    """
    try:
        # Get file sizes
        secmaster_size_gb = os.path.getsize(config['secmaster_path']) / (1024**3)
        fix_log_size_gb = os.path.getsize(config['fix_log_path']) / (1024**3)
        
        # Get available memory
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)
        
        # Estimate memory requirements
        required_mem_gb = estimate_memory_requirement(secmaster_size_gb) + estimate_memory_requirement(fix_log_size_gb)
        
        # Check if we have enough memory
        if required_mem_gb > available_mem_gb:
            return (False, f"Insufficient memory. Estimated requirement: {required_mem_gb:.1f}GB, Available: {available_mem_gb:.1f}GB")
        
        # Check disk space for output
        disk = psutil.disk_usage(config['output_dir'])
        output_size_estimate_gb = (secmaster_size_gb + fix_log_size_gb) * 0.1  # Output is typically much smaller
        
        if output_size_estimate_gb > disk.free / (1024**3):
            return (False, f"Insufficient disk space for output. Estimated requirement: {output_size_estimate_gb:.1f}GB, Available: {disk.free / (1024**3):.1f}GB")
        
        return (True, f"System meets requirements. Processing {secmaster_size_gb:.1f}GB security master and {fix_log_size_gb:.1f}GB FIX log.")
        
    except Exception as e:
        logging.error(f"Error checking file requirements: {e}")
        return (False, f"Error checking requirements: {e}")


def generate_default_config_file(output_path: str = 'config.ini') -> bool:
    """
    Generate a default configuration file with comments.
    
    Args:
        output_path: Path where to write the config file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create config parser
        config = configparser.ConfigParser()
        
        # Paths section
        config['Paths'] = {
            '# Security master CSV file path': None,
            'security_master': './data/sample/secmaster.csv',
            
            '# FIX log file path': None,
            'fix_log': './data/sample/fix.log',
            
            '# Output directory for reports': None,
            'output_dir': './reports',
            
            '# Optional log file path (comment out for console only)': None,
            '# log_file': './logs/fix_reconciliation.log'
        }
        
        # FIX section
        config['FIX'] = {
            '# Field delimiter in FIX messages': None,
            'delimiter': '|',
            
            '# Target message types to process (comma-separated)': None,
            'target_message_types': 'D,8'
        }
        
        # Logging section
        config['Logging'] = {
            '# Logging level (DEBUG, INFO, WARNING, ERROR)': None,
            'level': 'INFO'
        }
        
        # Performance section
        config['Performance'] = {
            '# Size of each chunk to process in MB (16-1024)': None,
            '# chunk_size_mb': '64',
            
            '# Maximum number of threads to use (1-N)': None,
            '# max_threads': '2',
            
            '# Use memory mapping for large files (true/false)': None,
            'use_mmap': 'true',
            
            '# Memory usage limit percentage (50-95)': None,
            'memory_limit_pct': '80',
            
            '# CPU usage limit percentage (50-99)': None,
            'cpu_limit_pct': '75',
            
            '# Enable throttling when resource limits approached (true/false)': None,
            'throttling_enabled': 'true'
        }
        
        # Resource protection section
        config['ResourceProtection'] = {
            '# Enable resource protection features (true/false)': None,
            'enabled': 'true',
            
            '# Resource monitoring interval in seconds (0.5-10.0)': None,
            'monitoring_interval_sec': '1.0',
            
            '# Backoff factor when limits approached (1.1-5.0)': None,
            'backoff_factor': '1.5',
            
            '# Critical memory percentage for emergency exit (85-99)': None,
            'critical_memory_pct': '90'
        }
        
        # Write to file with comments
        with open(output_path, 'w') as f:
            f.write("# FIX Symbol Reconciliation Tool Configuration\n")
            f.write("# Generated on: " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            config.write(f)
            
        logging.info(f"Default configuration file generated at {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Error generating default config file: {e}")
        return False


# Placeholder for future enhancement
def load_config_from_database(connection_string: str) -> Dict[str, Any]:
    """
    Load configuration from a centralized database (future enhancement).
    
    Args:
        connection_string: Database connection string
        
    Returns:
        Configuration dictionary
    """
    # This is a placeholder for future enhancement
    logging.warning("Database configuration loading not implemented yet")
    return _get_default_config()


if __name__ == "__main__":
    # Set up basic logging for standalone usage
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Simple CLI for generating default config and testing
    if len(sys.argv) > 1:
        if sys.argv[1] == "--generate-config":
            output_path = sys.argv[2] if len(sys.argv) > 2 else "config.ini"
            generate_default_config_file(output_path)
            sys.exit(0)
            
        if sys.argv[1] == "--test-load":
            config_file = sys.argv[2] if len(sys.argv) > 2 else "config.ini"
            config = load_config(config_file)
            print("Configuration loaded:")
            for key, value in config.items():
                # Don't print sensitive values
                if 'password' not in key and 'key' not in key:
                    print(f"  {key}: {value}")
            sys.exit(0)
            
        if sys.argv[1] == "--check-resources":
            print("System Resources:")
            resources = detect_system_resources()
            for key, value in resources.items():
                print(f"  {key}: {value}")
            sys.exit(0)
    
    # If no arguments, just load and display the configuration
    config = load_config()
    print("Configuration settings:")
    for key, value in config.items():
        # Don't display sensitive values
        if 'password' not in key and 'key' not in key:
            print(f"  {key}: {value}")