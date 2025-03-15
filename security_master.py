#!/usr/bin/env python3
"""
Security Master Data Loader

This module handles loading and indexing security master data from CSV files.
It provides efficient lookup by various identifiers (CUSIP, Symbol) 

"""

import os
import csv
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
import pandas as pd
from pathlib import Path


class SecurityMaster:
    """
    Manages security reference data with lookups by various identifiers.
    
    This class loads security master data from CSV files and provides methods
    to lookup securities by CUSIP, Symbol, or other identifiers. It's designed
    for high-performance lookups in trading operations.
    """
    
    def __init__(self):
        """Initialize an empty security master."""
        self.securities = {}
        self.cusip_index = {}
        self.symbol_index = {}
        self.region_index = {}
        self.loaded = False
        self.source_file = None
        self.record_count = 0
        
    def load_csv(self, file_path: str, delimiter: str = ',') -> bool:
        """
        Load security master data from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing security master data
            delimiter: CSV delimiter character (default: ',')
            
        Returns:
            bool: True if loading was successful, False otherwise
            
        Raises:
            FileNotFoundError: If the file does not exist
            PermissionError: If the file cannot be accessed
        """
        try:
            if not os.path.exists(file_path):
                logging.error(f"Security master file not found: {file_path}")
                return False
                
            self.source_file = file_path
            logging.info(f"Loading security master from {file_path}")
            
            # Using pandas for efficient CSV parsing
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            # Validate required columns
            required_columns = {'CUSIP', 'Symbol'}
            if not required_columns.issubset(set(df.columns)):
                missing = required_columns - set(df.columns)
                logging.error(f"Security master file missing required columns: {missing}")
                return False
            
            # Reset indexes
            self.securities = {}
            self.cusip_index = {}
            self.symbol_index = {}
            self.region_index = {}
            
            # Process each row and build indexes
            for _, row in df.iterrows():
                cusip = row['CUSIP']
                symbol = row['Symbol']
                
                # Store the complete record
                self.securities[cusip] = row.to_dict()
                
                # Build indexes for efficient lookups
                self.cusip_index[cusip] = cusip
                self.symbol_index[symbol] = cusip
                
                # Build region index if available
                if 'Region' in row and pd.notna(row['Region']):
                    region = row['Region']
                    if region not in self.region_index:
                        self.region_index[region] = []
                    self.region_index[region].append(cusip)
            
            self.record_count = len(self.securities)
            self.loaded = True
            logging.info(f"Successfully loaded {self.record_count} securities")
            return True
            
        except pd.errors.EmptyDataError:
            logging.error(f"Security master file is empty: {file_path}")
            return False
        except pd.errors.ParserError as e:
            logging.error(f"Error parsing security master CSV: {e}")
            return False
        except (PermissionError, OSError) as e:
            logging.error(f"Error accessing security master file: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error loading security master: {e}")
            return False
    
    def get_security_by_cusip(self, cusip: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve security information by CUSIP.
        
        Args:
            cusip: The CUSIP identifier
            
        Returns:
            Dict containing security information or None if not found
        """
        if not self.loaded:
            logging.warning("Attempted lookup before security master was loaded")
            return None
            
        return self.securities.get(cusip)
    
    def get_symbol_by_cusip(self, cusip: str) -> Optional[str]:
        """
        Get the official symbol for a given CUSIP.
        
        Args:
            cusip: The CUSIP identifier
            
        Returns:
            The official symbol or None if CUSIP not found
        """
        if not self.loaded:
            logging.warning("Attempted lookup before security master was loaded")
            return None
            
        security = self.securities.get(cusip)
        return security.get('Symbol') if security else None
    
    def get_cusip_by_symbol(self, symbol: str) -> Optional[str]:
        """
        Lookup CUSIP by symbol.
        
        Args:
            symbol: The security symbol
            
        Returns:
            The CUSIP or None if symbol not found
        """
        if not self.loaded:
            logging.warning("Attempted lookup before security master was loaded")
            return None
            
        return self.symbol_index.get(symbol)
    
    def get_securities_by_region(self, region: str) -> List[Dict[str, Any]]:
        """
        Get all securities for a specific region.
        
        Args:
            region: Region code (e.g., 'US', 'EUROPE', 'ASIA')
            
        Returns:
            List of securities in the specified region
        """
        if not self.loaded:
            logging.warning("Attempted lookup before security master was loaded")
            return []
            
        cusips = self.region_index.get(region, [])
        return [self.securities[cusip] for cusip in cusips]
    
    def get_security_count(self) -> int:
        """Return the total number of securities loaded."""
        return self.record_count
    
    def is_loaded(self) -> bool:
        """Check if security master data has been loaded."""
        return self.loaded
    
    def validate_symbol_cusip_pair(self, symbol: str, cusip: str) -> Tuple[bool, Optional[str]]:
        """
        Validate if a symbol matches the expected symbol for a CUSIP.
        
        Args:
            symbol: The symbol to validate
            cusip: The CUSIP to check against
            
        Returns:
            Tuple of (is_valid, expected_symbol)
            - is_valid: True if the symbol matches the CUSIP, False otherwise
            - expected_symbol: The expected symbol if different, None if match
        """
        if not self.loaded:
            logging.warning("Attempted validation before security master was loaded")
            return (False, None)
            
        expected_symbol = self.get_symbol_by_cusip(cusip)
        
        if not expected_symbol:
            logging.warning(f"CUSIP {cusip} not found in security master")
            return (False, None)
            
        if symbol == expected_symbol:
            return (True, None)
        else:
            return (False, expected_symbol)
            

# Module-level factory function for easier instantiation
def load_security_master(file_path: str) -> SecurityMaster:
    """
    Factory function to create and load a SecurityMaster instance.
    
    Args:
        file_path: Path to the security master CSV file
        
    Returns:
        Loaded SecurityMaster instance
    """
    sec_master = SecurityMaster()
    success = sec_master.load_csv(file_path)
    
    if not success:
        logging.error(f"Failed to load security master from {file_path}")
    
    return sec_master


if __name__ == "__main__":
    # Simple test code when run directly
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Get the path to the sample data
    script_dir = Path(__file__).parent.parent
    sample_path = script_dir / "data" / "sample" / "secmaster.csv"
    
    if sample_path.exists():
        # Load the security master
        master = load_security_master(str(sample_path))
        
        # Print some basic stats
        print(f"Loaded {master.get_security_count()} securities")
        
        # Test a lookup
        test_cusip = "037833100"  # Apple Inc.
        security = master.get_security_by_cusip(test_cusip)
        if security:
            print(f"Found security: {security['Symbol']} - {security.get('Description', 'N/A')}")
            
            # Test symbol validation
            is_valid, expected = master.validate_symbol_cusip_pair("AAPL", test_cusip)
            print(f"Symbol validation: {'Valid' if is_valid else 'Invalid'}")
            if not is_valid:
                print(f"Expected symbol: {expected}")
    else:
        print(f"Sample file not found: {sample_path}")