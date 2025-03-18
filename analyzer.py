#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX Message Analyzer for Symbol Discrepancies

This module analyzes parsed FIX messages against security master data to identify
symbol mismatches and calculate financial exposure from such discrepancies.

Focus is on detecting symbols that don't match their CUSIP, which could indicate
missed corporate actions or reference data issues.

Author: Carlyle
"""

import logging
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

# Configure module logger
logger = logging.getLogger(__name__)


class DiscrepancyAnalyzer:
    """
    Analyzes FIX messages for symbol discrepancies against security master data.
    
    This analyzer identifies cases where the symbol in a FIX message doesn't match
    the expected symbol for the given CUSIP in the security master, which could
    indicate missed corporate actions or reference data issues.
    """
    
    def __init__(self, security_master: Dict[str, Dict[str, Any]]):
        """
        Initialize with security master data.
        
        Args:
            security_master: Dictionary mapping CUSIP to security details
                             (should include at least a 'Symbol' field)
        """
        self.security_master = security_master
        self.discrepancies = []
        self.total_exposure = 0.0
        self.unknown_cusips = set()
        
        # For potential future extension - metrics by region/exchange
        self.region_metrics = defaultdict(lambda: {"count": 0, "exposure": 0.0})
        self.exchange_metrics = defaultdict(lambda: {"count": 0, "exposure": 0.0})
        
        logger.info(f"Initialized analyzer with {len(security_master)} securities")
    
    def analyze_messages(self, fix_messages: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Analyze FIX messages to identify symbol mismatches and calculate exposure.
        
        Args:
            fix_messages: List of parsed FIX messages containing Symbol, CUSIP, 
                         Quantity, and Price
        
        Returns:
            Tuple containing (list of discrepancies, total financial exposure)
        """
        self.discrepancies = []
        self.total_exposure = 0.0
        self.unknown_cusips = set()
        
        logger.info(f"Analyzing {len(fix_messages)} FIX messages for discrepancies")
        
        for msg in fix_messages:
            self._check_message(msg)
            
        # Sort discrepancies by exposure (highest first)
        self.discrepancies.sort(key=lambda x: x["Exposure"], reverse=True)
        
        logger.info(f"Analysis complete. Found {len(self.discrepancies)} discrepancies")
        logger.info(f"Total financial exposure: ${self.total_exposure:.2f}")
        
        if self.unknown_cusips:
            logger.warning(f"Found {len(self.unknown_cusips)} unknown CUSIPs in FIX messages")
            
        return self.discrepancies, self.total_exposure
    
    def _check_message(self, msg: Dict[str, Any]) -> None:
        """
        Check a single FIX message for symbol discrepancies.
        
        Args:
            msg: Parsed FIX message dictionary
        """
        # Extract necessary fields
        try:
            cusip = msg.get("CUSIP")
            fix_symbol = msg.get("Symbol")
            quantity = float(msg.get("Quantity", 0))
            price = float(msg.get("Price", 0))
        except (ValueError, TypeError) as e:
            logger.error(f"Error processing message fields: {e}, message: {msg}")
            return
            
        # Skip messages with missing critical fields
        if not all([cusip, fix_symbol, quantity, price]):
            logger.debug(f"Skipping message with missing fields: {msg}")
            return
            
        # Look up the CUSIP in the security master
        if cusip in self.security_master:
            master_data = self.security_master[cusip]
            master_symbol = master_data.get("Symbol")
            
            # Check for symbol mismatch
            if master_symbol and master_symbol != fix_symbol:
                # Calculate financial exposure
                exposure = quantity * price
                
                # Create discrepancy record
                discrepancy = {
                    "CUSIP": cusip,
                    "MasterSymbol": master_symbol,
                    "FIXSymbol": fix_symbol,
                    "Quantity": quantity,
                    "Price": price,
                    "Exposure": exposure,
                    "DiscrepancyType": "Symbol Mismatch"
                }
                
                # Add optional fields if available in security master
                for field in ["Exchange", "Region", "Asset Class", "Currency"]:
                    if field in master_data:
                        discrepancy[field] = master_data[field]
                
                self.discrepancies.append(discrepancy)
                self.total_exposure += exposure
                
                # Update region/exchange metrics if available
                if "Region" in master_data:
                    region = master_data["Region"]
                    self.region_metrics[region]["count"] += 1
                    self.region_metrics[region]["exposure"] += exposure
                    
                if "Exchange" in master_data:
                    exchange = master_data["Exchange"]
                    self.exchange_metrics[exchange]["count"] += 1
                    self.exchange_metrics[exchange]["exposure"] += exposure
                
                logger.debug(f"Found discrepancy: CUSIP={cusip}, "
                            f"FIX Symbol={fix_symbol}, "
                            f"Master Symbol={master_symbol}, "
                            f"Exposure=${exposure:.2f}")
        else:
            # CUSIP not found in security master - add as a discrepancy
            self.unknown_cusips.add(cusip)
            
            # Create discrepancy record for unknown CUSIP
            exposure = quantity * price
            
            discrepancy = {
                "CUSIP": cusip,
                "MasterSymbol": "UNKNOWN",  # CUSIP not in security master
                "FIXSymbol": fix_symbol,
                "Quantity": quantity,
                "Price": price,
                "Exposure": exposure,
                "DiscrepancyType": "Unknown CUSIP"  # Add a type to distinguish these
            }
            
            self.discrepancies.append(discrepancy)
            self.total_exposure += exposure
            logger.debug(f"Unknown CUSIP in FIX message: {cusip}, added as discrepancy")
    
    def get_region_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics on discrepancies grouped by region.
        
        Returns:
            Dictionary mapping regions to metrics (count and exposure)
        """
        return dict(self.region_metrics)
    
    def get_exchange_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metrics on discrepancies grouped by exchange.
        
        Returns:
            Dictionary mapping exchanges to metrics (count and exposure)
        """
        return dict(self.exchange_metrics)
    
    def get_unknown_cusips(self) -> List[str]:
        """
        Get list of CUSIPs found in FIX messages but not in security master.
        
        Returns:
            List of unknown CUSIP identifiers
        """
        return list(self.unknown_cusips)


def analyze_discrepancies(fix_messages: List[Dict[str, Any]], 
                         security_master: Dict[str, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
    """
    Convenience function to analyze FIX messages for symbol discrepancies.
    
    Args:
        fix_messages: List of parsed FIX messages
        security_master: Dictionary of security master data keyed by CUSIP
        
    Returns:
        Tuple containing (list of discrepancies, total financial exposure)
    """
    analyzer = DiscrepancyAnalyzer(security_master)
    return analyzer.analyze_messages(fix_messages)


def generate_summary_report(discrepancies: List[Dict[str, Any]], 
                           total_exposure: float,
                           group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Generate a summary report of discrepancies, optionally grouped by a field.
    
    Args:
        discrepancies: List of discrepancy records
        total_exposure: Total financial exposure
        group_by: Optional field to group by (e.g., 'Exchange', 'Region')
        
    Returns:
        DataFrame containing the summary report
    """
    if not discrepancies:
        return pd.DataFrame()
    
    df = pd.DataFrame(discrepancies)
    
    if group_by and group_by in df.columns:
        # Group by the specified field
        summary = df.groupby(group_by).agg({
            'CUSIP': 'count',
            'Exposure': 'sum'
        }).reset_index()
        
        # Rename columns for clarity
        summary.rename(columns={'CUSIP': 'DiscrepancyCount'}, inplace=True)
        
        # Add percentage of total exposure
        summary['ExposurePercentage'] = (summary['Exposure'] / total_exposure * 100).round(2)
        
        return summary.sort_values('Exposure', ascending=False)
    else:
        # Just return the raw discrepancies
        return df


def calculate_risk_score(discrepancy: Dict[str, Any]) -> float:
    """
    Calculate a risk score for a discrepancy based on various factors.
    
    This is a placeholder for more sophisticated risk scoring that could consider:
    - Financial exposure
    - Time since symbol change
    - Market volatility of the security
    - Number of affected orders
    
    Args:
        discrepancy: A discrepancy record
        
    Returns:
        Risk score (higher means higher risk)
    """
    # Placeholder implementation - just use exposure as risk score
    return float(discrepancy.get('Exposure', 0))


# Placeholder for future enhancements
def analyze_patterns(discrepancies: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in discrepancies to identify potential systemic issues.
    
    This could look for clusters of discrepancies by time, region, exchange, etc.
    to identify if there's a pattern suggesting a systematic data issue.
    
    Args:
        discrepancies: List of discrepancy records
        
    Returns:
        Dictionary with pattern analysis results
    """
    # Placeholder for future implementation
    logger.info("Pattern analysis not implemented in this version")
    return {"implemented": False, "message": "Pattern analysis planned for future versions"}


if __name__ == "__main__":
    # Simple test code that could be used during development
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example security master (minimal)
    test_security_master = {
        "037833100": {"Symbol": "AAPL", "Exchange": "NASDAQ", "Region": "US"},
        "594918104": {"Symbol": "MSFT", "Exchange": "NASDAQ", "Region": "US"}
    }
    
    # Example FIX messages (minimal)
    test_messages = [
        {"CUSIP": "037833100", "Symbol": "APL", "Quantity": 1000, "Price": 150.0},
        {"CUSIP": "594918104", "Symbol": "MSFT", "Quantity": 500, "Price": 200.0}
    ]
    
    # Run analysis
    test_discrepancies, test_exposure = analyze_discrepancies(test_messages, test_security_master)
    
    print(f"Found {len(test_discrepancies)} discrepancies with total exposure: ${test_exposure:.2f}")
    for disc in test_discrepancies:
        print(f"CUSIP: {disc['CUSIP']}, FIX: {disc['FIXSymbol']}, Master: {disc['MasterSymbol']}, "
              f"Exposure: ${disc['Exposure']:.2f}")