"""
Models for FIX reconciliation tool - contains data structures 
for handling security master entries, FIX messages, and discrepancies.

Author: Carlyle
"""

from typing import Dict, List, Optional, Any
from datetime import datetime


class SecurityMaster:
    """Security reference data loaded from master file."""
    
    def __init__(self, cusip, symbol, description=None, exchange=None, 
                 region=None, asset_class=None, currency=None):
        # Primary identifier used for matching
        self.cusip = cusip
        # Market symbol that should be used in FIX messages
        self.symbol = symbol
        # Optional descriptive fields
        self.description = description
        self.exchange = exchange
        self.region = region
        self.asset_class = asset_class
        self.currency = currency
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            'CUSIP': self.cusip,
            'Symbol': self.symbol,
            'Description': self.description or '',
            'Exchange': self.exchange or '',
            'Region': self.region or '',
            'AssetClass': self.asset_class or '',
            'Currency': self.currency or ''
        }


class FIXMessage:
    """
    Parsed FIX protocol message with key trading fields.
    
    This class represents a parsed FIX message with fields needed for analysis.
    NOTE: This implementation is compatible with fix_parser.FIXMessage for seamless integration.
    """
    
    def __init__(self, msg_type, symbol, cusip, quantity, price,
                 message_id, order_id=None, exec_id=None, side=None, raw_message=None):
        # Essential fields for reconciliation (matching fix_parser.FIXMessage structure)
        self.msg_type = msg_type      # 'D' = NewOrderSingle, '8' = ExecutionReport
        self.symbol = symbol          # Market symbol (tag 55)
        self.cusip = cusip            # Security identifier (tag 48)
        self.quantity = float(quantity)  # Order quantity (tag 38)
        self.price = float(price)        # Order price (tag 44)
        self.message_id = message_id  # For tracking/auditing
        
        # Additional fields
        self.order_id = order_id      # ClOrdID (tag 11)
        self.exec_id = exec_id        # ExecID (tag 17)
        self.side = side              # Side (tag 54): '1'=Buy, '2'=Sell
        self.raw_message = raw_message  # Original message for auditing
        
        # Timestamp for when we processed this message
        self.timestamp = datetime.now()
    
    def calculate_exposure(self) -> float:
        """Calculate financial exposure (price * quantity)."""
        return self.price * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format expected by analyzer.
        """
        return {
            'msg_type': self.msg_type,
            'Symbol': self.symbol,
            'CUSIP': self.cusip, 
            'Quantity': self.quantity,
            'Price': self.price,
            'message_id': self.message_id,
            'order_id': self.order_id
        }


class Discrepancy:
    """Symbol mismatch between FIX message and security master."""
    
    def __init__(self, cusip, fix_symbol, master_symbol, quantity, price,
                 exchange=None, region=None, order_id=None):
        self.cusip = cusip
        self.fix_symbol = fix_symbol        # What the trading system used
        self.master_symbol = master_symbol  # What it should have used
        self.quantity = float(quantity)
        self.price = float(price)
        
        # Optional contextual fields
        self.exchange = exchange
        self.region = region
        self.order_id = order_id
        
        # When this discrepancy was found
        self.timestamp = datetime.now()
    
    def exposure(self) -> float:
        """Calculate financial impact of this discrepancy."""
        return self.price * self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        result = {
            'CUSIP': self.cusip,
            'FIXSymbol': self.fix_symbol,
            'MasterSymbol': self.master_symbol,
            'Quantity': self.quantity,
            'Price': self.price,
            'Exposure': self.exposure()  # Capitalized to match analyzer.py
        }
        
        # Add optional fields if available
        if self.exchange:
            result['Exchange'] = self.exchange
        if self.region:
            result['Region'] = self.region
        if self.order_id:
            result['OrderID'] = self.order_id
            
        return result


class ReconReport:
    """Results container for a reconciliation run."""
    
    def __init__(self, sec_master_path=None, fix_log_path=None):
        self.discrepancies = []
        self.run_time = datetime.now()
        self.sec_master_path = sec_master_path
        self.fix_log_path = fix_log_path
    
    def add_discrepancy(self, discrepancy: Discrepancy) -> None:
        """Add a discrepancy to the report."""
        self.discrepancies.append(discrepancy)
    
    def total_count(self) -> int:
        """Get total number of discrepancies."""
        return len(self.discrepancies)
    
    def total_exposure(self) -> float:
        """Sum of financial exposure across all discrepancies."""
        return sum(d.exposure() for d in self.discrepancies)
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        return {
            'TotalDiscrepancies': self.total_count(),
            'TotalExposure': self.total_exposure(),
            'RunTimestamp': self.run_time.isoformat(),
            'SecurityMasterPath': self.sec_master_path or 'N/A',
            'FIXLogPath': self.fix_log_path or 'N/A'
        }
    
    def to_records(self) -> List[Dict[str, Any]]:
        """Convert all discrepancies to dictionaries for reporting."""
        return [d.to_dict() for d in self.discrepancies]