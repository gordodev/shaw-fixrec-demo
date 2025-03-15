"""
FIX Protocol Parser for Security Master Reconciliation Tool

This module handles parsing of FIX 4.2 messages from log files, focusing on
extracting security identifiers, quantities, and prices needed for reconciliation.
Only NewOrderSingle (35=D) and ExecutionReport (35=8) messages are processed.

Author: [Carlyle]
Date: March 14, 2025
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import os
from pathlib import Path


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class FIXMessage:
    """Data class representing essential fields of a FIX message for reconciliation purposes."""
    msg_type: str
    symbol: str
    cusip: str
    quantity: float
    price: float
    message_id: str  # For audit/tracing
    order_id: Optional[str] = None
    raw_message: Optional[str] = None  # For debugging, sanitized in production

    def calculate_exposure(self) -> float:
        """Calculate the financial exposure (price * quantity)."""
        return self.price * self.quantity


class FIXParser:
    """
    Parser for FIX 4.2 protocol messages.
    
    Extracts relevant fields from FIX messages for security reconciliation,
    with focus on symbols, CUSIPs, quantities, and prices. I would need to come up 
    with a plan to handle other versions. What happens if we start using a new FIX version? What will be the
    approval process before adding a new fix log to config? Maybe we should have a checklist, that we use
    for that.
    """
    
    # FIX message types we care about
    NEW_ORDER_SINGLE = "D"
    EXECUTION_REPORT = "8"
    
    # Required tags for our analysis
    REQUIRED_TAGS = {
        "35": None,  # MsgType
        "55": None,  # Symbol
        "48": None,  # SecurityID (CUSIP)
        "38": None,  # OrderQty
        "44": None,  # Price
        "11": None,  # ClOrdID (for Order ID)
    }
    
    # Tag mapping for readability
    TAG_MAPPING = {
        "35": "msg_type",
        "55": "symbol",
        "48": "cusip",
        "38": "quantity",
        "44": "price",
        "11": "order_id",
    }
    
    def __init__(self, delimiter: str = "|"):
        """
        Initialize FIX parser with the specified delimiter.
        
        Args:
            delimiter: Character separating FIX message fields, typically SOH (ASCII 01)
                      but often replaced with '|' in log files.
        """
        self.delimiter = delimiter
        self.message_count = 0
        self.valid_message_count = 0
        self.invalid_message_count = 0
        
        # Precompile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Precompile regex patterns for parsing FIX messages."""
        # Pattern to identify FIX version
        self.fix_version_pattern = re.compile(r"8=FIX\.4\.2")
        
        # Pattern to extract tag-value pairs
        self.tag_value_pattern = re.compile(r"(\d+)=([^{}]+)".format(self.delimiter))
    
    def parse_file(self, file_path: str) -> List[FIXMessage]:
        """
        Parse a FIX log file and extract relevant messages.
        
        Args:
            file_path: Path to the FIX log file.
            
        Returns:
            List of parsed FIX messages.
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            PermissionError: If the file cannot be accessed due to permissions.
        """
        logger.info(f"Parsing FIX messages from {file_path}")
        
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"FIX log file not found: {file_path}")
        
        messages = []
        
        try:
            with open(file_path, 'r') as file:
                for line_number, line in enumerate(file, 1):
                    try:
                        # Only process FIX 4.2 messages
                        if not self.fix_version_pattern.search(line):
                            continue
                        
                        self.message_count += 1
                        message = self.parse_message(line)
                        
                        if message:
                            messages.append(message)
                            self.valid_message_count += 1
                        else:
                            self.invalid_message_count += 1
                            
                    except Exception as e:
                        logger.warning(f"Error parsing line {line_number}: {e}")
                        self.invalid_message_count += 1
        
        except (PermissionError, IOError) as e:
            logger.error(f"Failed to open or read FIX log file: {e}")
            raise
        
        logger.info(f"Parsed {self.message_count} FIX messages: "
                    f"{self.valid_message_count} valid, {self.invalid_message_count} invalid")
        
        return messages
    
    def parse_message(self, message: str) -> Optional[FIXMessage]:
        """
        Parse a single FIX message and extract relevant fields.
        
        Args:
            message: Raw FIX message string with fields separated by the delimiter.
            
        Returns:
            FIXMessage object if successfully parsed, None otherwise.
        """
        # Extract all tag-value pairs
        tag_values = dict(self.tag_value_pattern.findall(message))
        
        # Check if this is a message type we care about
        msg_type = tag_values.get("35")
        if msg_type not in (self.NEW_ORDER_SINGLE, self.EXECUTION_REPORT):
            return None
        
        # Check for required fields
        for required_tag in self.REQUIRED_TAGS:
            if required_tag not in tag_values:
                logger.debug(f"Missing required tag {required_tag} in message")
                return None
        
        try:
            # Create a sanitized version of the raw message for logging
            # In production, we'd handle this more carefully
            sanitized_raw = self._sanitize_message(message) if logger.isEnabledFor(logging.DEBUG) else None
            
            # Create FIXMessage with the extracted fields
            return FIXMessage(
                msg_type=tag_values["35"],
                symbol=tag_values["55"],
                cusip=tag_values["48"],
                quantity=float(tag_values["38"]),
                price=float(tag_values["44"]),
                message_id=f"MSG-{self.message_count}",  # Simple ID for tracking
                order_id=tag_values.get("11"),
                raw_message=sanitized_raw
            )
        
        except (KeyError, ValueError) as e:
            logger.debug(f"Error extracting fields from message: {e}")
            return None
    
    def _sanitize_message(self, message: str) -> str:
        """
        Sanitize a FIX message by redacting sensitive fields.
        
        In a production environment, this would redact account IDs, passwords,
        and other sensitive information based on company policy.
        
        Args:
            message: Raw FIX message string.
            
        Returns:
            Sanitized message suitable for logging.
        """
        # This is a simplified version; in production we'd be more thorough
        # Placeholder for a more sophisticated implementation
        sanitized = message
        
        # Example: Redact password fields (tag 554)
        sanitized = re.sub(r"554=[^|]+", "554=REDACTED", sanitized)
        
        # Example: Redact account information
        sanitized = re.sub(r"1=[^|]+", "1=REDACTED", sanitized)
        
        return sanitized

    @staticmethod
    def get_statistics(messages: List[FIXMessage]) -> Dict[str, Any]:
        """
        Generate statistics about the parsed messages.
        
        Args:
            messages: List of parsed FIX messages.
            
        Returns:
            Dictionary containing statistics about the messages.
        """
        if not messages:
            return {"total_messages": 0}
        
        new_order_count = sum(1 for m in messages if m.msg_type == FIXParser.NEW_ORDER_SINGLE)
        exec_report_count = sum(1 for m in messages if m.msg_type == FIXParser.EXECUTION_REPORT)
        
        total_exposure = sum(message.calculate_exposure() for message in messages)
        
        return {
            "total_messages": len(messages),
            "new_order_singles": new_order_count,
            "execution_reports": exec_report_count,
            "total_exposure": total_exposure,
            "unique_symbols": len(set(message.symbol for message in messages)),
            "unique_cusips": len(set(message.cusip for message in messages)),
        }


# Placeholder for future enhancements
def process_fix_messages_parallel(file_path: str, num_workers: int = 4) -> List[FIXMessage]:
    """
    Process FIX messages in parallel using multiple workers.
    
    This is a placeholder for future enhancement to handle very large FIX logs.
    
    Args:
        file_path: Path to the FIX log file.
        num_workers: Number of parallel workers to use.
        
    Returns:
        List of parsed FIX messages.
    """
    logger.info(f"Parallel processing not yet implemented. Using single-threaded parser.")
    parser = FIXParser()
    return parser.parse_file(file_path)


# Command line testing capability
if __name__ == "__main__":
    import argparse
    import sys
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Parse FIX messages from a log file")
    parser.add_argument("file_path", help="Path to the FIX log file")
    parser.add_argument("--delimiter", default="|", help="Delimiter for FIX messages (default: |)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Parse the file
        fix_parser = FIXParser(delimiter=args.delimiter)
        messages = fix_parser.parse_file(args.file_path)
        
        # Display statistics
        stats = fix_parser.get_statistics(messages)
        print("\nMessage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Success exit
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)