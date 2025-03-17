"""
FIX Protocol Parser - parses fix msgs from log files

This module handles FIX 4.2 message parsing, extracts the important fields
we need for reconciliation like symbols, CUSIPs, price, etc.
Just handles NewOrderSingle and ExecReport messages cuz those are what matters.

Author: Carlyle
Date: March 17, 2025
"""

import re
import os
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import time

# setup logging
logger = logging.getLogger(__name__)


@dataclass
class FIXMessage:
    """Data class for important FIX message fields"""
    msg_type: str
    symbol: str
    cusip: str
    quantity: float
    price: float
    message_id: str  # tracking/audit id
    order_id: Optional[str] = None
    raw_message: Optional[str] = None  # for debugging, will redact in prod

    def calculate_exposure(self) -> float:
        """Gets exposure (price * qty)"""
        return self.price * self.quantity


class FIXParser:
    """
    Parser for FIX 4.2 protocol messages.
    
    Extracts fields from FIX msgs for reconciliation. Focus on symbols, 
    CUSIPs, quantities, prices etc. Note: will need to handle other FIX
    versions in the future, should make it configurable.
    """
    
    # FIX msg types we care about
    NEW_ORDER_SINGLE = "D"
    EXECUTION_REPORT = "8"
    
    # Tags we need to extract
    REQUIRED_TAGS = {
        "35": None,  # MsgType
        "55": None,  # Symbol
        "48": None,  # SecurityID (CUSIP)
        "38": None,  # OrderQty
        "44": None,  # Price
        "11": None,  # ClOrdID
    }
    
    # Tag mapping to field names
    TAG_MAPPING = {
        "35": "msg_type",
        "55": "symbol",
        "48": "cusip",
        "38": "quantity",
        "44": "price",
        "11": "order_id",
    }
    
    def __init__(self, delimiter: str = "|", max_line_length: int = 50000):
        """
        Initialize FIX parser with delimiter
        
        Args:
            delimiter: Character separating FIX fields, usually SOH (ASCII 01)
                      but often logs replace it with '|' or other character
            max_line_length: Max length of line to parse (safeguard)
        """
        self.delimiter = delimiter
        self.message_count = 0
        self.valid_message_count = 0
        self.invalid_message_count = 0
        self.max_line_length = max_line_length
        self.warning_logged = False  # prevents log spam
        
        # Precompile regex patterns for perf
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns - way faster than doing it each time"""
        # Pattern to identify FIX version - really just looking for 8=FIX
        self.fix_version_pattern = re.compile(r"8=FIX\.4\.[23]")  # match 4.2 or 4.3
        
        # Pattern to extract tag-value pairs - finds all tag=value with our delimiter
        self.tag_value_pattern = re.compile(r"(\d+)=([^{}]+)".format(self.delimiter))
    
    def parse_file(self, file_path: str) -> List[FIXMessage]:
        """
        Parse a FIX log file, extract msg we care about.
        
        Args:
            file_path: Path to the FIX log file.
            
        Returns:
            List of parsed FIX messages (just the ones we want)
            
        Raises:
            FileNotFoundError: If file doesn't exist.
            PermissionError: If access denied to the file.
        """
        logger.info(f"Parsing FIX messages from {file_path}")
        start_time = time.time()
        
        # Basic file existence check
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"FIX log file not found: {file_path}")
        
        # Basic file sanity check - just a quick peek to see if it looks like FIX
        try:
            with open(file_path, 'r', errors='replace') as f:
                first_chunk = f.read(min(8192, os.path.getsize(file_path)))  # First 8KB or whole file
                
                # See if there's any FIX message patterns in there
                if not self.fix_version_pattern.search(first_chunk):
                    logger.warning(f"File does not look like a FIX log: {file_path}")
                    logger.warning(f"First few bytes: {first_chunk[:100]}...")
                    logger.warning("Will continue parsing, but expect poor results")
        except Exception as e:
            # Non-critical - just log and continue
            logger.warning(f"Exception during file format check: {e}")
        
        messages = []
        malformed_count = 0
        line_count = 0
        
        try:
            with open(file_path, 'r', errors='replace') as file:
                for line_num, line in enumerate(file, 1):
                    line_count += 1
                    
                    # Skip empty lines - happens sometimes in logs
                    if not line.strip():
                        continue
                    
                    # Check line isn't absurdly long (prevents memory attacks)
                    if len(line) > self.max_line_length:
                        if not self.warning_logged:
                            logger.warning(f"Line {line_num} exceeds max length ({self.max_line_length}), will truncate this and future long lines")
                            self.warning_logged = True
                        line = line[:self.max_line_length]
                    
                    try:
                        # Only process if it has FIX header pattern
                        if not self.fix_version_pattern.search(line):
                            malformed_count += 1
                            if malformed_count <= 3:  # Log first few
                                logger.debug(f"Skipping non-FIX line {line_num}: {line[:50]}...")
                            continue
                        
                        self.message_count += 1
                        
                        # Parse it and see if we get something useful
                        message = self.parse_message(line)
                        
                        if message:
                            messages.append(message)
                            self.valid_message_count += 1
                        else:
                            self.invalid_message_count += 1
                            
                    except Exception as e:
                        self.invalid_message_count += 1
                        if self.invalid_message_count <= 5:  # Limit log spam
                            logger.warning(f"Error parsing line {line_num}: {str(e)[:100]}")
        
        except (PermissionError, IOError) as e:
            logger.error(f"Failed to open or read FIX log: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing log: {e}")
            # Don't re-raise - return what we got
        
        # Calculate stats & timing
        elapsed = time.time() - start_time
        parse_rate = line_count / elapsed if elapsed > 0 else 0
        
        logger.info(f"Parsed {self.message_count} FIX messages in {elapsed:.2f}s ({parse_rate:.0f} lines/sec): "
                    f"{self.valid_message_count} valid, {self.invalid_message_count} invalid")
        
        if self.message_count > 0 and self.valid_message_count / self.message_count < 0.1:
            logger.warning(f"Very low parsing success rate ({self.valid_message_count}/{self.message_count}). "
                          f"Check if this is really a FIX log file!")
        
        return messages
    
    def parse_message(self, message: str) -> Optional[FIXMessage]:
        """
        Parse a single FIX message line into structured object.
        
        Args:
            message: Raw FIX message string with fields separated by delimiter
            
        Returns:
            FIXMessage object if successfully parsed, None if not useful
        """
        # Extract all tag-value pairs - this finds all the field tags and values
        tag_values = dict(self.tag_value_pattern.findall(message))
        
        # Skip if we don't have a message type tag (35)
        if "35" not in tag_values:
            return None
        
        # Only care about NewOrderSingle and ExecReport for now
        msg_type = tag_values.get("35")
        if msg_type not in (self.NEW_ORDER_SINGLE, self.EXECUTION_REPORT):
            return None
        
        # Quick check for required fields - missing anything important?
        for required_tag in self.REQUIRED_TAGS:
            if required_tag not in tag_values:
                # Could log more specifically what's missing here if needed
                return None
        
        try:
            # Create sanitized version for debug logs (only if debug logging enabled)
            sanitized_raw = self._sanitize_message(message) if logger.isEnabledFor(logging.DEBUG) else None
            
            # For quantities & prices, handle possibility of localized formats 
            # with commas as decimal separators (e.g. European style)
            quantity_str = tag_values["38"].replace(',', '.')
            price_str = tag_values["44"].replace(',', '.')
            
            # Validate that quantity and price can be converted to floats
            try:
                quantity = float(quantity_str)
                price = float(price_str)
            except ValueError:
                logger.warning(f"Invalid quantity ({quantity_str}) or price ({price_str})")
                return None
            
            # Create FIXMessage with the extracted fields
            return FIXMessage(
                msg_type=tag_values["35"],
                symbol=tag_values["55"],
                cusip=tag_values["48"],
                quantity=quantity,
                price=price,
                message_id=f"MSG-{self.message_count}",  # Simple ID for tracking
                order_id=tag_values.get("11"),
                raw_message=sanitized_raw
            )
        
        except Exception as e:
            logger.debug(f"Error extracting fields from message: {e}")
            return None
    
    def _sanitize_message(self, message: str) -> str:
        """
        Clean up FIX msg by redacting sensitive fields.
        
        In prod we would redact account IDs, passwords, other PII.
        TODO: Add more fields to redact based on compliance reqs.
        
        Args:
            message: Raw FIX message
            
        Returns:
            Safe-to-log version
        """
        # This is simplified - in prod we'd redact more based on compliance rules
        sanitized = message
        
        # Redact password fields (tag 554)
        sanitized = re.sub(r"554=[^{}]+".format(self.delimiter), "554=REDACTED", sanitized)
        
        # Redact account information (tag 1)
        sanitized = re.sub(r"1=[^{}]+".format(self.delimiter), "1=REDACTED", sanitized)
        
        # NOTE: Should add more field redactions here - this isnt enough for production
        
        return sanitized

    @staticmethod
    def get_statistics(messages: List[FIXMessage]) -> Dict[str, Any]:
        """
        Generate stats about the parsed messages.
        
        Args:
            messages: List of parsed FIX messages.
            
        Returns:
            Dictionary with stats
        """
        if not messages:
            return {"total_messages": 0}
        
        # Get different message type counts
        new_order_count = sum(1 for m in messages if m.msg_type == FIXParser.NEW_ORDER_SINGLE)
        exec_report_count = sum(1 for m in messages if m.msg_type == FIXParser.EXECUTION_REPORT)
        
        # Calculate total exposure
        total_exposure = sum(message.calculate_exposure() for message in messages)
        
        # Count unique symbols and CUSIPs
        unique_symbols = set(message.symbol for message in messages)
        unique_cusips = set(message.cusip for message in messages)
        
        return {
            "total_messages": len(messages),
            "new_order_singles": new_order_count,
            "execution_reports": exec_report_count,
            "total_exposure": total_exposure,
            "unique_symbols": len(unique_symbols),
            "unique_cusips": len(unique_cusips),
        }


# TODO: Implement in future if perf becomes an issue - this just is a placeholder for now
def process_fix_messages_parallel(file_path: str, num_workers: int = 4) -> List[FIXMessage]:
    """
    Process FIX messages in parallel threads for huge files.
    
    Would need to implement for super large FIX logs - just a placeholder for now.
    Need some thought on how to split up the file since messages can span lines.
    
    Args:
        file_path: Path to the FIX log file.
        num_workers: How many parallel workers to use.
        
    Returns:
        List of parsed FIX messages.
    """
    logger.info(f"Parallel processing not implemented yet. Using single-threaded parser.")
    parser = FIXParser()
    return parser.parse_file(file_path)


# Simple CLI for testing
if __name__ == "__main__":
    import argparse
    import sys
    
    # Setup CLI args
    parser = argparse.ArgumentParser(description="Parse FIX messages from a log file")
    parser.add_argument("file_path", help="Path to the FIX log file")
    parser.add_argument("--delimiter", default="|", help="Delimiter for FIX messages (default: |)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--max-line", type=int, default=50000, help="Maximum line length to parse")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Parse the file
        fix_parser = FIXParser(delimiter=args.delimiter, max_line_length=args.max_line)
        messages = fix_parser.parse_file(args.file_path)
        
        # Display stats
        stats = fix_parser.get_statistics(messages)
        print("\nMessage Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Sample of messages
        if len(messages) > 0:
            print("\nSample message content:")
            for i, msg in enumerate(messages[:3]):  # Show first 3
                print(f"Message {i+1}:")
                print(f"  Type: {msg.msg_type}")
                print(f"  Symbol: {msg.symbol}")
                print(f"  CUSIP: {msg.cusip}")
                print(f"  Quantity: {msg.quantity}")
                print(f"  Price: {msg.price}")
                print(f"  Order ID: {msg.order_id}")
        
        # Success exit
        sys.exit(0)
        
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)