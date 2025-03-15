#!/usr/bin/env python3
"""
Reporter module for FIX Symbol Discrepancy Checker.

This module handles the generation of reports for symbol discrepancies found
between FIX messages and the security master database. It produces CSV reports
sorted by financial exposure and includes summary statistics.

Typical usage:
    from reporter import DiscrepancyReporter
    
    reporter = DiscrepancyReporter(output_path="reports/discrepancies.csv")
    reporter.generate_report(discrepancies, total_exposure=1234567.89)

    Author: Carlyle
"""

import os
import csv
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger(__name__)


class DiscrepancyReporter:
    """
    Handles the creation and formatting of discrepancy reports.
    
    This class is responsible for generating CSV reports that detail
    symbol discrepancies between FIX messages and the security master,
    along with their financial exposure.
    """
    
    def __init__(self, output_path: str):
        """
        Initialize the reporter with an output path.
        
        Args:
            output_path: Path where the report will be saved
        """
        self.output_path = output_path
        self._ensure_output_directory()
    
    def _ensure_output_directory(self) -> None:
        """Create the output directory if it doesn't exist."""
        output_dir = os.path.dirname(self.output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    def generate_report(self, 
                        discrepancies: List[Dict[str, Any]], 
                        total_exposure: float) -> None:
        """
        Generate a CSV report of discrepancies sorted by exposure.
        
        Args:
            discrepancies: List of discrepancy dictionaries
            total_exposure: Total financial exposure from all discrepancies
        """
        if not discrepancies:
            logger.info("No discrepancies found, generating empty report")
            self._write_empty_report()
            return
            
        # Sort discrepancies by Exposure (highest first)
        # Note: analyzer.py uses 'Exposure' with capital E
        sorted_discrepancies = sorted(
            discrepancies, 
            key=lambda x: x.get('Exposure', 0), 
            reverse=True
        )
        
        logger.info(f"Generating report with {len(discrepancies)} discrepancies")
        self._write_report(sorted_discrepancies, total_exposure)
        logger.info(f"Report generated successfully at: {self.output_path}")
    
    def _write_report(self, 
                      discrepancies: List[Dict[str, Any]], 
                      total_exposure: float) -> None:
        """
        Write the discrepancies to a CSV file with a summary footer.
        
        Args:
            discrepancies: Sorted list of discrepancy dictionaries
            total_exposure: Total financial exposure
        """
        try:
            with open(self.output_path, 'w', newline='') as csvfile:
                # Determine fieldnames from the first discrepancy
                fieldnames = list(discrepancies[0].keys())
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(discrepancies)
                
                # Add summary information as comments at the bottom
                csvfile.write(f"\n# Report Summary\n")
                csvfile.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                csvfile.write(f"# Total Discrepancies: {len(discrepancies)}\n")
                csvfile.write(f"# Total Financial Exposure: ${total_exposure:,.2f}\n")
                
                # Add statistics on highest exposure items
                if len(discrepancies) >= 3:
                    top_three = sum(d.get('Exposure', 0) for d in discrepancies[:3])
                    top_three_pct = (top_three / total_exposure) * 100 if total_exposure else 0
                    csvfile.write(f"# Top 3 Items Represent: ${top_three:,.2f} ({top_three_pct:.1f}% of total exposure)\n")

                # Additional metadata for audit purposes
                csvfile.write(f"# Report ID: {self._generate_report_id()}\n")
        except Exception as e:
            logger.error(f"Error writing report to {self.output_path}: {e}")
            raise
    
    def _write_empty_report(self) -> None:
        """Write an empty report when no discrepancies are found."""
        try:
            with open(self.output_path, 'w', newline='') as csvfile:
                csvfile.write("# No Symbol Discrepancies Found\n")
                csvfile.write(f"# Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                csvfile.write(f"# Report ID: {self._generate_report_id()}\n")
        except Exception as e:
            logger.error(f"Error writing empty report to {self.output_path}: {e}")
            raise
    
    def _generate_report_id(self) -> str:
        """
        Generate a unique report ID for audit purposes.
        
        Returns:
            A string containing a timestamp-based report ID
        """
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"FIXREC-{timestamp}"
    
    def generate_summary_only(self, 
                             discrepancy_count: int, 
                             total_exposure: float) -> None:
        """
        Generate a summary-only report without individual discrepancies.
        
        This is a placeholder for a future enhancement that would generate
        a more concise report for management or summary purposes.
        
        Args:
            discrepancy_count: Number of discrepancies found
            total_exposure: Total financial exposure amount
        """
        # Placeholder for future enhancement
        logger.info("Summary-only report requested (not implemented yet)")
        pass
    
    @staticmethod
    def format_currency(amount: Union[float, int]) -> str:
        """
        Format a number as a currency string.
        
        Args:
            amount: The amount to format
            
        Returns:
            Formatted currency string
        """
        return f"${amount:,.2f}"
    
    def export_to_html(self, 
                      discrepancies: List[Dict[str, Any]], 
                      total_exposure: float,
                      output_path: Optional[str] = None) -> None:
        """
        Export the report to HTML format (placeholder).
        
        This is a placeholder for a future enhancement to generate HTML reports
        with more visualization options.
        
        Args:
            discrepancies: List of discrepancy dictionaries
            total_exposure: Total financial exposure
            output_path: Optional custom output path for HTML
        """
        # Placeholder for future enhancement
        logger.info("HTML export requested (not implemented yet)")
        pass


# Simplified concrete implementation for quick deployment
class BasicDiscrepancyReporter(DiscrepancyReporter):
    """
    Simplified reporter implementation focused on core functionality.
    
    This class implements only the essential reporting features to ensure
    the project can be delivered quickly, while maintaining the structure
    for future enhancements.
    """
    
    def generate_report(self, 
                        discrepancies: List[Dict[str, Any]], 
                        total_exposure: float) -> None:
        """
        Generate a simple CSV report of discrepancies sorted by exposure.
        
        Args:
            discrepancies: List of discrepancy dictionaries
            total_exposure: Total financial exposure from all discrepancies
        """
        if not discrepancies:
            logger.info("No discrepancies found, generating empty report")
            with open(self.output_path, 'w', newline='') as csvfile:
                csvfile.write("# No Symbol Discrepancies Found\n")
                return
            
        # Sort discrepancies by Exposure (highest first)
        # Note: analyzer.py uses 'Exposure' with capital E
        sorted_discrepancies = sorted(
            discrepancies, 
            key=lambda x: x.get('Exposure', 0), 
            reverse=True
        )
        
        try:
            with open(self.output_path, 'w', newline='') as csvfile:
                fieldnames = list(sorted_discrepancies[0].keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(sorted_discrepancies)
                
                # Add summary information as a comment
                csvfile.write(f"\n# Total Discrepancies: {len(discrepancies)}\n")
                csvfile.write(f"# Total Financial Exposure: ${total_exposure:,.2f}\n")
            
            logger.info(f"Report successfully generated at: {self.output_path}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise