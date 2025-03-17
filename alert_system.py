#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alert System for FIX Symbol Discrepancy Checker

This module provides alerting mechanisms to protect production systems when
processing extremely large files. It enables safe handling of security master files
up to 200GB without risking production server stability.

Key features:
- File size validation with configurable thresholds
- Resource usage monitoring and alerting
- Scheduled execution recommendations based on system load
- Tiered alerting (warnings vs. critical alerts)
- Extensible framework for future integration with monitoring systems

Pushing the performance envelope a bit more before doing rapid QA, then UAT deployment stage
NOTE: This code has been built to be PROD friendly, not actually PROD ready. Trying to see how far 
      we can push it, in a short period of time strictly as a demonstration of my skills and abilities. 
      PROD version of this would take a lot more time and be much more complex, and would need EXTENSIVE 
      testing and QA before deployment. PROD must be done with caution and care, and with a lot of 
      testing. Financial risks are too high to do otherwise.

Author: Carlyle
Date: March 16, 2025
"""

import os
import sys
import time
import logging
import psutil
import json
import threading
import smtplib
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
from pathlib import Path
from email.message import EmailMessage


# Configure logger
logger = logging.getLogger(__name__)


class AlertLevel:
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class AlertType:
    """Types of alerts the system can generate."""
    FILE_SIZE = "FILE_SIZE"
    MEMORY_USAGE = "MEMORY_USAGE"
    CPU_USAGE = "CPU_USAGE"
    DISK_SPACE = "DISK_SPACE"
    EXECUTION_TIME = "EXECUTION_TIME"
    PROCESS_FAIL = "PROCESS_FAIL"


class AlertConfig:
    """Default configuration values for the alert system."""
    # File size thresholds (in GB)
    MAX_FILE_SIZE = 200.0  # Maximum supported file size
    WARNING_FILE_SIZE = 50.0  # Size at which to issue warnings
    LARGE_FILE_THRESHOLD = 20.0  # Size at which to apply special handling

    # Resource thresholds
    MAX_MEMORY_PERCENT = 85  # Critical memory threshold
    WARNING_MEMORY_PERCENT = 75  # Warning memory threshold
    MAX_CPU_PERCENT = 90  # Critical CPU threshold
    WARNING_CPU_PERCENT = 70  # Warning CPU threshold
    MIN_FREE_DISK_PERCENT = 15  # Minimum free disk space

    # Alert configuration
    EMAIL_ENABLED = False  # Whether to send email alerts
    SLACK_ENABLED = False  # Whether to send Slack alerts
    SYSLOG_ENABLED = False  # Whether to log to syslog
    
    # Execution management
    ENABLE_SCHEDULED_EXECUTION = False  # Whether to allow scheduling


class ResourceStatus:
    """Status of system resources."""
    def __init__(self):
        self.memory_percent = 0.0
        self.cpu_percent = 0.0
        self.disk_free_percent = 0.0
        self.process_memory_mb = 0.0
        self.timestamp = datetime.now()
        
    def update(self):
        """Update all resource metrics."""
        try:
            # System-wide metrics
            mem = psutil.virtual_memory()
            self.memory_percent = mem.percent
            self.cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.disk_free_percent = 100 - disk.percent
            
            # Process-specific metrics
            process = psutil.Process(os.getpid())
            self.process_memory_mb = process.memory_info().rss / (1024 * 1024)
            
            self.timestamp = datetime.now()
            return True
        except Exception as e:
            logger.error(f"Failed to update resource status: {e}")
            return False
    
    def is_memory_critical(self) -> bool:
        """Check if memory usage is at critical level."""
        return self.memory_percent >= AlertConfig.MAX_MEMORY_PERCENT
    
    def is_memory_warning(self) -> bool:
        """Check if memory usage is at warning level."""
        return self.memory_percent >= AlertConfig.WARNING_MEMORY_PERCENT
    
    def is_cpu_critical(self) -> bool:
        """Check if CPU usage is at critical level."""
        return self.cpu_percent >= AlertConfig.MAX_CPU_PERCENT
    
    def is_cpu_warning(self) -> bool:
        """Check if CPU usage is at warning level."""
        return self.cpu_percent >= AlertConfig.WARNING_CPU_PERCENT
    
    def is_disk_critical(self) -> bool:
        """Check if disk space is at critical level."""
        return self.disk_free_percent <= AlertConfig.MIN_FREE_DISK_PERCENT
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary for serialization."""
        return {
            "memory_percent": self.memory_percent,
            "cpu_percent": self.cpu_percent,
            "disk_free_percent": self.disk_free_percent,
            "process_memory_mb": self.process_memory_mb,
            "timestamp": self.timestamp.isoformat()
        }


class Alert:
    """Represents a system alert."""
    def __init__(self, 
                alert_type: str, 
                level: str, 
                message: str, 
                details: Optional[Dict[str, Any]] = None):
        """
        Initialize an alert.
        
        Args:
            alert_type: Type of alert (from AlertType)
            level: Severity level (from AlertLevel)
            message: Alert message
            details: Additional context for the alert
        """
        self.alert_type = alert_type
        self.level = level
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now()
        self.id = f"{int(time.time())}-{id(self)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.alert_type,
            "level": self.level,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat()
        }
    
    def __str__(self) -> str:
        """String representation of alert."""
        return f"[{self.level}] {self.alert_type}: {self.message}"


class AlertDispatcher:
    """
    Dispatches alerts to configured notification channels.
    
    This is currently a placeholder with hooks for future integration
    with external alerting systems (email, Slack, etc.).
    """
    
    def __init__(self):
        """Initialize the alert dispatcher."""
        self.email_config = None
        self.slack_config = None
        self.recent_alerts = []  # Track recent alerts to avoid duplication
        self.max_recent_alerts = 100
    
    def dispatch(self, alert: Alert) -> bool:
        """
        Dispatch an alert to all configured channels.
        
        Args:
            alert: Alert to dispatch
            
        Returns:
            True if dispatched successfully to at least one channel
        """
        # Log all alerts
        if alert.level == AlertLevel.INFO:
            logger.info(str(alert))
        elif alert.level == AlertLevel.WARNING:
            logger.warning(str(alert))
        elif alert.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            logger.error(str(alert))
        
        # Store in recent alerts
        self.recent_alerts.append(alert)
        if len(self.recent_alerts) > self.max_recent_alerts:
            self.recent_alerts.pop(0)
            
        # Attempt to dispatch to configured channels
        success = False
        
        # Email alerts (if enabled)
        if AlertConfig.EMAIL_ENABLED and self.email_config:
            email_success = self._send_email_alert(alert)
            success = success or email_success
        
        # Slack alerts (if enabled)
        if AlertConfig.SLACK_ENABLED and self.slack_config:
            slack_success = self._send_slack_alert(alert)
            success = success or slack_success
        
        # Syslog (if enabled)
        if AlertConfig.SYSLOG_ENABLED:
            syslog_success = self._send_syslog_alert(alert)
            success = success or syslog_success
        
        return success
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """
        Send an alert via email.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        # Placeholder for email notification
        # For production, this would connect to SMTP server and send the alert
        logger.debug(f"Would send email for alert: {alert.id}")
        return True
    
    def _send_slack_alert(self, alert: Alert) -> bool:
        """
        Send an alert via Slack.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        # Placeholder for Slack notification
        # For production, this would use Slack API to post the alert
        logger.debug(f"Would send Slack message for alert: {alert.id}")
        return True
    
    def _send_syslog_alert(self, alert: Alert) -> bool:
        """
        Send an alert to syslog.
        
        Args:
            alert: Alert to send
            
        Returns:
            True if sent successfully
        """
        # Placeholder for syslog integration
        # For production, this would log to system syslog
        logger.debug(f"Would send to syslog: {alert.id}")
        return True
    
    def configure_email(self, 
                       server: str, 
                       port: int, 
                       username: str, 
                       password: str,
                       sender: str,
                       recipients: List[str]) -> None:
        """
        Configure email notifications.
        
        Args:
            server: SMTP server hostname
            port: SMTP port
            username: SMTP username
            password: SMTP password
            sender: Sender email address
            recipients: List of recipient email addresses
        """
        self.email_config = {
            "server": server,
            "port": port,
            "username": username,
            "password": password,
            "sender": sender,
            "recipients": recipients
        }
    
    def configure_slack(self, webhook_url: str, channel: str) -> None:
        """
        Configure Slack notifications.
        
        Args:
            webhook_url: Slack webhook URL
            channel: Slack channel to post to
        """
        self.slack_config = {
            "webhook_url": webhook_url,
            "channel": channel
        }


class AlertManager:
    """
    Centralized manager for processing and dispatching alerts.
    
    This component monitors system resources, checks file sizes, and
    dispatches appropriate alerts based on configured thresholds.
    """
    
    def __init__(self):
        """Initialize the alert manager."""
        self.dispatcher = AlertDispatcher()
        self.resource_status = ResourceStatus()
        self.resource_monitor_active = False
        self.resource_monitor_thread = None
        self.resource_monitor_interval = 5  # seconds
    
    def check_file_size(self, file_path: str) -> Tuple[bool, Optional[Alert]]:
        """
        Check if a file exceeds size thresholds.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (is_safe, alert) where alert is None if file is safe
        """
        try:
            if not os.path.exists(file_path):
                alert = Alert(
                    AlertType.FILE_SIZE,
                    AlertLevel.WARNING,
                    f"File not found: {file_path}"
                )
                return (False, alert)
            
            # Get file size in GB
            file_size_bytes = os.path.getsize(file_path)
            file_size_gb = file_size_bytes / (1024**3)
            
            # Check against thresholds
            if file_size_gb > AlertConfig.MAX_FILE_SIZE:
                alert = Alert(
                    AlertType.FILE_SIZE,
                    AlertLevel.CRITICAL,
                    f"File exceeds maximum supported size: {file_size_gb:.2f}GB > {AlertConfig.MAX_FILE_SIZE}GB",
                    {"file_path": file_path, "file_size_gb": file_size_gb}
                )
                self.dispatcher.dispatch(alert)
                return (False, alert)
            
            if file_size_gb > AlertConfig.WARNING_FILE_SIZE:
                alert = Alert(
                    AlertType.FILE_SIZE,
                    AlertLevel.WARNING,
                    f"File is very large: {file_size_gb:.2f}GB. Processing may take significant time and resources.",
                    {"file_path": file_path, "file_size_gb": file_size_gb}
                )
                self.dispatcher.dispatch(alert)
                return (True, alert)
            
            if file_size_gb > AlertConfig.LARGE_FILE_THRESHOLD:
                alert = Alert(
                    AlertType.FILE_SIZE,
                    AlertLevel.INFO,
                    f"Processing large file: {file_size_gb:.2f}GB. Using chunked processing.",
                    {"file_path": file_path, "file_size_gb": file_size_gb}
                )
                self.dispatcher.dispatch(alert)
                return (True, alert)
            
            # File is within normal size range
            return (True, None)
            
        except Exception as e:
            logger.error(f"Error checking file size: {e}")
            alert = Alert(
                AlertType.FILE_SIZE,
                AlertLevel.WARNING,
                f"Error checking file size: {str(e)}",
                {"file_path": file_path, "error": str(e)}
            )
            return (False, alert)
    
    def check_system_resources(self) -> Tuple[bool, Optional[Alert]]:
        """
        Check if system resources are sufficient for processing.
        
        Returns:
            Tuple of (is_safe, alert) where alert is None if resources are safe
        """
        try:
            self.resource_status.update()
            
            # Check memory usage
            if self.resource_status.is_memory_critical():
                alert = Alert(
                    AlertType.MEMORY_USAGE,
                    AlertLevel.CRITICAL,
                    f"System memory usage critical: {self.resource_status.memory_percent:.1f}%",
                    self.resource_status.to_dict()
                )
                self.dispatcher.dispatch(alert)
                return (False, alert)
            
            if self.resource_status.is_memory_warning():
                alert = Alert(
                    AlertType.MEMORY_USAGE,
                    AlertLevel.WARNING,
                    f"System memory usage high: {self.resource_status.memory_percent:.1f}%",
                    self.resource_status.to_dict()
                )
                self.dispatcher.dispatch(alert)
                return (True, alert)
            
            # Check CPU usage
            if self.resource_status.is_cpu_critical():
                alert = Alert(
                    AlertType.CPU_USAGE,
                    AlertLevel.CRITICAL,
                    f"System CPU usage critical: {self.resource_status.cpu_percent:.1f}%",
                    self.resource_status.to_dict()
                )
                self.dispatcher.dispatch(alert)
                return (False, alert)
            
            if self.resource_status.is_cpu_warning():
                alert = Alert(
                    AlertType.CPU_USAGE,
                    AlertLevel.WARNING,
                    f"System CPU usage high: {self.resource_status.cpu_percent:.1f}%",
                    self.resource_status.to_dict()
                )
                self.dispatcher.dispatch(alert)
                return (True, alert)
            
            # Check disk space
            if self.resource_status.is_disk_critical():
                alert = Alert(
                    AlertType.DISK_SPACE,
                    AlertLevel.CRITICAL,
                    f"Disk space critically low: {self.resource_status.disk_free_percent:.1f}% free",
                    self.resource_status.to_dict()
                )
                self.dispatcher.dispatch(alert)
                return (False, alert)
            
            # All resources are within acceptable limits
            return (True, None)
            
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            alert = Alert(
                AlertType.MEMORY_USAGE,
                AlertLevel.WARNING,
                f"Error checking system resources: {str(e)}",
                {"error": str(e)}
            )
            return (False, alert)
    
    def estimate_resource_requirements(self, file_path: str) -> Dict[str, Any]:
        """
        Estimate resources needed to process a file.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            Dictionary with estimated resource requirements
        """
        if not os.path.exists(file_path):
            return {"error": "File not found"}
        
        try:
            # Get file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_gb = file_size_bytes / (1024**3)
            
            # I've found in practice that CSV parsing will typically need about
            # 1.3-1.5x the file size in RAM for our chunked processing approach.
            # This a rough estimate, actual results will vary based on parsing approach
            est_peak_memory_gb = min(file_size_gb * 1.5, file_size_gb * 0.2 + 2)
            
            # Estimate processing time (based on observed performance)
            # For larger files, I use more efficient chunked processing
            if file_size_gb > 10:
                est_processing_time_min = file_size_gb * 2.5
            else:
                est_processing_time_min = file_size_gb * 4
            
            # Calculate recommended chunk size
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            recommended_chunk_mb = int(min(available_memory_gb * 0.3, 512) * 1024)
            
            return {
                "file_size_gb": file_size_gb,
                "est_peak_memory_gb": est_peak_memory_gb,
                "est_processing_time_min": est_processing_time_min,
                "recommended_chunk_mb": recommended_chunk_mb
            }
            
        except Exception as e:
            logger.error(f"Error estimating resource requirements: {e}")
            return {"error": str(e)}
    
    def check_file_and_resources(self, file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check both file size and system resources.
        
        This is a convenience method that combines file size and resource checks.
        
        Args:
            file_path: Path to the file to check
            
        Returns:
            Tuple of (is_safe, message) where message explains any issues
        """
        # Check file size
        file_safe, file_alert = self.check_file_size(file_path)
        if not file_safe:
            return (False, str(file_alert))
        
        # Check system resources
        resources_safe, resource_alert = self.check_system_resources()
        if not resources_safe:
            return (False, str(resource_alert))
        
        # If we got file alert but it was safe, include that in the message
        if file_safe and file_alert:
            return (True, str(file_alert))
        
        # Everything is OK
        return (True, None)
    
    def start_resource_monitoring(self):
        """Start continuous resource monitoring in a background thread."""
        if self.resource_monitor_active:
            return
        
        self.resource_monitor_active = True
        self.resource_monitor_thread = threading.Thread(target=self._resource_monitor_loop)
        self.resource_monitor_thread.daemon = True
        self.resource_monitor_thread.start()
    
    def stop_resource_monitoring(self):
        """Stop the resource monitoring thread."""
        self.resource_monitor_active = False
        if self.resource_monitor_thread:
            self.resource_monitor_thread.join(timeout=1.0)
            self.resource_monitor_thread = None
    
    def _resource_monitor_loop(self):
        """Background thread function for continuous resource monitoring."""
        last_warning_time = datetime.min
        warning_cooldown = timedelta(minutes=5)  # Don't spam warnings
        
        while self.resource_monitor_active:
            try:
                self.resource_status.update()
                
                # Issue alerts if needed, respecting cooldown
                now = datetime.now()
                if now - last_warning_time > warning_cooldown:
                    issued_warning = False
                    
                    # Check critical thresholds
                    if self.resource_status.is_memory_critical():
                        alert = Alert(
                            AlertType.MEMORY_USAGE,
                            AlertLevel.CRITICAL,
                            f"Memory usage critical: {self.resource_status.memory_percent:.1f}%",
                            self.resource_status.to_dict()
                        )
                        self.dispatcher.dispatch(alert)
                        issued_warning = True
                    
                    elif self.resource_status.is_cpu_critical():
                        alert = Alert(
                            AlertType.CPU_USAGE,
                            AlertLevel.CRITICAL,
                            f"CPU usage critical: {self.resource_status.cpu_percent:.1f}%",
                            self.resource_status.to_dict()
                        )
                        self.dispatcher.dispatch(alert)
                        issued_warning = True
                    
                    elif self.resource_status.is_disk_critical():
                        alert = Alert(
                            AlertType.DISK_SPACE,
                            AlertLevel.CRITICAL,
                            f"Disk space critically low: {self.resource_status.disk_free_percent:.1f}% free",
                            self.resource_status.to_dict()
                        )
                        self.dispatcher.dispatch(alert)
                        issued_warning = True
                    
                    # Only check warning thresholds if no critical alerts were issued
                    elif self.resource_status.is_memory_warning():
                        alert = Alert(
                            AlertType.MEMORY_USAGE,
                            AlertLevel.WARNING,
                            f"Memory usage high: {self.resource_status.memory_percent:.1f}%",
                            self.resource_status.to_dict()
                        )
                        self.dispatcher.dispatch(alert)
                        issued_warning = True
                    
                    elif self.resource_status.is_cpu_warning():
                        alert = Alert(
                            AlertType.CPU_USAGE,
                            AlertLevel.WARNING,
                            f"CPU usage high: {self.resource_status.cpu_percent:.1f}%",
                            self.resource_status.to_dict()
                        )
                        self.dispatcher.dispatch(alert)
                        issued_warning = True
                    
                    if issued_warning:
                        last_warning_time = now
                
            except Exception as e:
                logger.error(f"Error in resource monitor: {e}")
            
            # Sleep before next check
            time.sleep(self.resource_monitor_interval)
    
    def recommend_execution_time(self) -> Tuple[bool, Optional[str]]:
        """
        Recommend if now is a good time to execute resource-intensive tasks.
        
        Returns:
            Tuple of (should_execute_now, recommendation) where recommendation
            explains why execution should be delayed if applicable
        """
        try:
            # Update resource status
            self.resource_status.update()
            
            # Check if system is currently under heavy load
            if self.resource_status.is_memory_critical() or self.resource_status.is_cpu_critical():
                return (False, "System resources are currently heavily utilized. Recommend delaying execution.")
            
            if self.resource_status.is_memory_warning() or self.resource_status.is_cpu_warning():
                return (False, "System resources are moderately utilized. Consider delaying execution if possible.")
            
            # Check time of day - in production we'd want to avoid peak trading hours
            hour = datetime.now().hour
            if 9 <= hour <= 16:  # 9 AM to 4 PM, typical trading hours
                return (False, "Current time falls within trading hours. Recommend scheduling execution for off-hours.")
            
            # All checks passed
            return (True, None)
            
        except Exception as e:
            logger.error(f"Error recommending execution time: {e}")
            return (False, f"Error evaluating system resources: {str(e)}")


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """
    Get the global alert manager instance.
    
    Returns:
        AlertManager instance
    """
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


def check_file_size(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to check if a file is within size limits.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        Tuple of (is_safe, message) where message explains any issues
    """
    manager = get_alert_manager()
    file_safe, alert = manager.check_file_size(file_path)
    return (file_safe, str(alert) if alert else None)


def check_system_resources() -> Tuple[bool, Optional[str]]:
    """
    Convenience function to check if system resources are sufficient.
    
    Returns:
        Tuple of (is_safe, message) where message explains any issues
    """
    manager = get_alert_manager()
    resources_safe, alert = manager.check_system_resources()
    return (resources_safe, str(alert) if alert else None)


def start_resource_monitoring():
    """Start continuous resource monitoring in a background thread."""
    manager = get_alert_manager()
    manager.start_resource_monitoring()


def stop_resource_monitoring():
    """Stop the resource monitoring thread."""
    manager = get_alert_manager()
    manager.stop_resource_monitoring()


def recommend_execution_time() -> Tuple[bool, Optional[str]]:
    """
    Recommend if now is a good time to execute resource-intensive tasks.
    
    Returns:
        Tuple of (should_execute_now, recommendation) where recommendation
        explains why execution should be delayed if applicable
    """
    manager = get_alert_manager()
    return manager.recommend_execution_time()


def schedule_execution(command: List[str], 
                     delay_minutes: Optional[int] = None,
                     wait_for_resources: bool = True) -> bool:
    """
    Schedule execution for a better time if system resources are constrained.
    
    This is a placeholder for a more sophisticated scheduling system. In a
    production environment, this would integrate with job scheduling systems.
    
    Args:
        command: Command to execute
        delay_minutes: Minutes to delay execution (or None for automatic)
        wait_for_resources: Whether to wait for resources to be available
        
    Returns:
        True if scheduling was successful
    """
    if not AlertConfig.ENABLE_SCHEDULED_EXECUTION:
        logger.warning("Scheduled execution is disabled in configuration")
        return False
    
    # This is a simplified placeholder for demonstration purposes
    # In production, this would use more sophisticated scheduling mechanisms
    
    if delay_minutes is None:
        # Auto-determine delay based on resource availability
        should_execute, reason = recommend_execution_time()
        if should_execute:
            delay_minutes = 0
        else:
            # Default delay if resources are constrained
            delay_minutes = 60
    
    if delay_minutes <= 0:
        # Execute immediately
        logger.info(f"Executing command immediately: {' '.join(command)}")
        
        try:
            subprocess.Popen(command)
            return True
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return False
    else:
        # Schedule for later execution
        logger.info(f"Scheduling command for execution in {delay_minutes} minutes: {' '.join(command)}")
        
        # In a real implementation, this would use at, cron, or a job scheduler
        # For this demonstration, we just log the intent
        execution_time = datetime.now() + timedelta(minutes=delay_minutes)
        
        logger.info(f"Command would be executed at {execution_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("NOTE: This is a placeholder. In production, this would use a proper scheduler.")
        
        return True


def estimate_resources(file_path: str) -> Dict[str, Any]:
    """
    Estimate resources needed to process a file.
    
    Args:
        file_path: Path to the file to process
        
    Returns:
        Dictionary with estimated resource requirements
    """
    manager = get_alert_manager()
    return manager.estimate_resource_requirements(file_path)


# Helper functions for testing the module directly
def _display_resource_estimates(file_path: str):
    """Display resource estimates for a file."""
    estimates = estimate_resources(file_path)
    
    if "error" in estimates:
        print(f"Error: {estimates['error']}")
        return
    
    print(f"Resource estimates for {file_path}:")
    print(f"  File size: {estimates['file_size_gb']:.2f} GB")
    print(f"  Peak memory usage: {estimates['est_peak_memory_gb']:.2f} GB")
    print(f"  Estimated processing time: {estimates['est_processing_time_min']:.1f} minutes")
    print(f"  Recommended chunk size: {estimates['recommended_chunk_mb']} MB")


if __name__ == "__main__":
    # Configure basic logging for direct execution
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="FIX Symbol Discrepancy Alert System")
    parser.add_argument("--check-file", type=str, help="Check if a file is within size limits")
    parser.add_argument("--estimate-resources", type=str, help="Estimate resources needed for processing")
    parser.add_argument("--check-resources", action="store_true", help="Check system resources")
    parser.add_argument("--monitor", action="store_true", help="Start resource monitoring")
    parser.add_argument("--monitor-duration", type=int, default=30, help="Duration for monitoring in seconds")
    parser.add_argument("--execution-time", action="store_true", help="Recommend execution time")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    
    args = parser.parse_args()
    
    if args.check_file:
        file_safe, message = check_file_size(args.check_file)
        if file_safe:
            if message:
                print(f"File {args.check_file} is within limits, but note: {message}")
            else:
                print(f"File {args.check_file} is within acceptable size limits")
        else:
            print(f"File size warning: {message}")
    
    elif args.estimate_resources:
        _display_resource_estimates(args.estimate_resources)
    
    elif args.check_resources:
        resources_safe, message = check_system_resources()
        if resources_safe:
            print("System resources are within acceptable limits")
        else:
            print(f"Resource warning: {message}")
    
    elif args.monitor:
        print(f"Starting resource monitoring for {args.monitor_duration} seconds...")
        start_resource_monitoring()
        try:
            time.sleep(args.monitor_duration)
        except KeyboardInterrupt:
            print("Monitoring interrupted by user")
        finally:
            stop_resource_monitoring()
        print("Resource monitoring stopped")
    
    elif args.execution_time:
        should_execute, reason = recommend_execution_time()
        if should_execute:
            print("System resources and time are optimal for execution")
        else:
            print(f"Recommendation: {reason}")
    
    elif args.config:
        # This would load configuration settings from a file in a production version
        print(f"Configuration would be loaded from {args.config}")
        print("Note: Configuration loading is a placeholder in this demo version")
    
    else:
        parser.print_help()