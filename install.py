#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIX Symbol Discrepancy Checker - Installation Script

This script ensures all required dependencies are installed and up to date
before running the FIX reconciliation tool. It checks for required packages,
updates them if needed, and installs any missing dependencies.

Usage:
    python install.py [--dev] [--quiet]

Author: Carlyle
Date: March 15, 2025
"""

import os
import sys
import subprocess
import argparse
import platform
from typing import List, Dict, Tuple, Optional


# Required packages for the application
REQUIRED_PACKAGES = [
    "pandas",       # For data manipulation and CSV handling
    "numpy",        # Dependency for pandas
    "pytz",         # Timezone support for timestamps
    "python-dateutil",  # Date parsing for FIX messages
]

# Additional packages for development/testing
DEV_PACKAGES = [
    "pytest",       # For unit testing
    "pytest-cov",   # For test coverage reports
    "black",        # Code formatter
    "flake8",       # Linter
    "mypy",         # Type checking
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Install dependencies for FIX Symbol Discrepancy Checker")
    parser.add_argument("--dev", action="store_true", help="Install development dependencies")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    return parser.parse_args()


def check_python_version() -> bool:
    """
    Check if the Python version meets requirements (3.8+).
    
    Returns:
        bool: True if Python version is sufficient, False otherwise
    """
    major, minor, _ = platform.python_version_tuple()
    
    if int(major) < 3 or (int(major) == 3 and int(minor) < 8):
        print(f"Error: Python 3.8+ is required. Found: {platform.python_version()}")
        return False
    
    return True


def run_command(command: List[str], quiet: bool = False) -> Tuple[int, str, str]:
    """
    Run a shell command and return its output.
    
    Args:
        command: List containing the command and its arguments
        quiet: If True, suppress command output
        
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    if not quiet:
        print(f"Running: {' '.join(command)}")
    
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True
    )
    
    stdout, stderr = process.communicate()
    return_code = process.returncode
    
    return return_code, stdout, stderr


def get_installed_packages() -> Dict[str, str]:
    """
    Get a dictionary of installed packages and their versions.
    
    Returns:
        Dictionary mapping package names to version strings
    """
    return_code, stdout, stderr = run_command([sys.executable, "-m", "pip", "list", "--format=json"], quiet=True)
    
    if return_code != 0:
        print(f"Warning: Could not get installed packages list: {stderr}")
        return {}
    
    try:
        import json
        packages = json.loads(stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in packages}
    except Exception as e:
        print(f"Warning: Error parsing pip list output: {e}")
        return {}


def update_pip(quiet: bool = False) -> bool:
    """
    Update pip itself to the latest version.
    
    Args:
        quiet: If True, suppress command output
        
    Returns:
        bool: True if pip was updated successfully, False otherwise
    """
    print("Updating pip to latest version...")
    
    command = [sys.executable, "-m", "pip", "install", "--upgrade", "pip"]
    return_code, stdout, stderr = run_command(command, quiet)
    
    if return_code != 0:
        print(f"Warning: Failed to update pip: {stderr}")
        return False
    
    if not quiet:
        print("Pip updated successfully.")
    
    return True


def install_package(package: str, quiet: bool = False) -> bool:
    """
    Install a single package using pip.
    
    Args:
        package: Name of the package to install
        quiet: If True, suppress command output
        
    Returns:
        bool: True if installation was successful, False otherwise
    """
    print(f"Installing {package}...")
    
    command = [sys.executable, "-m", "pip", "install", package]
    return_code, stdout, stderr = run_command(command, quiet)
    
    if return_code != 0:
        print(f"Error: Failed to install {package}: {stderr}")
        return False
    
    if not quiet:
        print(f"{package} installed successfully.")
    
    return True


def ensure_requirements(dev: bool = False, quiet: bool = False) -> bool:
    """
    Ensure all required packages are installed.
    
    Args:
        dev: If True, install development dependencies as well
        quiet: If True, suppress command output
        
    Returns:
        bool: True if all packages were installed successfully, False otherwise
    """
    packages_to_install = REQUIRED_PACKAGES.copy()
    if dev:
        packages_to_install.extend(DEV_PACKAGES)
    
    installed_packages = get_installed_packages()
    missing_packages = [pkg for pkg in packages_to_install if pkg.lower() not in installed_packages]
    
    if not missing_packages:
        print("All required packages are already installed.")
        return True
    
    print(f"Installing {len(missing_packages)} missing packages...")
    
    success = True
    for package in missing_packages:
        if not install_package(package, quiet):
            success = False
    
    return success


def create_requirements_file() -> None:
    """Create requirements.txt file for the project."""
    with open("requirements.txt", "w") as f:
        for package in REQUIRED_PACKAGES:
            f.write(f"{package}\n")
    
    print("Created requirements.txt")
    
    if os.path.exists("requirements-dev.txt"):
        print("requirements-dev.txt already exists, skipping creation")
        return
        
    with open("requirements-dev.txt", "w") as f:
        f.write("-r requirements.txt\n")
        for package in DEV_PACKAGES:
            f.write(f"{package}\n")
    
    print("Created requirements-dev.txt")


def main() -> int:
    """
    Main entry point for the installation script.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    args = parse_args()
    
    print("FIX Symbol Discrepancy Checker - Dependency Installer")
    print("-" * 50)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Update pip first
    if not update_pip(args.quiet):
        print("Warning: Failed to update pip. Continuing with installation...")
    
    # Install required packages
    if not ensure_requirements(args.dev, args.quiet):
        print("Error: Failed to install all required packages.")
        return 1
    
    # Create requirements files for documentation and future installations
    create_requirements_file()
    
    print("-" * 50)
    print("Installation completed successfully.")
    print("You can now run the application using: python main.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())