#!/usr/bin/env python3
"""
Setup script for Safe Autonomous Vehicle Control project.

This script helps set up the development environment by:
1. Checking Python version
2. Creating virtual environment (optional)
3. Installing dependencies
4. Running environment tests
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: Python {version.major}.{version.minor}.{version.micro}")
        sys.exit(1)
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{description}...")
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False


def install_dependencies(upgrade=False):
    """Install Python dependencies from requirements.txt."""
    cmd = "pip install -r requirements.txt"
    if upgrade:
        cmd += " --upgrade"
    return run_command(cmd, "Installing dependencies")


def run_tests():
    """Run environment tests."""
    return run_command(
        "python tests/test_environment.py --test-only",
        "Running environment tests"
    )


def main():
    """Main setup script."""
    parser = argparse.ArgumentParser(description="Setup Safe Autonomous Control project")
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip dependency installation"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests"
    )
    parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Upgrade dependencies to latest versions"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Safe Autonomous Vehicle Control - Setup")
    print("=" * 80)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not args.skip_install:
        success = install_dependencies(upgrade=args.upgrade)
        if not success:
            print("\n❌ Setup failed during dependency installation")
            sys.exit(1)
    
    # Run tests
    if not args.skip_tests:
        success = run_tests()
        if not success:
            print("\nTests failed, but setup is complete")
            
        else:
            
            print("✅ Setup completed successfully!")
            
    else:
        print("Setup completed (tests skipped)")
        


if __name__ == "__main__":
    main()
