"""
Virtual environment setup utility.

This module handles automatic virtual environment creation and requirements installation.
It ensures the script runs in a proper virtual environment with all dependencies installed.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path


def setup_venv():
    """
    Create virtual environment if it doesn't exist and install requirements.
    
    This function:
    1. Checks if we're already in a virtual environment
    2. If not, creates a venv in the project root
    3. Installs requirements.txt
    4. Re-executes the script with the venv Python
    
    If already in venv, it checks if key packages are installed and installs
    them if missing.
    """
    project_root = Path(__file__).parent.parent.parent
    venv_path = project_root / ".venv"
    requirements_file = project_root / "requirements.txt"
    
    # Check if we're already in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    # If not in venv, create one and re-execute
    if not in_venv:
        print("=" * 80)
        print("Setting up virtual environment...")
        print("=" * 80)
        
        # Create venv if it doesn't exist
        if not venv_path.exists():
            print(f"Creating virtual environment at {venv_path}...")
            venv.create(venv_path, with_pip=True)
            print("✓ Virtual environment created")
        else:
            print(f"✓ Virtual environment already exists at {venv_path}")
        
        # Determine Python and pip executables in venv
        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
            venv_pip = venv_path / "Scripts" / "pip.exe"
        else:
            venv_python = venv_path / "bin" / "python"
            venv_pip = venv_path / "bin" / "pip"
        
        # Verify venv Python exists
        if not venv_python.exists():
            print(f"⚠ Error: Virtual environment Python not found at {venv_python}")
            print("Please recreate the virtual environment manually.")
            sys.exit(1)
        
        # Install requirements if requirements.txt exists
        # This is done using the venv's pip (which is effectively "activating" the venv)
        if requirements_file.exists():
            print(f"Installing requirements from {requirements_file}...")
            print(f"Using venv Python: {venv_python}")
            try:
                # Upgrade pip first using venv's Python
                print("Upgrading pip in virtual environment...")
                subprocess.check_call([
                    str(venv_python), "-m", "pip", "install", "--upgrade", "pip", "--quiet"
                ])
                
                # Install requirements using venv's pip
                print("Installing requirements in virtual environment...")
                subprocess.check_call([
                    str(venv_python), "-m", "pip", "install", "-r", str(requirements_file)
                ])
                print("✓ Requirements installed in virtual environment")
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: Failed to install some requirements: {e}")
                print("Continuing anyway...")
        else:
            print(f"⚠ Warning: {requirements_file} not found, skipping requirements installation")
        
        # Re-execute script with venv Python (this activates the venv for the script)
        print(f"\nActivating virtual environment and re-executing script...")
        print(f"Using: {venv_python}")
        print("=" * 80)
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)
    
    # If we're in venv, ensure requirements are installed
    elif in_venv and requirements_file.exists():
        # Check if key packages are installed by trying to import them
        try:
            import importlib
            importlib.import_module('tensorflow')
            importlib.import_module('sionna')
        except ImportError:
            print("=" * 80)
            print("Virtual environment is active. Installing missing requirements...")
            print(f"Using venv Python: {sys.executable}")
            print("=" * 80)
            try:
                # Upgrade pip in the active venv
                print("Upgrading pip in virtual environment...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "--upgrade", "pip", "--quiet"
                ])
                
                # Install requirements in the active venv
                print("Installing requirements in virtual environment...")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
                ])
                print("✓ Requirements installed in virtual environment")
                print("Re-executing script to load new packages...")
                print("=" * 80)
                # Re-execute to ensure new packages are loaded
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except subprocess.CalledProcessError as e:
                print(f"⚠ Warning: Failed to install requirements: {e}")
                print("Continuing anyway...")
                print("=" * 80)

