#!/usr/bin/env python3
"""Setup script for Visual Reasoning Systems."""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def main():
    """Setup the Visual Reasoning Systems project."""
    print("Visual Reasoning Systems - Setup Script")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("✗ Python 3.10+ is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directories
    directories = ["data", "checkpoints", "logs", "assets", "assets/evaluation"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Install dependencies
    print("\nInstalling dependencies...")
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        print("⚠ Some packages may not have installed correctly")
    
    # Install pre-commit hooks (optional)
    print("\nSetting up pre-commit hooks...")
    if run_command("pip install pre-commit", "Installing pre-commit"):
        if run_command("pre-commit install", "Installing pre-commit hooks"):
            print("✓ Pre-commit hooks installed")
        else:
            print("⚠ Pre-commit hooks installation failed")
    else:
        print("⚠ Pre-commit installation failed")
    
    # Test basic functionality
    print("\nTesting basic functionality...")
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} available")
        
        import transformers
        print(f"✓ Transformers {transformers.__version__} available")
        
        # Test device detection
        from src.utils.device import get_device
        device = get_device("auto")
        print(f"✓ Device detection working: {device}")
        
    except ImportError as e:
        print(f"⚠ Import test failed: {e}")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("\nNext steps:")
    print("1. Test the framework: python 0568.py")
    print("2. Run training: python scripts/train.py")
    print("3. Launch demo: streamlit run demo/streamlit_demo.py")
    print("4. Read documentation: README.md")


if __name__ == "__main__":
    main()
