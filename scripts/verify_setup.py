#!/usr/bin/env python3
"""
Quick test to verify setup is correct
"""

import sys
from pathlib import Path

def test_imports():
    """Test that modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.tep.data_structures import ActionType, FrameData
        print("  ✓ data_structures imports successfully")
    except ImportError as e:
        print(f"  ✗ Failed to import data_structures: {e}")
        return False
    
    # Add more import tests as modules are implemented
    
    return True

def test_directory_structure():
    """Test that all directories exist"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        "src/procedure",
        "src/tep",
        "src/integration",
        "templates",
        "tests",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ Missing: {dir_path}/")
            all_exist = False
    
    return all_exist

if __name__ == "__main__":
    print("=== Setup Verification ===\n")
    
    structure_ok = test_directory_structure()
    imports_ok = test_imports()
    
    if structure_ok and imports_ok:
        print("\n✓ Setup verified successfully!")
        print("\nReady to implement modules from IMPLEMENTATION_GUIDE.md")
        sys.exit(0)
    else:
        print("\n✗ Setup verification failed")
        print("\nPlease fix the issues above and try again")
        sys.exit(1)
