#!/usr/bin/env python3
"""
Setup script to create directory structure for TEP and Procedure Generator
Run this first before implementing the modules
"""

import os
from pathlib import Path


def create_directory_structure():
    """Create all required directories and __init__.py files"""
    
    base_dirs = [
        "src",
        "src/procedure",
        "src/tep",
        "src/integration",
        "templates",
        "tests",
        "tests/fixtures",
        "tests/fixtures/sample_procedures",
        "tests/fixtures/sample_videos",
        "scripts",
    ]
    
    print("Creating directory structure...")
    for dir_path in base_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created {dir_path}/")
    
    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/procedure/__init__.py",
        "src/tep/__init__.py",
        "src/integration/__init__.py",
        "tests/__init__.py",
    ]
    
    print("\nCreating __init__.py files...")
    for init_file in init_files:
        Path(init_file).touch()
        print(f"  ✓ Created {init_file}")
    
    # Create placeholder files with docstrings
    module_files = {
        "src/tep/data_structures.py": '"""\nShared data structures for TEP system\n"""\n\n# TODO: Implement data structures from IMPLEMENTATION_GUIDE.md\n',
        "src/procedure/template_generator.py": '"""\nProcedure Template Generator - Main orchestrator\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/procedure/text_preprocessor.py": '"""\nStage 1: Text Preprocessing\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/procedure/parameter_extractor.py": '"""\nStage 2: Parameter Extraction\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/procedure/action_classifier.py": '"""\nStage 3: Action Classification\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/tep/temporal_event_parser.py": '"""\nTemporal Event Parser - Main orchestrator\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/tep/window_manager.py": '"""\nTemporal Window Manager\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/tep/action_classifier.py": '"""\nRule-based Action Classifier\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/integration/frame_converter.py": '"""\nConvert YOLO/SAM/CLIP output to FrameData\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "src/integration/procedure_executor.py": '"""\nRuntime procedure execution orchestration\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "scripts/test_video_with_tep.py": '"""\nTest TEP on video with procedure template\n"""\n\n# TODO: Implement from IMPLEMENTATION_GUIDE.md\n',
        "templates/template_schema.json": '{\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "title": "Procedure Template",\n  "description": "Schema for laboratory procedure templates"\n}\n',
    }
    
    print("\nCreating module placeholder files...")
    for file_path, content in module_files.items():
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"  ✓ Created {file_path}")
    
    # Create README for templates
    readme_content = """# Procedure Templates

This directory contains JSON procedure templates.

## Creating a Template

1. Write step descriptions in a text file (one per line)
2. Run: `python scripts/create_template.py your_steps.txt`
3. Template will be saved as JSON in this directory

## Template Structure

See `template_schema.json` for the complete schema.

## Example

See `dna_extraction.json` for a complete example.
"""
    
    with open("templates/README.md", 'w') as f:
        f.write(readme_content)
    
    print("\n✓ Directory structure created successfully!")
    print("\nNext steps:")
    print("1. Read IMPLEMENTATION_GUIDE.md")
    print("2. Implement modules following the guide")
    print("3. Run: python scripts/test_video_with_tep.py")


def create_gitignore_entries():
    """Add entries to .gitignore"""
    gitignore_entries = [
        "\n# TEP and Procedure Generator",
        "templates/*.json",
        "!templates/template_schema.json",
        "tests/__pycache__/",
        "src/__pycache__/",
        "src/*/__pycache__/",
        "output/execution_log.json",
    ]
    
    gitignore_path = Path(".gitignore")
    
    if gitignore_path.exists():
        with open(gitignore_path, 'r') as f:
            existing = f.read()
        
        # Check if already added
        if "TEP and Procedure Generator" not in existing:
            with open(gitignore_path, 'a') as f:
                f.write('\n'.join(gitignore_entries))
            print("\n✓ Updated .gitignore")
    else:
        with open(gitignore_path, 'w') as f:
            f.write('\n'.join(gitignore_entries))
        print("\n✓ Created .gitignore")


def check_existing_files():
    """Check if required existing files are present"""
    required_files = [
        "process_video.py",
        "adaptive_lab_system.py",
        "requirements.txt",
        "videos/DNA_Extraction.txt",
    ]
    
    print("\nChecking existing files...")
    all_present = True
    
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"  ✓ Found {file_path}")
        else:
            print(f"  ✗ Missing {file_path}")
            all_present = False
    
    if not all_present:
        print("\n⚠ Warning: Some required files are missing")
        print("  Make sure you're running this from the project root")
    
    return all_present


def create_quick_test():
    """Create a quick test script to verify setup"""
    test_content = '''#!/usr/bin/env python3
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
    print("\\nChecking directory structure...")
    
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
    print("=== Setup Verification ===\\n")
    
    structure_ok = test_directory_structure()
    imports_ok = test_imports()
    
    if structure_ok and imports_ok:
        print("\\n✓ Setup verified successfully!")
        print("\\nReady to implement modules from IMPLEMENTATION_GUIDE.md")
        sys.exit(0)
    else:
        print("\\n✗ Setup verification failed")
        print("\\nPlease fix the issues above and try again")
        sys.exit(1)
'''
    
    with open("scripts/verify_setup.py", 'w') as f:
        f.write(test_content)
    
    os.chmod("scripts/verify_setup.py", 0o755)
    print("\n✓ Created scripts/verify_setup.py")


if __name__ == "__main__":
    print("=" * 60)
    print("TEP and Procedure Generator Setup")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not Path("adaptive_lab_system.py").exists():
        print("⚠ Warning: adaptive_lab_system.py not found")
        print("Please run this script from the project root directory")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(1)
    
    # Create structure
    create_directory_structure()
    
    # Update gitignore
    create_gitignore_entries()
    
    # Check existing files
    check_existing_files()
    
    # Create verification script
    create_quick_test()
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Read: IMPLEMENTATION_GUIDE.md")
    print("2. Verify: python scripts/verify_setup.py")
    print("3. Implement modules following the guide")
    print("4. Test: python scripts/test_video_with_tep.py")
