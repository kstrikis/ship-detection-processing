#!/usr/bin/env python3
"""
Script to fix imports in the improved_pipeline module.
This script changes absolute imports to relative imports.
"""

import os
import re
import glob
from pathlib import Path

def fix_imports_in_file(file_path):
    """Fix imports in a single file."""
    print(f"Processing {file_path}")
    
    with open(file_path, 'r') as f:
        original_content = f.read()
    
    # Replace direct imports from other module files
    patterns = [
        (r'from utils import', 'from .utils import'),
        (r'from preprocessor import', 'from .preprocessor import'),
        (r'from ship_detector import', 'from .ship_detector import'),
        (r'from manual_selection import', 'from .manual_selection import'),
        (r'from phase_extractor import', 'from .phase_extractor import'),
        (r'from time_frequency_analyzer import', 'from .time_frequency_analyzer import'),
        (r'from component_classifier import', 'from .component_classifier import'),
        (r'from physics_constraints import', 'from .physics_constraints import'),
        (r'from visualizer import', 'from .visualizer import'),
    ]
    
    modified_content = original_content
    for pattern, replacement in patterns:
        modified_content = re.sub(pattern, replacement, modified_content)
    
    if original_content != modified_content:
        with open(file_path, 'w') as f:
            f.write(modified_content)
        return True
    
    return False

def main():
    """Fix imports in all Python files in the improved_pipeline module."""
    pipeline_dir = Path('src/improved_pipeline')
    py_files = glob.glob(str(pipeline_dir / '*.py'))
    
    files_changed = 0
    for py_file in py_files:
        if fix_imports_in_file(py_file):
            files_changed += 1
    
    print(f"\nDone! Fixed imports in {files_changed} files.")

if __name__ == "__main__":
    main() 