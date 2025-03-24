#!/usr/bin/env python3
"""
Wrapper script to run the improved ship detection pipeline.

Usage:
    python run_improved_pipeline.py <input_file> [options]
"""

import sys
import argparse
from src.improved_pipeline.main import parse_arguments, run_pipeline

if __name__ == "__main__":
    # If only an input file is provided as positional argument, convert it to --input format
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-'):
        input_file = sys.argv[1]
        sys.argv[1:2] = ['--input', input_file]
    
    # Parse arguments using the main module's parser
    args = parse_arguments()
    
    # Run the pipeline with the provided arguments
    run_pipeline(args) 