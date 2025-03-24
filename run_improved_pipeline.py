#!/usr/bin/env python3
"""
Wrapper script for the improved ship micro-motion analysis pipeline.

Usage:
    python run_improved_pipeline.py --input /path/to/sar_data.cphd --output-dir results [options]
    python run_improved_pipeline.py --run-stage STAGE --input /path/to/input.npz --output /path/to/output.npz [options]
"""

import sys
import argparse
from src.improved_pipeline import main

if __name__ == "__main__":
    # Forward all arguments to the main module
    args = main.parse_arguments()
    main.run_pipeline(args) 