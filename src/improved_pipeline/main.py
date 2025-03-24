#!/usr/bin/env python3
"""
Main entrypoint for the improved ship micro-motion analysis pipeline.
This script coordinates the execution of all pipeline stages and allows
for running individual stages in isolation.

Usage:
    python main.py --input <input_file> --output-dir <output_dir> [options]
    python main.py --run-stage <stage_name> --input <input_file> --output <output_file> [options]

For help and options:
    python main.py --help
"""

import os
import sys
import logging
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

# Import pipeline stages
from .preprocessor import preprocess_sar_data
from .ship_detector import detect_ships
from .manual_selection import manually_select_ships
from .phase_extractor import extract_phase_history
from .time_frequency_analyzer import analyze_time_frequency
from .component_classifier import classify_components
from .physics_constraints import apply_physics_constraints
from .visualizer import create_visualizations

# Import utilities
from .utils import setup_logging, save_results, load_step_output


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Improved Ship Micro-Motion Analysis Pipeline',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Common arguments
    parser.add_argument('--input', type=str, help='Input SAR data file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--log-file', type=str,
                        help='Log file path (default: auto-generated)')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level')
    
    # Pipeline execution control
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--run-all', action='store_true',
                     help='Run all pipeline stages (default behavior)')
    group.add_argument('--run-stage', type=str,
                     choices=[
                         'preprocess', 'detect', 'manual-select', 'extract-phase',
                         'analyze-frequency', 'classify-components', 
                         'apply-physics', 'visualize'
                     ],
                     help='Run a specific pipeline stage')
    
    # Stage-specific input/output files
    parser.add_argument('--stage-input', type=str,
                      help='Input file for specific stage (when using --run-stage)')
    parser.add_argument('--stage-output', type=str,
                      help='Output file for specific stage (when using --run-stage)')
    
    # Feature flags and options
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--skip-constraints', action='store_true',
                      help='Skip physical constraint application')
    parser.add_argument('--crop-only', action='store_true',
                      help='Only perform data cropping')
    parser.add_argument('--num-subapertures', type=int, default=200,
                      help='Number of subapertures to generate')
    parser.add_argument('--speckle-filter-size', type=int, default=5,
                      help='Kernel size for speckle filtering (0 to disable)')
    
    return parser.parse_args()


def run_pipeline(args):
    """Run the complete pipeline or a specific stage."""
    # Setup logging
    if not args.log_file:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(args.output_dir, f"pipeline_{timestamp}.log")
    
    logger = setup_logging(args.log_file, args.log_level)
    logger.info(f"Starting improved micro-motion pipeline. Args: {args}")
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run specific stage if requested
    if args.run_stage:
        run_single_stage(args, logger)
        return
    
    # Otherwise run full pipeline
    logger.info("Running complete pipeline")
    
    # Stage 1: Preprocess SAR data
    preprocess_output = os.path.join(args.output_dir, "01_preprocessed.npz")
    preprocess_result = preprocess_sar_data(
        input_file=args.input,
        output_file=preprocess_output,
        speckle_filter_size=args.speckle_filter_size,
        use_gpu=args.use_gpu,
        crop_only=args.crop_only,
        logger=logger
    )
    
    # Exit if only cropping was requested
    if args.crop_only:
        logger.info("Crop-only mode: pipeline execution stopped after preprocessing")
        return
    
    # Stage 2: Detect ships
    detect_output = os.path.join(args.output_dir, "02_detected_ships.npz")
    detect_result = detect_ships(
        input_file=preprocess_output,
        output_file=detect_output,
        use_gpu=args.use_gpu,
        logger=logger
    )
    
    # Stage 3: Extract phase history
    phase_output = os.path.join(args.output_dir, "03_phase_history.npz")
    phase_result = extract_phase_history(
        input_file=preprocess_output,
        ship_file=detect_output,
        output_file=phase_output,
        num_subapertures=args.num_subapertures,
        use_gpu=args.use_gpu,
        logger=logger
    )
    
    # Stage 4: Analyze time-frequency
    tf_output = os.path.join(args.output_dir, "04_time_frequency.npz")
    tf_result = analyze_time_frequency(
        input_file=phase_output,
        output_file=tf_output,
        use_gpu=args.use_gpu,
        logger=logger
    )
    
    # Stage 5: Classify ship components
    comp_output = os.path.join(args.output_dir, "05_components.npz")
    comp_result = classify_components(
        input_file=phase_output,
        ship_file=detect_output,
        output_file=comp_output,
        use_gpu=args.use_gpu,
        logger=logger
    )
    
    # Stage 6: Apply physics constraints
    if not args.skip_constraints:
        physics_output = os.path.join(args.output_dir, "06_physics_constrained.npz")
        physics_result = apply_physics_constraints(
            input_file=tf_output,
            component_file=comp_output,
            output_file=physics_output,
            logger=logger
        )
        vibration_results = physics_output
    else:
        logger.info("Skipping physical constraints as requested")
        vibration_results = tf_output
    
    # Stage 7: Create visualizations
    vis_output = os.path.join(args.output_dir, "07_visualizations")
    os.makedirs(vis_output, exist_ok=True)
    vis_result = create_visualizations(
        preprocessed_file=preprocess_output,
        ship_file=detect_output,
        component_file=comp_output,
        vibration_file=vibration_results,
        output_dir=vis_output,
        logger=logger
    )
    
    logger.info("Pipeline execution completed successfully")


def run_single_stage(args, logger):
    """Run a single pipeline stage."""
    if not args.stage_input and args.run_stage != 'preprocess':
        logger.error(f"--stage-input is required when running stage {args.run_stage}")
        sys.exit(1)
    
    if not args.stage_output:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.stage_output = os.path.join(
            args.output_dir, 
            f"{args.run_stage}_{timestamp}.npz"
        )
    
    logger.info(f"Running single stage: {args.run_stage}")
    os.makedirs(os.path.dirname(args.stage_output), exist_ok=True)
    
    if args.run_stage == 'preprocess':
        input_file = args.input or args.stage_input
        preprocess_sar_data(
            input_file=input_file,
            output_file=args.stage_output,
            speckle_filter_size=args.speckle_filter_size,
            use_gpu=args.use_gpu,
            crop_only=args.crop_only,
            logger=logger
        )
    
    elif args.run_stage == 'detect':
        detect_ships(
            input_file=args.stage_input,
            output_file=args.stage_output,
            use_gpu=args.use_gpu,
            logger=logger
        )
    
    elif args.run_stage == 'manual-select':
        manually_select_ships(
            input_file=args.stage_input,
            output_file=args.stage_output,
            logger=logger
        )
    
    elif args.run_stage == 'extract-phase':
        # For this stage, we need both preprocessed data and ship locations
        if not args.ship_file:
            logger.error("--ship-file is required for extract-phase stage")
            sys.exit(1)
        
        extract_phase_history(
            input_file=args.stage_input,
            ship_file=args.ship_file,
            output_file=args.stage_output,
            num_subapertures=args.num_subapertures,
            use_gpu=args.use_gpu,
            logger=logger
        )
    
    elif args.run_stage == 'analyze-frequency':
        analyze_time_frequency(
            input_file=args.stage_input,
            output_file=args.stage_output,
            use_gpu=args.use_gpu,
            logger=logger
        )
    
    elif args.run_stage == 'classify-components':
        # Need both phase history and ship detection
        if not args.ship_file:
            logger.error("--ship-file is required for classify-components stage")
            sys.exit(1)
            
        classify_components(
            input_file=args.stage_input,
            ship_file=args.ship_file,
            output_file=args.stage_output,
            use_gpu=args.use_gpu,
            logger=logger
        )
    
    elif args.run_stage == 'apply-physics':
        # Need both time-frequency analysis and component classification
        if not args.component_file:
            logger.error("--component-file is required for apply-physics stage")
            sys.exit(1)
            
        apply_physics_constraints(
            input_file=args.stage_input,
            component_file=args.component_file,
            output_file=args.stage_output,
            logger=logger
        )
    
    elif args.run_stage == 'visualize':
        # Need all previous results
        if not all([args.preprocessed_file, args.ship_file, 
                    args.component_file, args.vibration_file]):
            logger.error("All input files are required for visualization stage")
            sys.exit(1)
            
        os.makedirs(args.stage_output, exist_ok=True)
        create_visualizations(
            preprocessed_file=args.preprocessed_file,
            ship_file=args.ship_file,
            component_file=args.component_file,
            vibration_file=args.vibration_file,
            output_dir=args.stage_output,
            logger=logger
        )
    
    logger.info(f"Stage {args.run_stage} completed successfully")


if __name__ == "__main__":
    args = parse_arguments()
    
    # Add additional arguments to parser for stage-specific needs
    if args.run_stage == 'extract-phase':
        parser = argparse.ArgumentParser()
        parser.add_argument('--ship-file', type=str, required=True,
                          help='Ship detection result file')
        extra_args, _ = parser.parse_known_args()
        args.ship_file = extra_args.ship_file
    
    elif args.run_stage == 'classify-components':
        parser = argparse.ArgumentParser()
        parser.add_argument('--ship-file', type=str, required=True,
                          help='Ship detection result file')
        extra_args, _ = parser.parse_known_args()
        args.ship_file = extra_args.ship_file
    
    elif args.run_stage == 'apply-physics':
        parser = argparse.ArgumentParser()
        parser.add_argument('--component-file', type=str, required=True,
                          help='Component classification result file')
        extra_args, _ = parser.parse_known_args()
        args.component_file = extra_args.component_file
    
    elif args.run_stage == 'visualize':
        parser = argparse.ArgumentParser()
        parser.add_argument('--preprocessed-file', type=str, required=True,
                          help='Preprocessed data file')
        parser.add_argument('--ship-file', type=str, required=True,
                          help='Ship detection result file')
        parser.add_argument('--component-file', type=str, required=True,
                          help='Component classification result file')
        parser.add_argument('--vibration-file', type=str, required=True,
                          help='Vibration analysis result file')
        extra_args, _ = parser.parse_known_args()
        args.preprocessed_file = extra_args.preprocessed_file
        args.ship_file = extra_args.ship_file
        args.component_file = extra_args.component_file
        args.vibration_file = extra_args.vibration_file
    
    run_pipeline(args) 