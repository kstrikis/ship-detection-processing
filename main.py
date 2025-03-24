"""
Ship detection and micro-motion analysis from SAR imagery.
Enhanced with pixel-level phase history analysis and component classification.
"""

import os
import sys
import argparse
import logging
import numpy as np
from datetime import datetime

from src.ship_detection.processor import EnhancedShipDetectionProcessor
from src.ship_detection.utils.helpers import setup_logging

def print_separator():
    """Print a separator line to the console."""
    print("\n" + "="*80 + "\n")

def monitor_memory_usage():
    """Check current memory usage and return as string."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
        return f"{memory_gb:.2f} GB"
    except ImportError:
        return "psutil not installed"

def main():
    """
    Main entry point for the ship detection and micro-motion analysis application.
    Implements advanced component-specific vibration analysis.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Ship detection and micro-motion analysis from SAR imagery with component classification",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input_file", 
        help="Path to input SAR data file (CPHD, SICD, or NITF format)"
    )
    
    # Basic options
    basic_group = parser.add_argument_group('Basic options')
    basic_group.add_argument(
        "-o", 
        "--output-dir", 
        default="results",
        help="Directory to save results"
    )
    basic_group.add_argument(
        "-l", 
        "--log-file", 
        help="Path to log file (defaults to timestamp-based file in output directory)"
    )
    basic_group.add_argument(
        "-v", 
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    basic_group.add_argument(
        "--manual", 
        action="store_true",
        help="Use manual ship selection instead of automatic detection"
    )
    basic_group.add_argument(
        "--crop-only", 
        action="store_true",
        help="Only perform cropping on the input file, skip detection and analysis"
    )
    
    # Memory and performance options
    memory_group = parser.add_argument_group('Memory and performance options')
    memory_group.add_argument(
        "--preview-factor", 
        type=int, 
        default=4,
        help="Downsampling factor for preview image used in ship detection (higher = less memory)"
    )
    memory_group.add_argument(
        "--memory-tracking", 
        action="store_true",
        help="Enable memory usage tracking (requires psutil)"
    )
    memory_group.add_argument(
        "--progress-tracking", 
        action="store_true",
        default=True,
        help="Show progress indicators during processing"
    )
    
    # Parallelization options
    parallel_group = parser.add_argument_group('Parallelization options')
    parallel_group.add_argument(
        "--gpu", 
        choices=["true", "false"], 
        default="true",
        help="Use GPU acceleration if available"
    )
    parallel_group.add_argument(
        "--parallel", 
        choices=["true", "false"], 
        default="true",
        help="Use parallel processing with multiple threads"
    )
    parallel_group.add_argument(
        "--pipeline", 
        choices=["true", "false"], 
        default="true",
        help="Use pipeline parallelism with Dask"
    )
    parallel_group.add_argument(
        "--tile-processing", 
        choices=["true", "false"], 
        default="false",
        help="Use tile-based processing for large datasets"
    )
    parallel_group.add_argument(
        "--tile-size", 
        type=int, 
        default=512,
        help="Size of tiles for tile-based processing"
    )
    
    # Micro-motion analysis options
    mm_group = parser.add_argument_group('Micro-motion analysis options')
    mm_group.add_argument(
        "-s", 
        "--subapertures", 
        type=int, 
        default=200,
        help="Number of sub-apertures for micro-motion analysis"
    )
    mm_group.add_argument(
        "--skip-constraints", 
        action="store_true",
        help="Skip physical constraints on vibration analysis"
    )
    mm_group.add_argument(
        "--component-analysis",
        action="store_true",
        default=True,
        help="Enable component-specific vibration analysis"
    )
    
    # Image processing options
    img_group = parser.add_argument_group('Image processing options')
    img_group.add_argument(
        "--speckle-filter", 
        type=int, 
        default=5,
        help="Speckle filter kernel size (0 to disable)"
    )
    img_group.add_argument(
        "--apply-normalization",
        action="store_true",
        default=True,
        help="Apply image normalization for better visualization"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        return 1
    
    # Check memory usage before starting
    if args.memory_tracking:
        print(f"Initial memory usage: {monitor_memory_usage()}")
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if not args.log_file:
        # Create a log file with timestamp in the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        log_filename = os.path.join(
            args.output_dir, 
            f"ship_detection_{timestamp}.log"
        )
    else:
        log_filename = args.log_file
        
        # Ensure directory for log file exists
        log_dir = os.path.dirname(log_filename)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logging(log_filename, level=log_level)
    
    # Print welcome message
    print_separator()
    print(f"Enhanced Ship Detection and Micro-Motion Analysis")
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log file: {log_filename}")
    print("\nAnalysis parameters:")
    print(f"- Sub-apertures: {args.subapertures}")
    print(f"- Speckle filter: {args.speckle_filter if args.speckle_filter > 0 else 'Disabled'}")
    print(f"- Physical constraints: {'Disabled' if args.skip_constraints else 'Enabled'}")
    print(f"- Component analysis: {'Enabled' if args.component_analysis else 'Disabled'}")
    print(f"- Ship selection: {'Manual' if args.manual else 'Automatic'}")
    print(f"- Mode: {'Crop only' if args.crop_only else 'Full analysis'}")
    print("\nPerformance settings:")
    print(f"- Preview downsampling: {args.preview_factor}x")
    print(f"- Memory tracking: {'Enabled' if args.memory_tracking else 'Disabled'}")
    print(f"- Progress tracking: {'Enabled' if args.progress_tracking else 'Disabled'}")
    print("\nParallelization settings:")
    print(f"- GPU acceleration: {'Enabled' if args.gpu.lower() == 'true' else 'Disabled'}")
    print(f"- Thread parallelism: {'Enabled' if args.parallel.lower() == 'true' else 'Disabled'}")
    print(f"- Pipeline parallelism: {'Enabled' if args.pipeline.lower() == 'true' else 'Disabled'}")
    print(f"- Tile-based processing: {'Enabled' if args.tile_processing.lower() == 'true' else 'Disabled'}")
    if args.tile_processing.lower() == "true":
        print(f"- Tile size: {args.tile_size}x{args.tile_size}")
    print_separator()
    
    if args.memory_tracking:
        try:
            import psutil
        except ImportError:
            print("Warning: psutil package not installed. Cannot track memory usage.")
            print("Install with: pip install psutil")
            args.memory_tracking = False
    
    # Add progress tracking
    if args.progress_tracking:
        def progress_callback(stage, progress=None, message=None):
            """Callback function for progress updates."""
            if progress is not None:
                print(f"\r{stage}: {progress:.1f}% - {message}", end="")
            else:
                print(f"\r{stage}{' - ' + message if message else ''}", end="")
                sys.stdout.flush()
    else:
        progress_callback = None
    
    logger.info(f"Starting enhanced ship detection processing for {args.input_file}")
    logger.info(f"Using {args.subapertures} subapertures for micro-motion analysis")
    logger.info(f"Using preview factor {args.preview_factor}x for ship detection")
    
    try:
        # Initialize processor
        processor = EnhancedShipDetectionProcessor(
            args.input_file,
            args.output_dir,
            log_filename,
            use_gpu=args.gpu.lower() == "true",
            use_parallel=args.parallel.lower() == "true",
            use_pipeline=args.pipeline.lower() == "true",
            tile_processing=args.tile_processing.lower() == "true",
            tile_size=args.tile_size
        )
        
        # Override parameters based on command-line arguments
        processor.num_subapertures = args.subapertures
        processor.speckle_filter_size = args.speckle_filter
        processor.skip_constraints = args.skip_constraints
        processor.use_manual_selection = args.manual
        processor.crop_only = args.crop_only
        processor.component_analysis = args.component_analysis
        processor.apply_normalization = args.apply_normalization
        processor.preview_downsample_factor = args.preview_factor
        
        if args.memory_tracking:
            print(f"Memory usage before processing: {monitor_memory_usage()}")
        
        # Run processing
        if args.progress_tracking:
            print("Starting processing pipeline...")
        
        results = processor.process()
        
        if args.progress_tracking:
            print("\rProcessing complete!                                ")
        
        if args.memory_tracking:
            print(f"Memory usage after processing: {monitor_memory_usage()}")
        
        # Check status
        if results['status'] == 'success':
            logger.info("Processing completed successfully")
            print_separator()
            print("Processing completed successfully!")
            
            # Print summary information
            if 'cropped_file' in results:
                logger.info(f"Cropped data saved to {results['cropped_file']}")
                print(f"Cropped data saved to: {results['cropped_file']}")
                print("Use this file for further processing.")
                print_separator()
                return 0
            
            # Detection results
            num_ships = len(results['detection_results']['filtered_ships']) if results['detection_results'] else 0
            logger.info(f"Detected {num_ships} ships")
            print(f"Detected {num_ships} ships")
            
            # Micro-motion analysis
            if results['vibration_results']:
                logger.info("Micro-motion analysis completed")
                print("\nMicro-motion analysis completed:")
                
                # Process results for each ship
                for i, ship_result in enumerate(results['vibration_results'].get('ship_results', [])):
                    ship_num = i + 1
                    logger.info(f"Ship {ship_num} analysis:")
                    print(f"\nShip {ship_num} analysis:")
                    
                    # Get bounding box for reporting
                    if 'bbox' in ship_result:
                        x1, y1, x2, y2 = ship_result['bbox']
                        width, height = x2-x1+1, y2-y1+1
                        print(f"  Location: ({x1},{y1}) to ({x2},{y2}) [size: {width}×{height} pixels]")
                    
                    # Report component information if available
                    if 'component_map' in ship_result:
                        # Count pixels in each component
                        unique, counts = np.unique(ship_result['component_map'], return_counts=True)
                        components = dict(zip(unique, counts))
                        
                        # Skip background (0)
                        if 0 in components:
                            components.pop(0)
                            
                        # Map component IDs to names
                        component_names = {
                            1: 'Hull', 2: 'Deck', 3: 'Superstructure', 
                            4: 'Bow', 5: 'Stern'
                        }
                        
                        print("  Components:")
                        
                        # Print component information
                        for comp_id, count in components.items():
                            comp_name = component_names.get(comp_id, f"Component {comp_id}")
                            logger.info(f"  - {comp_name}: {count} pixels")
                            print(f"    - {comp_name}: {count} pixels")
                            
                            # Get average frequency for this component if available
                            if ('constrained_frequencies' in ship_result and 
                                'constrained_amplitudes' in ship_result):
                                comp_mask = ship_result['component_map'] == comp_id
                                freqs = ship_result['constrained_frequencies'][comp_mask]
                                amps = ship_result['constrained_amplitudes'][comp_mask]
                                
                                # Filter out zeros
                                valid_mask = amps > 0
                                if np.any(valid_mask):
                                    avg_freq = np.average(freqs[valid_mask], weights=amps[valid_mask])
                                    logger.info(f"      Average frequency: {avg_freq:.2f} Hz")
                                    print(f"        Average frequency: {avg_freq:.2f} Hz")
            
            # Print saved files
            if results['saved_files']:
                num_files = len(results['saved_files'])
                logger.info(f"Saved {num_files} result files to {args.output_dir}")
                print(f"\nSaved {num_files} result files to: {args.output_dir}")
                
                # Group files by type
                images = [f for f in results['saved_files'] if f.endswith(('.png', '.jpg', '.jpeg'))]
                data = [f for f in results['saved_files'] if f.endswith(('.json', '.npy', '.npz'))]
                
                if images:
                    print(f"\nGenerated {len(images)} visualizations:")
                    for file_path in images[:5]:  # Show only first 5 to avoid overwhelming output
                        print(f"  - {os.path.basename(file_path)}")
                    if len(images) > 5:
                        print(f"  - ... and {len(images) - 5} more")
                
                if data:
                    print(f"\nSaved {len(data)} data files:")
                    for file_path in data:
                        print(f"  - {os.path.basename(file_path)}")
                
            print_separator()
                
        else:
            logger.error(f"Processing failed: {results.get('message', 'Unknown error')}")
            print(f"Error: Processing failed: {results.get('message', 'Unknown error')}")
            return 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        logger.info("Processing interrupted by user")
        return 130  # Standard return code for SIGINT
    except MemoryError as e:
        logger.exception(f"Memory error during processing: {str(e)}")
        print(f"\nOut of memory error during processing.")
        print(f"Try using a larger preview factor (current: {args.preview_factor}x) or crop the data first.")
        return 1
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        print(f"Error during processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
