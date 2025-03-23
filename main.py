"""
Ship detection and micro-motion analysis from SAR imagery.
"""

import os
import argparse
import logging
from datetime import datetime

from src.ship_detection.processor import ShipDetectionProcessor
from src.ship_detection.utils.helpers import setup_logging

def main():
    """
    Main entry point for the ship detection and micro-motion analysis application.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Ship detection and micro-motion analysis from SAR imagery"
    )
    parser.add_argument(
        "input_file", 
        help="Path to input SAR data file (CPHD, SICD, or NITF format)"
    )
    parser.add_argument(
        "-o", 
        "--output-dir", 
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "-l", 
        "--log-file", 
        help="Path to log file (default: None)"
    )
    parser.add_argument(
        "-v", 
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    if not args.log_file:
        # Create a log file with timestamp in the output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(
            args.output_dir, 
            f"ship_detection_{timestamp}.log"
        )
    else:
        log_filename = args.log_file
        
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    logger = setup_logging(log_filename, level=log_level)
    logger.info(f"Starting ship detection processing for {args.input_file}")
    
    try:
        # Initialize processor
        processor = ShipDetectionProcessor(
            args.input_file,
            args.output_dir,
            log_filename
        )
        
        # Run processing
        results = processor.process()
        
        # Check status
        if results['status'] == 'success':
            logger.info("Processing completed successfully")
            
            # Print summary information
            num_ships = len(results['detection_results']['filtered_ships']) if results['detection_results'] else 0
            logger.info(f"Detected {num_ships} ships")
            
            if results['vibration_results']:
                logger.info("Vibration analysis completed")
                
                # Print dominant frequencies if available
                if 'vibration_params' in results['vibration_results']:
                    vib_params = results['vibration_results']['vibration_params']
                    if 'dominant_freq_range' in vib_params:
                        logger.info(f"Dominant range frequency: {vib_params['dominant_freq_range']:.2f} Hz")
                    if 'dominant_freq_azimuth' in vib_params:
                        logger.info(f"Dominant azimuth frequency: {vib_params['dominant_freq_azimuth']:.2f} Hz")
            
            # Print saved files
            if results['saved_files']:
                logger.info(f"Saved {len(results['saved_files'])} result files to {args.output_dir}")
                
        else:
            logger.error(f"Processing failed: {results.get('message', 'Unknown error')}")
            
    except Exception as e:
        logger.exception(f"Error during processing: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
