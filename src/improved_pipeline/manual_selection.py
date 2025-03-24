#!/usr/bin/env python3
"""
Manual ship selection module for the improved ship micro-motion analysis pipeline.

This module provides a graphical interface for manually selecting ships in SAR images.
It's particularly useful when automatic detection fails or needs to be refined.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from .utils import (
    setup_logging, save_results, load_step_output, scale_for_display,
    scale_coordinates
)


def manually_select_ships(
    input_file: str,
    output_file: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Allow manual selection of ships in the image.
    
    Parameters
    ----------
    input_file : str
        Path to input file (preprocessed SAR data)
    output_file : str
        Path to output file
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing selected ship regions
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Starting manual ship selection for {input_file}")
    
    # Load preprocessed data
    preprocessed = load_step_output(input_file)
    logger.info(f"Loaded preprocessed data with keys: {preprocessed.keys()}")
    
    # Use the most appropriate image for display
    if 'preview_image' in preprocessed:
        # Prefer preview image for faster interaction
        display_image = preprocessed['preview_image']
        using_preview = True
        preview_factor = preprocessed.get('preview_factor', 1)
        original_shape = preprocessed.get('original_shape', display_image.shape)
        preview_shape = display_image.shape
        logger.info(f"Using preview image with shape {display_image.shape}")
    elif 'filtered_image' in preprocessed:
        display_image = preprocessed['filtered_image']
        using_preview = False
        logger.info(f"Using filtered image with shape {display_image.shape}")
    elif 'focused_image' in preprocessed:
        display_image = preprocessed['focused_image']
        using_preview = False
        logger.info(f"Using focused image with shape {display_image.shape}")
    else:
        logger.error("No suitable image found in preprocessed data")
        raise ValueError("No suitable image found in preprocessed data")
    
    # Scale image data for better visualization
    display_data = scale_for_display(display_image)
    
    # Store selected regions
    selected_regions = []
    
    # Callback function for selection
    def onselect(eclick, erelease):
        """Store the coordinates of the selected region."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        
        # Ensure coordinates are in bounds
        x1 = max(0, min(x1, display_image.shape[1] - 1))
        y1 = max(0, min(y1, display_image.shape[0] - 1))
        x2 = max(0, min(x2, display_image.shape[1] - 1))
        y2 = max(0, min(y2, display_image.shape[0] - 1))
        
        # Swap if needed to ensure x1 < x2 and y1 < y2
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Add region with coordinates in the displayed image
        region_data = {
            'bbox': (x1, y1, x2, y2),
            'width': x2 - x1 + 1,
            'height': y2 - y1 + 1,
            'area': (x2 - x1 + 1) * (y2 - y1 + 1),
            'id': len(selected_regions) + 1
        }
        
        selected_regions.append(region_data)
        
        # Draw rectangle on the plot
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        
        # Display region number
        plt.text(x1, y1-5, f"Ship {len(selected_regions)}", 
                color='red', fontsize=10, backgroundcolor='white')
        
        plt.draw()
        logger.info(f"Added ship region {len(selected_regions)} at ({x1},{y1})-({x2},{y2})")
    
    # Create figure for selection
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.imshow(display_data, cmap='gray')
    plt.title('Select ships by drawing rectangles\nPress Enter when finished, Escape to cancel selection')
    plt.colorbar(label='Normalized Amplitude (dB)')
    
    # Add instructions text
    if using_preview:
        info_text = f'Using downsampled preview ({preview_factor}x) for efficiency. '
    else:
        info_text = 'Using full resolution image. '
        
    plt.figtext(0.5, 0.01, 
                info_text + 'Click and drag to select ships. Press Enter when done.', 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
    
    # Create RectangleSelector
    rect_selector = RectangleSelector(
        ax, onselect, useblit=True,
        button=[1],  # Left mouse button only
        minspanx=5, minspany=5,  # Minimum selection size
        spancoords='pixels',
        interactive=True
    )
    
    # Function to handle key press events
    def on_key_press(event):
        if event.key == 'enter':
            plt.close()
        elif event.key == 'escape':
            selected_regions.clear()
            plt.close()
    
    # Connect key press event
    fig.canvas.mpl_connect('key_press_event', on_key_press)
    
    # Show the plot and wait for user interaction
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Manual selection complete. {len(selected_regions)} ships selected.")
    
    # If using preview, scale coordinates back to original
    if using_preview and preview_factor > 1:
        logger.info("Scaling selection coordinates to original resolution...")
        
        # Get the full resolution image data
        if 'filtered_image' in preprocessed:
            full_image = preprocessed['filtered_image']
        elif 'focused_image' in preprocessed:
            full_image = preprocessed['focused_image']
        else:
            logger.warning("No full resolution image found, using preview as is")
            full_image = display_image
        
        # Scale and extract from original for each region
        original_regions = []
        for region in selected_regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Scale back to original coordinates
            x1_orig, y1_orig, x2_orig, y2_orig = scale_coordinates(
                (x1, y1, x2, y2), original_shape, preview_shape)
            
            # Extract region from full resolution image
            orig_region = full_image[y1_orig:y2_orig+1, x1_orig:x2_orig+1]
            
            # Create mask
            mask = np.ones((y2_orig-y1_orig+1, x2_orig-x1_orig+1), dtype=bool)
            
            # Calculate center
            center_y = (y1_orig + y2_orig) // 2
            center_x = (x1_orig + x2_orig) // 2
            
            # Add to original regions
            orig_region_data = {
                'id': region['id'],
                'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                'center': (center_x, center_y),
                'centroid': (center_y, center_x),  # Row, column format
                'region': orig_region,
                'mask': mask,
                'width': x2_orig - x1_orig + 1,
                'height': y2_orig - y1_orig + 1,
                'area': (x2_orig - x1_orig + 1) * (y2_orig - y1_orig + 1)
            }
            
            original_regions.append(orig_region_data)
        
        selected_regions = original_regions
            
    else:
        # Extract regions from displayed image
        for region in selected_regions:
            x1, y1, x2, y2 = region['bbox']
            
            # Extract region
            region['region'] = display_image[y1:y2+1, x1:x2+1]
            
            # Create mask
            region['mask'] = np.ones((y2-y1+1, x2-x1+1), dtype=bool)
            
            # Calculate center
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            
            region['center'] = (center_x, center_y)  # Column, row format
            region['centroid'] = (center_y, center_x)  # Row, column format
    
    # Format results similar to automatic detection
    detection_results = {
        'filtered_ships': selected_regions,
        'ship_regions': selected_regions,
        'num_ships': len(selected_regions),
        'manual_selection': True,
        'timestamp': preprocessed.get('timestamp', None),
        'input_file': input_file
    }
    
    # Save detection results
    save_results(output_file, detection_results)
    logger.info(f"Saved manual selection results to {output_file}")
    
    # Create a visualization
    create_selection_visualization(
        display_image if not using_preview else full_image,
        detection_results,
        os.path.splitext(output_file)[0] + '_selection.png'
    )
    
    return detection_results


def create_selection_visualization(
    image_data: np.ndarray, 
    detection_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create a visualization of manually selected ship regions.
    
    Parameters
    ----------
    image_data : np.ndarray
        SAR image data
    detection_results : Dict[str, Any]
        Ship detection results
    output_path : str
        Path to save visualization
    """
    # Scale image for display
    display_data = scale_for_display(image_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Display the image
    ax.imshow(display_data, cmap='gray')
    
    # Overlay ships
    for i, ship in enumerate(detection_results['filtered_ships']):
        x1, y1, x2, y2 = ship['bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"Ship {i+1}", 
               color='red', fontsize=10, backgroundcolor='white')
    
    ax.set_title(f"Manually Selected Ships ({len(detection_results['filtered_ships'])})")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manual Ship Selector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input file (preprocessed SAR data)')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--log-file', type=str,
                      help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    try:
        # Run manual ship selection
        result = manually_select_ships(
            args.input,
            args.output,
            logger
        )
        logger.info("Manual ship selection completed successfully")
    except Exception as e:
        logger.error(f"Error during manual ship selection: {str(e)}")
        sys.exit(1) 