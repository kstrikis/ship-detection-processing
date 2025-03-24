#!/usr/bin/env python3
"""
Ship component classification module for the improved ship micro-motion analysis pipeline.

This module implements classification of different ship components (hull, deck, 
superstructure, bow, stern) to enable component-specific vibration analysis.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from .utils import (
    setup_logging, save_results, load_step_output, check_gpu_availability,
    scale_for_display
)


class ShipComponentClassifier:
    """
    Classifies ship pixels into different structural components.
    """
    
    def __init__(
        self, 
        use_gpu: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the component classifier.
        
        Parameters
        ----------
        use_gpu : bool, optional
            Whether to use GPU acceleration if available, by default False
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        """
        self.use_gpu = use_gpu and check_gpu_availability()
        self.logger = logger or logging.getLogger(__name__)
        
        # Component IDs
        self.component_ids = {
            0: 'background',
            1: 'hull',
            2: 'deck',
            3: 'superstructure',
            4: 'bow',
            5: 'stern'
        }
        
        if self.use_gpu:
            self.logger.info("Using GPU acceleration for component classification")
    
    def classify_components(
        self, 
        ship_region: np.ndarray, 
        intensity_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Segment a ship into components based on image features.
        
        Parameters
        ----------
        ship_region : np.ndarray
            Ship region image data
        intensity_mask : Optional[np.ndarray], optional
            Binary mask of ship pixels, by default None
            
        Returns
        -------
        np.ndarray
            Component map with component IDs
        """
        self.logger.info(f"Classifying ship components for region with shape {ship_region.shape}")
        
        # Take magnitude if complex
        if np.iscomplexobj(ship_region):
            ship_magnitude = np.abs(ship_region)
        else:
            ship_magnitude = ship_region
        
        # Create mask if not provided
        if intensity_mask is None:
            intensity_mask = ship_magnitude > np.mean(ship_magnitude)
            self.logger.info("Created intensity mask using mean threshold")
        
        # Create empty component map (0 = background)
        component_map = np.zeros_like(intensity_mask, dtype=int)
        
        # Check if GPU acceleration is available and requested
        if self.use_gpu:
            try:
                import cupy as cp
                self.logger.info("Using GPU acceleration with CuPy for component classification")
                
                # Transfer data to GPU
                gpu_ship_region = cp.asarray(ship_magnitude)
                gpu_intensity_mask = cp.asarray(intensity_mask)
                
                # Extract features for classification
                # Use Sobel filters to compute gradient
                gradient_x = cp.zeros_like(gpu_ship_region)
                gradient_y = cp.zeros_like(gpu_ship_region)
                
                # Implement Sobel filter on GPU
                rows, cols = gpu_ship_region.shape
                
                # This is an inefficient implementation - ideally would use a CUDA kernel
                # But for demonstration, we'll use this simple loop-based approach
                for i in range(1, rows - 1):
                    for j in range(1, cols - 1):
                        # X gradient (horizontal)
                        gradient_x[i, j] = (
                            (gpu_ship_region[i-1, j+1] + 2*gpu_ship_region[i, j+1] + gpu_ship_region[i+1, j+1]) -
                            (gpu_ship_region[i-1, j-1] + 2*gpu_ship_region[i, j-1] + gpu_ship_region[i+1, j-1])
                        )
                        
                        # Y gradient (vertical)
                        gradient_y[i, j] = (
                            (gpu_ship_region[i+1, j-1] + 2*gpu_ship_region[i+1, j] + gpu_ship_region[i+1, j+1]) -
                            (gpu_ship_region[i-1, j-1] + 2*gpu_ship_region[i-1, j] + gpu_ship_region[i-1, j+1])
                        )
                
                gradient_magnitude = cp.sqrt(gradient_x**2 + gradient_y**2)
                
                # Transfer back to CPU for additional processing
                gradient_magnitude_cpu = cp.asnumpy(gradient_magnitude)
                intensity_mask_cpu = cp.asnumpy(gpu_intensity_mask)
                component_map_cpu = self._cpu_classify_components(
                    ship_magnitude, gradient_magnitude_cpu, intensity_mask_cpu)
                
                return component_map_cpu
                
            except (ImportError, ModuleNotFoundError):
                self.logger.warning("CuPy not available, falling back to CPU implementation")
                return self._cpu_classify_components(ship_magnitude, None, intensity_mask)
        else:
            # Use CPU implementation
            return self._cpu_classify_components(ship_magnitude, None, intensity_mask)
    
    def _cpu_classify_components(
        self, 
        ship_magnitude: np.ndarray, 
        gradient_magnitude: Optional[np.ndarray] = None,
        intensity_mask: np.ndarray = None
    ) -> np.ndarray:
        """
        Classify ship components using CPU implementation.
        
        Parameters
        ----------
        ship_magnitude : np.ndarray
            Ship region magnitude image
        gradient_magnitude : Optional[np.ndarray], optional
            Pre-computed gradient magnitude, by default None
        intensity_mask : np.ndarray, optional
            Binary mask of ship pixels, by default None
            
        Returns
        -------
        np.ndarray
            Component map with component IDs
        """
        self.logger.info("Using CPU implementation for component classification")
        
        # Calculate gradient if not provided
        if gradient_magnitude is None:
            gradient_x = ndimage.sobel(ship_magnitude, axis=1)
            gradient_y = ndimage.sobel(ship_magnitude, axis=0)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Create mask if not provided
        if intensity_mask is None:
            intensity_mask = ship_magnitude > np.mean(ship_magnitude)
        
        # Initialize component map (0 = background)
        component_map = np.zeros_like(intensity_mask, dtype=int)
        
        # Get ship dimensions
        rows, cols = ship_magnitude.shape
        
        # Define bow and stern regions (front and back quarters)
        left_quarter = int(cols * 0.25)
        right_quarter = int(cols * 0.75)
        
        # Check gradient patterns to determine which end is bow
        # Bows typically have sharper gradients due to pointed shape
        left_gradients = np.sum(gradient_magnitude[:, :left_quarter])
        right_gradients = np.sum(gradient_magnitude[:, right_quarter:])
        
        # Find vertical center of the ship
        ship_rows = np.where(np.any(intensity_mask, axis=1))[0]
        if len(ship_rows) > 0:
            top_row = np.min(ship_rows)
            bottom_row = np.max(ship_rows)
            vertical_center = (top_row + bottom_row) // 2
        else:
            vertical_center = rows // 2
        
        # Initialize different component areas
        if left_gradients > right_gradients:
            # Bow is on the left
            component_map[:, :left_quarter][intensity_mask[:, :left_quarter]] = 4  # Bow
            component_map[:, right_quarter:][intensity_mask[:, right_quarter:]] = 5  # Stern
        else:
            # Bow is on the right
            component_map[:, right_quarter:][intensity_mask[:, right_quarter:]] = 4  # Bow
            component_map[:, :left_quarter][intensity_mask[:, :left_quarter]] = 5  # Stern
        
        # Central section (middle 50%)
        middle_section = intensity_mask.copy()
        middle_section[:, :left_quarter] = False
        middle_section[:, right_quarter:] = False
        
        # Split into hull (bottom), deck (middle) and superstructure (top)
        # First find the vertical extents of the ship
        vertical_profile = np.sum(middle_section, axis=1)
        active_rows = np.where(vertical_profile > 0)[0]
        
        if len(active_rows) > 0:
            top_row = np.min(active_rows)
            bottom_row = np.max(active_rows)
            center_row = (top_row + bottom_row) // 2
            
            # Divide into three regions vertically
            top_third = top_row + (center_row - top_row) // 2
            bottom_third = center_row + (bottom_row - center_row) // 2
            
            # Hull (bottom part)
            hull_mask = middle_section.copy()
            hull_mask[:bottom_third, :] = False
            component_map[hull_mask] = 1
            
            # Deck (middle part)
            deck_mask = middle_section.copy()
            deck_mask[:top_third, :] = False
            deck_mask[bottom_third:, :] = False
            component_map[deck_mask] = 2
            
            # Superstructure (top part)
            superstructure_mask = middle_section.copy()
            superstructure_mask[top_third:, :] = False
            component_map[superstructure_mask] = 3
        else:
            # Fallback if we can't find active rows
            # Split into equal thirds
            top_third = rows // 3
            bottom_third = 2 * (rows // 3)
            
            # Hull (bottom part)
            hull_mask = middle_section.copy()
            hull_mask[:bottom_third, :] = False
            component_map[hull_mask] = 1
            
            # Deck (middle part)
            deck_mask = middle_section.copy()
            deck_mask[:top_third, :] = False
            deck_mask[bottom_third:, :] = False
            component_map[deck_mask] = 2
            
            # Superstructure (top part)
            superstructure_mask = middle_section.copy()
            superstructure_mask[top_third:, :] = False
            component_map[superstructure_mask] = 3
        
        # Clean up with morphological operations
        for component_id in range(1, 6):
            component_mask = component_map == component_id
            # Close gaps
            closed_mask = ndimage.binary_closing(component_mask)
            # Fill holes
            filled_mask = ndimage.binary_fill_holes(closed_mask)
            # Update component map
            component_map[filled_mask] = component_id
        
        return component_map


def classify_components(
    input_file: str,
    ship_file: str,
    output_file: str,
    use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Classify ship components from phase history data.
    
    Parameters
    ----------
    input_file : str
        Path to input file (phase history data)
    ship_file : str
        Path to ship detection results
    output_file : str
        Path to output file
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing component classification results
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Classifying ship components in {input_file} with ship data from {ship_file}")
    
    # Load phase history data
    phase_data = load_step_output(input_file)
    logger.info(f"Loaded phase history data with keys: {phase_data.keys()}")
    
    # Load ship detection results
    ships = load_step_output(ship_file)
    logger.info(f"Loaded ship detection data with keys: {ships.keys()}")
    
    # Initialize component classifier
    classifier = ShipComponentClassifier(
        use_gpu=use_gpu,
        logger=logger
    )
    
    # Process each ship
    ship_results = []
    filtered_ships = ships.get('filtered_ships', [])
    
    for i, ship in enumerate(filtered_ships):
        logger.info(f"Processing ship {i+1}/{len(filtered_ships)}")
        
        # Extract ship region for classification
        bbox = ship.get('bbox')
        if bbox is None:
            logger.warning(f"No bounding box found for ship {i+1}, skipping")
            continue
        
        # Try to find corresponding phase history data
        phase_ship = None
        for phase_ship_result in phase_data.get('ship_results', []):
            if phase_ship_result.get('ship_index', -1) == i:
                phase_ship = phase_ship_result
                break
        
        if phase_ship is None:
            logger.warning(f"No phase history data found for ship {i+1}, skipping")
            continue
        
        # Extract and process the ship region
        ship_region = ship.get('region')
        mask = ship.get('mask')
        
        if ship_region is None:
            logger.warning(f"No region data found for ship {i+1}, skipping")
            continue
        
        # Classify ship components
        component_map = classifier.classify_components(ship_region, mask)
        
        # Calculate component statistics
        component_stats = {}
        for comp_id in range(1, 6):
            comp_mask = component_map == comp_id
            comp_stats[comp_id] = {
                'name': classifier.component_ids[comp_id],
                'pixel_count': np.sum(comp_mask),
                'percentage': 100 * np.sum(comp_mask) / np.sum(component_map > 0) if np.sum(component_map > 0) > 0 else 0
            }
        
        # Store results
        result = {
            'ship_index': i,
            'bbox': bbox,
            'component_map': component_map,
            'component_stats': comp_stats
        }
        
        ship_results.append(result)
    
    # Create complete results dictionary
    results = {
        'ship_results': ship_results,
        'num_ships': len(ship_results),
        'component_ids': classifier.component_ids,
        'timestamp': phase_data.get('timestamp', None),
        'input_file': input_file,
        'ship_file': ship_file
    }
    
    # Save results
    save_results(output_file, results)
    logger.info(f"Saved component classification results to {output_file}")
    
    # Create visualization for each ship
    create_component_visualizations(
        ship_results, 
        classifier.component_ids,
        os.path.splitext(output_file)[0]
    )
    
    return results


def create_component_visualizations(
    ship_results: List[Dict[str, Any]],
    component_ids: Dict[int, str],
    output_prefix: str
) -> None:
    """
    Create visualizations of ship component classifications.
    
    Parameters
    ----------
    ship_results : List[Dict[str, Any]]
        List of ship component classification results
    component_ids : Dict[int, str]
        Dictionary mapping component IDs to names
    output_prefix : str
        Prefix for output file paths
    """
    # Define colors for each component
    colors = {
        0: [0, 0, 0, 0],          # Background (transparent)
        1: [0, 0, 1, 0.7],        # Hull (blue)
        2: [0, 0.7, 0, 0.7],      # Deck (green)
        3: [1, 0, 0, 0.7],        # Superstructure (red)
        4: [1, 0.7, 0, 0.7],      # Bow (orange)
        5: [0.7, 0, 1, 0.7]       # Stern (purple)
    }
    
    for i, ship in enumerate(ship_results):
        # Skip if no component map available
        if 'component_map' not in ship:
            continue
            
        component_map = ship['component_map']
        component_stats = ship.get('component_stats', {})
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display component map using colors
        component_rgb = np.zeros((*component_map.shape, 4))
        for comp_id, color in colors.items():
            mask = component_map == comp_id
            for c in range(4):  # RGBA channels
                component_rgb[mask, c] = color[c]
        
        ax.imshow(component_rgb)
        
        # Create legend
        legend_elements = []
        for comp_id, name in component_ids.items():
            if comp_id > 0:  # Skip background
                color = colors[comp_id]
                # Check if this component exists in the ship
                if comp_id in component_stats and component_stats[comp_id]['pixel_count'] > 0:
                    pixel_count = component_stats[comp_id]['pixel_count']
                    percentage = component_stats[comp_id]['percentage']
                    label = f"{name.capitalize()}: {pixel_count} px ({percentage:.1f}%)"
                else:
                    label = name.capitalize()
                    
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, color=color, label=label)
                )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        ax.set_title(f"Ship {i+1} - Component Classification")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save figure
        component_file = f"{output_prefix}_ship{i+1}_components.png"
        plt.tight_layout()
        plt.savefig(component_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ship Component Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input file (phase history data)')
    parser.add_argument('--ship-file', type=str, required=True,
                      help='Ship detection results file')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--log-file', type=str,
                      help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    try:
        # Run component classification
        result = classify_components(
            args.input,
            args.ship_file,
            args.output,
            args.use_gpu,
            logger
        )
        logger.info("Component classification completed successfully")
    except Exception as e:
        logger.error(f"Error during component classification: {str(e)}")
        sys.exit(1) 