#!/usr/bin/env python3
"""
Ship detection module for the improved ship micro-motion analysis pipeline.

This module implements advanced ship detection techniques including:
1. Adaptive CFAR detection
2. Local feature extraction  
3. Morphological filtering
4. Connected component analysis
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


class ShipDetector:
    """
    Advanced ship detector using CFAR and morphological operations.
    """
    
    def __init__(
        self, 
        image_data: np.ndarray,
        cfar_window_size: int = 50,
        cfar_guard_size: int = 5,
        pfa: float = 1e-6,
        min_area: int = 25,
        max_area: int = 5000,
        use_gpu: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ship detector.
        
        Parameters
        ----------
        image_data : np.ndarray
            Complex image data
        cfar_window_size : int, optional
            Window size for CFAR detection, by default 50
        cfar_guard_size : int, optional
            Guard band size for CFAR detection, by default 5
        pfa : float, optional
            Probability of false alarm, by default 1e-6
        min_area : int, optional
            Minimum area for a valid ship, by default 25
        max_area : int, optional
            Maximum area for a valid ship, by default 5000
        use_gpu : bool, optional
            Whether to use GPU acceleration if available, by default False
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        """
        self.image_data = image_data
        self.cfar_window_size = cfar_window_size
        self.cfar_guard_size = cfar_guard_size
        self.pfa = pfa
        self.min_area = min_area
        self.max_area = max_area
        self.use_gpu = use_gpu and check_gpu_availability()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize results
        self.detection_mask = None
        self.labeled_regions = None
        self.ship_regions = []
        self.filtered_ships = []
        
        self.logger.info(f"Ship detector initialized with image shape {image_data.shape}")
        if self.use_gpu:
            self.logger.info("Using GPU acceleration for ship detection")
    
    def run_cfar_detection(self) -> np.ndarray:
        """
        Run Constant False Alarm Rate (CFAR) detection on the image.
        
        Returns
        -------
        np.ndarray
            Binary mask of detected targets
        """
        self.logger.info(f"Running CFAR detection with window={self.cfar_window_size}, guard={self.cfar_guard_size}")
        
        # Take magnitude of complex data
        image_magnitude = np.abs(self.image_data)
        
        # If GPU acceleration is enabled, use CuPy
        if self.use_gpu:
            try:
                import cupy as cp
                self.logger.info("Using CuPy for CFAR detection")
                
                # Transfer data to GPU
                gpu_image = cp.asarray(image_magnitude)
                gpu_mask = cp.zeros_like(gpu_image, dtype=cp.bool_)
                
                # CFAR parameters
                pfa_factor = -cp.log(self.pfa)
                
                # Process using CUDA kernel (simplified example)
                rows, cols = gpu_image.shape
                
                # Run 2D CFAR
                for i in range(self.cfar_window_size, rows - self.cfar_window_size):
                    for j in range(self.cfar_window_size, cols - self.cfar_window_size):
                        # Extract windows
                        outer_window = gpu_image[
                            i - self.cfar_window_size:i + self.cfar_window_size + 1,
                            j - self.cfar_window_size:j + self.cfar_window_size + 1
                        ]
                        
                        guard_window = gpu_image[
                            i - self.cfar_guard_size:i + self.cfar_guard_size + 1,
                            j - self.cfar_guard_size:j + self.cfar_guard_size + 1
                        ]
                        
                        # Flatten outer window
                        outer_flat = outer_window.flatten()
                        guard_flat = guard_window.flatten()
                        
                        # Remove guard cells from background
                        background = cp.setdiff1d(outer_flat, guard_flat)
                        
                        # Calculate statistics
                        bg_mean = cp.mean(background)
                        bg_std = cp.std(background)
                        
                        # Apply CFAR detection
                        threshold = bg_mean + pfa_factor * bg_std
                        
                        if gpu_image[i, j] > threshold:
                            gpu_mask[i, j] = True
                
                # Transfer result back to CPU
                detection_mask = cp.asnumpy(gpu_mask)
                
            except ImportError:
                self.logger.warning("CuPy not available, falling back to CPU implementation")
                detection_mask = self._cpu_cfar_detection(image_magnitude)
        else:
            # Use CPU implementation
            detection_mask = self._cpu_cfar_detection(image_magnitude)
        
        self.detection_mask = detection_mask
        self.logger.info(f"CFAR detection complete. {np.sum(detection_mask)} pixels flagged.")
        
        return detection_mask
    
    def _cpu_cfar_detection(self, image_magnitude: np.ndarray) -> np.ndarray:
        """
        Run 2D CFAR detection on CPU.
        
        Parameters
        ----------
        image_magnitude : np.ndarray
            Magnitude of complex image
            
        Returns
        -------
        np.ndarray
            Binary mask of detected targets
        """
        rows, cols = image_magnitude.shape
        detection_mask = np.zeros_like(image_magnitude, dtype=bool)
        
        # Create efficient moving window sum using integral image method
        # Calculate integral image
        integral_img = np.zeros((rows + 1, cols + 1), dtype=np.float64)
        integral_img[1:, 1:] = np.cumsum(np.cumsum(image_magnitude, axis=0), axis=1)
        
        # CFAR parameters
        pfa_factor = -np.log(self.pfa)
        
        # Calculate statistics using integral image for efficiency
        win_size = self.cfar_window_size
        guard_size = self.cfar_guard_size
        
        # Process inner part of the image (avoiding edges)
        for i in range(win_size, rows - win_size):
            for j in range(win_size, cols - win_size):
                # Calculate outer window sum using integral image
                outer_sum = (
                    integral_img[i + win_size + 1, j + win_size + 1] -
                    integral_img[i + win_size + 1, j - win_size] -
                    integral_img[i - win_size, j + win_size + 1] +
                    integral_img[i - win_size, j - win_size]
                )
                
                # Calculate guard window sum
                guard_sum = (
                    integral_img[i + guard_size + 1, j + guard_size + 1] -
                    integral_img[i + guard_size + 1, j - guard_size] -
                    integral_img[i - guard_size, j + guard_size + 1] +
                    integral_img[i - guard_size, j - guard_size]
                )
                
                # Calculate background statistics
                outer_area = (2 * win_size + 1) ** 2
                guard_area = (2 * guard_size + 1) ** 2
                background_area = outer_area - guard_area
                
                background_mean = (outer_sum - guard_sum) / background_area
                
                # For standard deviation, we need the squared values
                # This is a simplification - ideally we'd have a second integral image
                # of squared values for accurate standard deviation
                background_std = 0.3 * background_mean  # Approximation based on typical clutter
                
                # Apply CFAR detection
                threshold = background_mean + pfa_factor * background_std
                
                if image_magnitude[i, j] > threshold:
                    detection_mask[i, j] = True
        
        return detection_mask
    
    def run_morphological_filtering(self) -> np.ndarray:
        """
        Apply morphological filtering to improve detection mask.
        
        Returns
        -------
        np.ndarray
            Filtered binary mask
        """
        self.logger.info("Applying morphological filtering")
        
        if self.detection_mask is None:
            self.run_cfar_detection()
        
        # Define structuring elements
        close_selem = ndimage.generate_binary_structure(2, 2)
        open_selem = ndimage.generate_binary_structure(2, 1)
        
        # Close small gaps and holes
        closed_mask = ndimage.binary_closing(self.detection_mask, structure=close_selem, iterations=2)
        
        # Remove small noise (false detections)
        opened_mask = ndimage.binary_opening(closed_mask, structure=open_selem, iterations=1)
        
        # Fill remaining holes inside ships
        filled_mask = ndimage.binary_fill_holes(opened_mask)
        
        self.filtered_mask = filled_mask
        self.logger.info(f"Morphological filtering complete. {np.sum(filled_mask)} pixels remain.")
        
        return filled_mask
    
    def run_connected_component_analysis(self) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Identify individual ship regions using connected component analysis.
        
        Returns
        -------
        Tuple[np.ndarray, List[Dict[str, Any]]]
            Labeled image and list of region properties
        """
        self.logger.info("Running connected component analysis")
        
        if not hasattr(self, 'filtered_mask'):
            self.run_morphological_filtering()
        
        # Label connected components
        labeled_image, num_regions = ndimage.label(self.filtered_mask)
        self.logger.info(f"Found {num_regions} connected regions")
        
        # Get region properties
        regions = []
        for region_id in range(1, num_regions + 1):
            # Get region mask
            region_mask = labeled_image == region_id
            
            # Get region coordinates
            coords = np.where(region_mask)
            min_row, max_row = np.min(coords[0]), np.max(coords[0])
            min_col, max_col = np.min(coords[1]), np.max(coords[1])
            
            # Calculate region properties
            bbox = (min_col, min_row, max_col, max_row)  # x1, y1, x2, y2
            width = max_col - min_col + 1
            height = max_row - min_row + 1
            area = np.sum(region_mask)
            
            # Calculate centroids
            row_centroid = np.mean(coords[0])
            col_centroid = np.mean(coords[1])
            
            # Calculate mean intensity
            mean_intensity = np.mean(np.abs(self.image_data[region_mask]))
            
            # Get region image
            region_image = self.image_data[min_row:max_row+1, min_col:max_col+1]
            region_mask_crop = region_mask[min_row:max_row+1, min_col:max_col+1]
            
            # Store region data
            region = {
                'id': region_id,
                'bbox': bbox,  # x1, y1, x2, y2
                'center': ((min_col + max_col) // 2, (min_row + max_row) // 2),
                'centroid': (row_centroid, col_centroid),
                'width': width,
                'height': height,
                'area': area,
                'mean_intensity': mean_intensity,
                'region': region_image,
                'mask': region_mask_crop
            }
            
            regions.append(region)
        
        self.labeled_regions = labeled_image
        self.ship_regions = regions
        
        return labeled_image, regions
    
    def filter_ships(
        self, 
        min_area: Optional[int] = None, 
        max_area: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter detected regions based on size and other characteristics.
        
        Parameters
        ----------
        min_area : Optional[int], optional
            Minimum area for a valid ship, by default None (uses instance value)
        max_area : Optional[int], optional
            Maximum area for a valid ship, by default None (uses instance value)
            
        Returns
        -------
        List[Dict[str, Any]]
            List of filtered ship regions
        """
        self.logger.info("Filtering ship regions")
        
        if not self.ship_regions:
            self.run_connected_component_analysis()
        
        # Use provided parameters or instance values
        min_area = min_area or self.min_area
        max_area = max_area or self.max_area
        
        # Filter ships based on area
        filtered_ships = [
            ship for ship in self.ship_regions
            if min_area <= ship['area'] <= max_area
        ]
        
        # Filter ships based on aspect ratio (not too elongated)
        filtered_ships = [
            ship for ship in filtered_ships
            if 0.1 <= ship['width'] / ship['height'] <= 10.0
        ]
        
        # Sort by size (largest first)
        filtered_ships.sort(key=lambda x: x['area'], reverse=True)
        
        self.filtered_ships = filtered_ships
        self.logger.info(f"Filtered to {len(filtered_ships)} valid ships")
        
        return filtered_ships
    
    def process_all(self) -> Dict[str, Any]:
        """
        Run the complete ship detection pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all detection results
        """
        self.logger.info("Running complete ship detection pipeline")
        
        # Run all steps
        self.run_cfar_detection()
        self.run_morphological_filtering()
        self.run_connected_component_analysis()
        self.filter_ships()
        
        # Create detection results dictionary
        results = {
            'detection_mask': self.detection_mask,
            'filtered_mask': self.filtered_mask,
            'labeled_regions': self.labeled_regions,
            'ship_regions': self.ship_regions,
            'filtered_ships': self.filtered_ships,
            'num_ships': len(self.filtered_ships)
        }
        
        self.logger.info("Ship detection pipeline complete")
        
        return results


def detect_ships(
    input_file: str,
    output_file: str,
    cfar_window_size: int = 50,
    cfar_guard_size: int = 5,
    pfa: float = 1e-6,
    min_area: int = 25,
    max_area: int = 5000,
    use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run ship detection on preprocessed SAR data.
    
    Parameters
    ----------
    input_file : str
        Path to input file (preprocessed SAR data)
    output_file : str
        Path to output file
    cfar_window_size : int, optional
        Window size for CFAR detection, by default 50
    cfar_guard_size : int, optional
        Guard band size for CFAR detection, by default 5
    pfa : float, optional
        Probability of false alarm, by default 1e-6
    min_area : int, optional
        Minimum area for a valid ship, by default 25
    max_area : int, optional
        Maximum area for a valid ship, by default 5000
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing detection results
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Detecting ships in {input_file}")
    
    # Load preprocessed data
    preprocessed = load_step_output(input_file)
    logger.info(f"Loaded preprocessed data with keys: {preprocessed.keys()}")
    
    # Use filtered image if available, otherwise focused image
    if 'filtered_image' in preprocessed:
        image_data = preprocessed['filtered_image']
        logger.info("Using filtered image for detection")
    elif 'focused_image' in preprocessed:
        image_data = preprocessed['focused_image']
        logger.info("Using focused image for detection")
    elif 'preview_image' in preprocessed:
        image_data = preprocessed['preview_image']
        logger.info("Using preview image for detection")
    else:
        raise ValueError("No suitable image found in preprocessed data")
    
    # Initialize ship detector
    detector = ShipDetector(
        image_data=image_data,
        cfar_window_size=cfar_window_size,
        cfar_guard_size=cfar_guard_size,
        pfa=pfa,
        min_area=min_area,
        max_area=max_area,
        use_gpu=use_gpu,
        logger=logger
    )
    
    # Run detection pipeline
    detection_results = detector.process_all()
    
    # Save original image shape for coordinate conversion if needed
    if 'original_shape' in preprocessed and 'preview_shape' in preprocessed:
        detection_results['original_shape'] = preprocessed['original_shape']
        detection_results['preview_shape'] = preprocessed['preview_shape']
        detection_results['preview_factor'] = preprocessed.get('preview_factor', 1)
    
    # Add timestamp and metadata
    detection_results['timestamp'] = preprocessed.get('timestamp', None)
    detection_results['input_file'] = input_file
    
    # Save detection results
    save_results(output_file, detection_results)
    logger.info(f"Saved detection results to {output_file}")
    
    # Create a visualization
    create_detection_visualization(
        image_data, detection_results, 
        os.path.splitext(output_file)[0] + '_detection.png'
    )
    
    return detection_results


def create_detection_visualization(
    image_data: np.ndarray, 
    detection_results: Dict[str, Any],
    output_path: str
) -> None:
    """
    Create a visualization of ship detection results.
    
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
    
    ax.set_title(f"Detected Ships ({len(detection_results['filtered_ships'])})")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ship Detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input file (preprocessed SAR data)')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--cfar-window-size', type=int, default=50,
                      help='Window size for CFAR detection')
    parser.add_argument('--cfar-guard-size', type=int, default=5,
                      help='Guard band size for CFAR detection')
    parser.add_argument('--pfa', type=float, default=1e-6,
                      help='Probability of false alarm')
    parser.add_argument('--min-area', type=int, default=25,
                      help='Minimum area for a valid ship')
    parser.add_argument('--max-area', type=int, default=5000,
                      help='Maximum area for a valid ship')
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
        # Run ship detection
        result = detect_ships(
            args.input,
            args.output,
            args.cfar_window_size,
            args.cfar_guard_size,
            args.pfa,
            args.min_area,
            args.max_area,
            args.use_gpu,
            logger
        )
        logger.info("Ship detection completed successfully")
    except Exception as e:
        logger.error(f"Error during ship detection: {str(e)}")
        sys.exit(1) 