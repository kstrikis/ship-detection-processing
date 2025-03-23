"""
Module for ship detection in SAR images.
"""

import logging
from typing import Dict, Tuple, List, Optional, Union, Any

import numpy as np
from scipy import ndimage
from skimage import filters, morphology, measure, segmentation

logger = logging.getLogger(__name__)

class ShipDetector:
    """
    Class for detecting ships in SAR images and extracting regions of interest.
    """
    
    def __init__(self, image_data: np.ndarray, window_size: int = 101):
        """
        Initialize the ship detector.
        
        Parameters
        ----------
        image_data : np.ndarray
            The SAR image data.
        window_size : int, optional
            Window size for local statistics, by default 101
        """
        self.image_data = image_data
        self.window_size = window_size
        self.detected_ships = []
        self.ship_masks = None
        self.ship_properties = None
    
    def preprocess(self) -> np.ndarray:
        """
        Preprocess the image to enhance ship features.
        
        Returns
        -------
        np.ndarray
            Preprocessed image.
        """
        # Convert to magnitude if complex
        if np.iscomplexobj(self.image_data):
            magnitude = np.abs(self.image_data)
        else:
            magnitude = self.image_data
            
        # Apply logarithmic transformation to handle dynamic range
        log_image = np.log10(magnitude + 1)
        
        # Normalize to [0, 1]
        normalized = (log_image - np.min(log_image)) / (np.max(log_image) - np.min(log_image))
        
        # Apply median filter to reduce speckle noise
        median_filtered = ndimage.median_filter(normalized, size=3)
        
        return median_filtered
    
    def detect_ships_by_cfar(self, pfa: float = 1e-5, guard_size: int = 3, 
                           bg_size: int = 5) -> np.ndarray:
        """
        Detect ships using Constant False Alarm Rate (CFAR) detector.
        
        Parameters
        ----------
        pfa : float, optional
            Probability of false alarm, by default 1e-5
        guard_size : int, optional
            Size of guard cells, by default 3
        bg_size : int, optional
            Size of background cells, by default 5
            
        Returns
        -------
        np.ndarray
            Binary mask of detected ships.
        """
        preprocessed = self.preprocess()
        
        # Total window size
        window_radius = bg_size + guard_size
        
        # Initialize binary mask
        ship_mask = np.zeros_like(preprocessed, dtype=bool)
        
        # Pad the image to handle edges
        padded = np.pad(preprocessed, window_radius, mode='reflect')
        
        # Get image dimensions
        rows, cols = preprocessed.shape
        
        # Compute threshold factor from PFA using inverse CDF of exponential distribution
        threshold_factor = -np.log(pfa)
        
        # Apply CFAR detection
        for i in range(rows):
            for j in range(cols):
                # Extract CUT (Cell Under Test)
                cut = preprocessed[i, j]
                
                # Extract background window
                bg_window = padded[i:i+2*window_radius+1, j:j+2*window_radius+1]
                
                # Create guard and background masks
                guard_mask = np.ones((2*window_radius+1, 2*window_radius+1), dtype=bool)
                guard_mask[window_radius-guard_size:window_radius+guard_size+1, 
                          window_radius-guard_size:window_radius+guard_size+1] = False
                
                # Extract background values
                bg_values = bg_window[guard_mask]
                
                # Compute background statistics (mean)
                bg_mean = np.mean(bg_values)
                
                # Apply threshold
                if cut > threshold_factor * bg_mean:
                    ship_mask[i, j] = True
        
        # Apply morphological operations to clean up mask
        ship_mask = morphology.opening(ship_mask, morphology.disk(1))
        ship_mask = morphology.closing(ship_mask, morphology.disk(2))
        
        self.ship_masks = ship_mask
        return ship_mask
    
    def detect_ships_by_adaptive_threshold(self, sensitivity: float = 0.7,
                                         min_area: int = 20) -> np.ndarray:
        """
        Detect ships using adaptive thresholding and morphological operations.
        
        Parameters
        ----------
        sensitivity : float, optional
            Threshold sensitivity (0-1), by default 0.7
        min_area : int, optional
            Minimum area for ship candidates, by default 20
            
        Returns
        -------
        np.ndarray
            Binary mask of detected ships.
        """
        preprocessed = self.preprocess()
        
        # Apply adaptive thresholding
        thresh = filters.threshold_otsu(preprocessed)
        binary = preprocessed > (thresh * sensitivity)
        
        # Apply morphological operations to clean up
        binary = morphology.remove_small_objects(binary, min_size=min_area)
        binary = morphology.opening(binary, morphology.disk(1))
        binary = morphology.closing(binary, morphology.disk(2))
        
        # Label connected components
        labeled, num_ships = ndimage.label(binary)
        
        # Filter small regions
        for i in range(1, num_ships + 1):
            if np.sum(labeled == i) < min_area:
                binary[labeled == i] = 0
        
        logger.info(f"Detected {np.max(labeled)} ship candidates")
        
        self.ship_masks = binary
        return binary
    
    def extract_ship_regions(self, mask: Optional[np.ndarray] = None,
                            padding: int = 10) -> List[Dict[str, Any]]:
        """
        Extract individual ship regions from the image.
        
        Parameters
        ----------
        mask : Optional[np.ndarray], optional
            Binary mask indicating ship locations, by default None
        padding : int, optional
            Padding around each ship region, by default 10
            
        Returns
        -------
        List[Dict[str, Any]]
            List of dictionaries containing ship regions and properties.
        """
        if mask is None:
            if self.ship_masks is None:
                logger.warning("No ship mask available. Running detection first.")
                self.detect_ships_by_adaptive_threshold()
            mask = self.ship_masks
            
        # Label connected components in the mask
        labeled, num_ships = ndimage.label(mask)
        
        # Measure properties of each ship
        properties = measure.regionprops(labeled, intensity_image=self.image_data)
        
        ship_regions = []
        for i, prop in enumerate(properties):
            # Extract bounding box
            min_row, min_col, max_row, max_col = prop.bbox
            
            # Add padding (ensuring we stay within image boundaries)
            img_shape = self.image_data.shape
            min_row = max(0, min_row - padding)
            min_col = max(0, min_col - padding)
            max_row = min(img_shape[0], max_row + padding)
            max_col = min(img_shape[1], max_col + padding)
            
            # Extract the region
            region = self.image_data[min_row:max_row, min_col:max_col]
            
            # Create a mask for this specific ship
            ship_mask = (labeled == prop.label)
            region_mask = ship_mask[min_row:max_row, min_col:max_col]
            
            ship_data = {
                'region': region,
                'mask': region_mask,
                'bbox': (min_row, min_col, max_row, max_col),
                'centroid': prop.centroid,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'major_axis_length': prop.major_axis_length,
                'minor_axis_length': prop.minor_axis_length,
                'orientation': prop.orientation,
            }
            
            ship_regions.append(ship_data)
            
        logger.info(f"Extracted {len(ship_regions)} ship regions")
        self.detected_ships = ship_regions
        self.ship_properties = properties
        
        return ship_regions
    
    def filter_ships_by_size(self, min_area: int = 50, 
                            max_area: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Filter detected ships by size.
        
        Parameters
        ----------
        min_area : int, optional
            Minimum ship area in pixels, by default 50
        max_area : Optional[int], optional
            Maximum ship area in pixels, by default None
            
        Returns
        -------
        List[Dict[str, Any]]
            Filtered list of ship regions.
        """
        if not self.detected_ships:
            logger.warning("No ships detected yet. Running extraction first.")
            self.extract_ship_regions()
            
        filtered_ships = []
        for ship in self.detected_ships:
            area = ship['area']
            if area >= min_area and (max_area is None or area <= max_area):
                filtered_ships.append(ship)
                
        logger.info(f"Filtered from {len(self.detected_ships)} to {len(filtered_ships)} ships")
        return filtered_ships
    
    def get_ship_masks(self) -> np.ndarray:
        """
        Get the binary mask of all detected ships.
        
        Returns
        -------
        np.ndarray
            Binary mask of detected ships.
        """
        if self.ship_masks is None:
            logger.warning("No ships detected yet. Running detection first.")
            self.detect_ships_by_adaptive_threshold()
            
        return self.ship_masks
    
    def process_all(self) -> Dict[str, Any]:
        """
        Run the full processing chain.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all processing results.
        """
        # Preprocess the image
        preprocessed = self.preprocess()
        
        # Detect ships
        ship_mask = self.detect_ships_by_adaptive_threshold()
        
        # Extract ship regions
        ship_regions = self.extract_ship_regions(ship_mask)
        
        # Filter ships by size
        filtered_ships = self.filter_ships_by_size()
        
        return {
            'preprocessed': preprocessed,
            'ship_mask': ship_mask,
            'ship_regions': ship_regions,
            'filtered_ships': filtered_ships,
        } 