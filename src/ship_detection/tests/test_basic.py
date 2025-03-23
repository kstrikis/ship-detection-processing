"""
Basic tests for the ship detection modules.
"""

import os
import unittest
import numpy as np

from ship_detection.processing.ship_detector import ShipDetector
from ship_detection.utils.helpers import enhance_image_contrast, setup_logging

class TestBasicFunctionality(unittest.TestCase):
    """
    Basic tests for the ship detection modules.
    """
    
    def setUp(self):
        """
        Set up test case.
        """
        # Create a synthetic SAR image with a ship-like target
        image_size = 512
        self.test_image = np.zeros((image_size, image_size), dtype=np.float32)
        
        # Add noise
        np.random.seed(42)  # For reproducibility
        self.test_image += 0.1 * np.random.randn(image_size, image_size)
        
        # Add a ship-like target
        ship_size = (40, 20)
        ship_center = (image_size // 2, image_size // 2)
        
        ship_row_start = ship_center[0] - ship_size[0] // 2
        ship_row_end = ship_center[0] + ship_size[0] // 2
        ship_col_start = ship_center[1] - ship_size[1] // 2
        ship_col_end = ship_center[1] + ship_size[1] // 2
        
        self.test_image[ship_row_start:ship_row_end, ship_col_start:ship_col_end] = 1.0
        
        # Add a smaller target
        small_size = (10, 5)
        small_center = (image_size // 4, image_size // 4)
        
        small_row_start = small_center[0] - small_size[0] // 2
        small_row_end = small_center[0] + small_size[0] // 2
        small_col_start = small_center[1] - small_size[1] // 2
        small_col_end = small_center[1] + small_size[1] // 2
        
        self.test_image[small_row_start:small_row_end, small_col_start:small_col_end] = 0.8
        
        # Set up logger
        self.logger = setup_logging()
    
    def test_ship_detector(self):
        """
        Test the ShipDetector class.
        """
        # Initialize ship detector
        detector = ShipDetector(self.test_image)
        
        # Run detection pipeline
        detection_results = detector.process_all()
        
        # Check that we have detection results
        self.assertIsNotNone(detection_results, "Detection results should not be None")
        
        # Check that we have a ship mask
        self.assertIn('ship_mask', detection_results, "Detection results should contain a ship mask")
        
        # Check that we have ship regions
        self.assertIn('ship_regions', detection_results, "Detection results should contain ship regions")
        
        # Check filtered ships
        self.assertIn('filtered_ships', detection_results, "Detection results should contain filtered ships")
        
        # Check that we detected at least one ship
        self.assertGreater(len(detection_results['filtered_ships']), 0, 
                         "Should detect at least one ship")
    
    def test_preprocess(self):
        """
        Test the preprocess method.
        """
        # Initialize ship detector
        detector = ShipDetector(self.test_image)
        
        # Run preprocessing
        preprocessed = detector.preprocess()
        
        # Check that result is not None
        self.assertIsNotNone(preprocessed, "Preprocessed image should not be None")
        
        # Check shape
        self.assertEqual(preprocessed.shape, self.test_image.shape,
                       "Preprocessed image should have the same shape as input")
        
        # Check that values are in [0, 1]
        self.assertGreaterEqual(np.min(preprocessed), 0, 
                             "Preprocessed image values should be >= 0")
        self.assertLessEqual(np.max(preprocessed), 1, 
                           "Preprocessed image values should be <= 1")
    
    def test_enhance_contrast(self):
        """
        Test the enhance_image_contrast function.
        """
        # Enhance contrast
        enhanced = enhance_image_contrast(self.test_image)
        
        # Check that result is not None
        self.assertIsNotNone(enhanced, "Enhanced image should not be None")
        
        # Check shape
        self.assertEqual(enhanced.shape, self.test_image.shape,
                       "Enhanced image should have the same shape as input")
        
        # Check that values are in [0, 1]
        self.assertGreaterEqual(np.min(enhanced), 0, 
                             "Enhanced image values should be >= 0")
        self.assertLessEqual(np.max(enhanced), 1, 
                           "Enhanced image values should be <= 1")

if __name__ == '__main__':
    unittest.main() 