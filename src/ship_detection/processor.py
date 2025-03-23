"""
Main processor module for ship detection and micro-motion analysis.
"""

import os
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt

from src.ship_detection.io.readers import SARDataReader
from src.ship_detection.processing.ship_detector import ShipDetector
from src.ship_detection.processing.doppler_subaperture import DopplerSubapertureProcessor
from src.ship_detection.visualization.heatmaps import VibrationHeatmapVisualizer
from src.ship_detection.utils.helpers import setup_logging, save_results

logger = logging.getLogger(__name__)

class ShipDetectionProcessor:
    """
    Main processor for ship detection and micro-motion analysis.
    """
    
    def __init__(self, 
                input_file: str, 
                output_dir: str = "results",
                log_file: Optional[str] = None):
        """
        Initialize the processor.
        
        Parameters
        ----------
        input_file : str
            Path to the input SAR data file.
        output_dir : str, optional
            Directory to save results, by default "results"
        log_file : Optional[str], optional
            Path to log file, by default None
        """
        # Setup logging
        self.logger = setup_logging(log_file)
        self.logger.info(f"Initializing processor for file: {input_file}")
        
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Components
        self.reader = None
        self.ship_detector = None
        self.vibration_processor = None
        self.visualizer = None
        
        # Results
        self.detection_results = None
        self.vibration_results = None
        self.visualization_results = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def read_data(self) -> Dict[str, Any]:
        """
        Read SAR data from the input file.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing read results.
        """
        self.logger.info("Reading SAR data...")
        
        # Initialize the reader
        self.reader = SARDataReader(self.input_file)
        
        # For CPHD data
        if hasattr(self.reader.reader, 'cphd_meta'):
            self.logger.info("Processing CPHD data")
            
            # Read signal data
            signal_data = self.reader.read_cphd_signal_data()
            
            # Read PVP data
            pvp_data = self.reader.read_pvp_data()
            
            # Convert signal data to complex image through basic focusing
            # (This is a simplified approach, real focusing would be more complex)
            focused_image = np.fft.fftshift(np.fft.fft2(signal_data))
            
            return {
                'type': 'cphd',
                'metadata': self.reader.get_metadata(),
                'signal_data': signal_data,
                'pvp_data': pvp_data,
                'focused_image': focused_image
            }
            
        # For SICD or similar complex data
        elif hasattr(self.reader.reader, 'sicd_meta'):
            self.logger.info("Processing SICD or similar complex data")
            
            # Read complex image data
            image_data = self.reader.read_sicd_data()
            
            return {
                'type': 'sicd',
                'metadata': self.reader.get_metadata(),
                'image_data': image_data
            }
            
        else:
            self.logger.warning("Unsupported data type")
            return {
                'type': 'unsupported',
                'metadata': self.reader.get_metadata()
            }
    
    def detect_ships(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Detect ships in the image.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detection results.
        """
        self.logger.info("Detecting ships...")
        
        # Initialize ship detector
        self.ship_detector = ShipDetector(image_data)
        
        # Run detection pipeline
        detection_results = self.ship_detector.process_all()
        
        num_ships = len(detection_results['filtered_ships'])
        self.logger.info(f"Detected {num_ships} ships")
        
        return detection_results
    
    def analyze_vibrations(
        self, 
        signal_data: np.ndarray, 
        pvp_data: Dict[str, np.ndarray],
        num_subapertures: int = 200
    ) -> Dict[str, Any]:
        """
        Analyze micro-motion vibrations using Doppler sub-aperture processing.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Signal data.
        pvp_data : Dict[str, np.ndarray]
            PVP data.
        num_subapertures : int, optional
            Number of sub-apertures to create, by default 200
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing vibration analysis results.
        """
        self.logger.info(f"Analyzing vibrations using {num_subapertures} sub-apertures...")
        
        # Initialize vibration processor
        self.vibration_processor = DopplerSubapertureProcessor(
            signal_data, pvp_data, num_subapertures)
        
        # Run vibration analysis
        vibration_results = self.vibration_processor.process_all()
        
        self.logger.info("Vibration analysis complete")
        
        return vibration_results
    
    def create_visualizations(
        self, 
        image_data: np.ndarray, 
        ship_regions: List[Dict[str, Any]],
        vibration_data: Dict[str, Any]
    ) -> Dict[str, plt.Figure]:
        """
        Create visualizations.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data.
        ship_regions : List[Dict[str, Any]]
            List of ship regions.
        vibration_data : Dict[str, Any]
            Vibration data.
            
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary of visualization figures.
        """
        self.logger.info("Creating visualizations...")
        
        # Initialize visualizer
        self.visualizer = VibrationHeatmapVisualizer(
            image_data, ship_regions, vibration_data)
        
        # Create figures
        figures = {}
        
        # Ship detection results
        figures['ship_detection'] = self.visualizer.plot_ship_detection_results()
        
        # Process each ship
        for i, ship in enumerate(ship_regions):
            # Vibration heatmap
            try:
                fig, _ = self.visualizer.create_vibration_heatmap(i)
                figures[f'ship_{i}_heatmap'] = fig
            except Exception as e:
                self.logger.error(f"Error creating heatmap for ship {i}: {str(e)}")
            
            # Vibration spectra
            try:
                fig = self.visualizer.plot_vibration_spectra(i)
                figures[f'ship_{i}_spectra'] = fig
            except Exception as e:
                self.logger.error(f"Error creating spectra for ship {i}: {str(e)}")
            
            # Combined visualization
            try:
                fig = self.visualizer.create_combined_visualization(i)
                figures[f'ship_{i}_combined'] = fig
            except Exception as e:
                self.logger.error(f"Error creating combined visualization for ship {i}: {str(e)}")
        
        self.logger.info(f"Created {len(figures)} visualization figures")
        
        return figures
    
    def process(self) -> Dict[str, Any]:
        """
        Run the complete processing pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all processing results.
        """
        self.logger.info("Starting processing pipeline...")
        
        # Read data
        read_results = self.read_data()
        
        # Check data type
        if read_results['type'] == 'cphd':
            # For CPHD data, we need to detect ships on the focused image and analyze vibrations
            focused_image = read_results['focused_image']
            signal_data = read_results['signal_data']
            pvp_data = read_results['pvp_data']
            
            # Detect ships
            self.detection_results = self.detect_ships(focused_image)
            
            # Analyze vibrations
            self.vibration_results = self.analyze_vibrations(signal_data, pvp_data)
            
            # Create visualizations
            self.visualization_results = self.create_visualizations(
                focused_image, 
                self.detection_results['filtered_ships'],
                self.vibration_results
            )
            
        elif read_results['type'] == 'sicd':
            # For SICD data, we can only detect ships since we need raw signal data for vibration analysis
            image_data = read_results['image_data']
            
            # Detect ships
            self.detection_results = self.detect_ships(image_data)
            
            # Skip vibration analysis
            self.logger.warning("Skipping vibration analysis - requires CPHD data")
            self.vibration_results = None
            
            # Create limited visualizations (only ship detection)
            figures = {}
            
            # Initialize visualizer with dummy vibration data
            self.visualizer = VibrationHeatmapVisualizer(
                image_data, 
                self.detection_results['filtered_ships'],
                {'vibration_params': {}}  # Dummy data
            )
            
            # Ship detection results
            figures['ship_detection'] = self.visualizer.plot_ship_detection_results()
            
            self.visualization_results = figures
            
        else:
            self.logger.error("Unsupported data type for processing")
            return {
                'status': 'error',
                'message': 'Unsupported data type'
            }
        
        # Generate timestamp for results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results
        saved_files = []
        if self.visualization_results:
            base_filename = f"{os.path.splitext(os.path.basename(self.input_file))[0]}_{timestamp}"
            
            # Prepare data to save
            data_to_save = {}
            
            if self.detection_results:
                # Remove large objects that can't be easily serialized
                serializable_detection = self.detection_results.copy()
                if 'ship_regions' in serializable_detection:
                    # Keep only metadata, not the image data
                    simplified_regions = []
                    for region in serializable_detection['ship_regions']:
                        simplified_region = {k: v for k, v in region.items() 
                                           if k not in ['region', 'mask']}
                        simplified_regions.append(simplified_region)
                    serializable_detection['ship_regions'] = simplified_regions
                
                data_to_save['detection'] = serializable_detection
            
            if self.vibration_results:
                # Only keep selected vibration parameters
                vibration_params = self.vibration_results['vibration_params']
                simplified_vibration = {
                    'frequencies': vibration_params['frequencies'],
                    'range_spectrum': vibration_params['range_spectrum'],
                    'azimuth_spectrum': vibration_params['azimuth_spectrum'],
                    'dominant_freq_range': vibration_params['dominant_freq_range'],
                    'dominant_freq_azimuth': vibration_params['dominant_freq_azimuth'],
                }
                data_to_save['vibration'] = simplified_vibration
            
            # Save results
            saved_files = save_results(
                self.output_dir,
                base_filename,
                self.visualization_results,
                data_to_save
            )
            
            self.logger.info(f"Saved {len(saved_files)} result files to {self.output_dir}")
        
        # Close reader
        if self.reader:
            self.reader.close()
        
        self.logger.info("Processing complete")
        
        return {
            'status': 'success',
            'read_results': read_results,
            'detection_results': self.detection_results,
            'vibration_results': self.vibration_results,
            'visualization_results': self.visualization_results,
            'saved_files': saved_files
        } 