"""
Main processor module for ship detection and micro-motion analysis.
"""

import os
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

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
        
        # Check if the file is one of our cropped NPY files
        if self.input_file.endswith('.npy') and '_cropped_' in self.input_file:
            self.logger.info("Detected cropped NPY file, loading directly...")
            try:
                return self.load_cropped_cphd(self.input_file)
            except Exception as e:
                self.logger.error(f"Error loading cropped data: {str(e)}")
                raise
        
        # Regular SAR data handling
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
    
    def manually_select_ships(self, image_data: np.ndarray) -> Dict[str, Any]:
        """
        Allow manual selection of ships in the image.
        
        Parameters
        ----------
        image_data : np.ndarray
            Image data.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing ship regions.
        """
        self.logger.info("Starting manual ship selection...")
        
        # Scale image data for better visualization
        display_data = np.abs(image_data)
        display_data = 20 * np.log10(display_data / np.max(display_data) + 1e-10)
        display_data = np.clip(display_data, -50, 0)
        display_data = (display_data + 50) / 50
        
        # Store selected regions
        selected_regions = []
        
        # Callback function for selection
        def onselect(eclick, erelease):
            """Store the coordinates of the selected region."""
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Ensure coordinates are in bounds
            x1 = max(0, min(x1, image_data.shape[1] - 1))
            y1 = max(0, min(y1, image_data.shape[0] - 1))
            x2 = max(0, min(x2, image_data.shape[1] - 1))
            y2 = max(0, min(y2, image_data.shape[0] - 1))
            
            # Swap if needed to ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            
            # Extract region
            region = image_data[y1:y2+1, x1:x2+1]
            
            # Create mask
            mask = np.ones((y2-y1+1, x2-x1+1), dtype=bool)
            
            # Calculate center
            center_y = (y1 + y2) // 2
            center_x = (x1 + x2) // 2
            
            # Add region
            region_data = {
                'bbox': (x1, y1, x2, y2),
                'center': (center_x, center_y),
                'centroid': (center_y, center_x),  # Add centroid key (row, col format for visualization)
                'region': region,
                'mask': mask,
                'width': x2 - x1 + 1,
                'height': y2 - y1 + 1,
                'area': (x2 - x1 + 1) * (y2 - y1 + 1)
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
            self.logger.info(f"Added ship region {len(selected_regions)} at ({x1},{y1})-({x2},{y2})")
        
        # Create figure for selection
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.imshow(display_data, cmap='gray')
        plt.title('Select ships by drawing rectangles\nPress Enter when finished, Escape to cancel selection')
        plt.colorbar(label='Normalized Amplitude (dB)')
        
        # Add instructions text
        plt.figtext(0.5, 0.01, 
                    'Click and drag to select ships. Press Enter when done.', 
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
        
        self.logger.info(f"Manual selection complete. {len(selected_regions)} ships selected.")
        
        # Format results similar to automatic detection
        return {
            'filtered_ships': selected_regions,
            'ship_regions': selected_regions,
            'num_ships': len(selected_regions)
        }
    
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
    
    def crop_and_save_cphd(self, focused_image: np.ndarray, signal_data: np.ndarray, pvp_data: Dict[str, np.ndarray], metadata: Any) -> str:
        """
        Display focused image and allow user to select a region to crop.
        Then save the cropped CPHD data to a new file.
        
        Parameters
        ----------
        focused_image : np.ndarray
            Focused SAR image for visualization.
        signal_data : np.ndarray
            Original CPHD signal data.
        pvp_data : Dict[str, np.ndarray]
            Original PVP data.
        metadata : Dict[str, Any]
            Original metadata.
            
        Returns
        -------
        str
            Path to the cropped output file.
        """
        self.logger.info("Starting CPHD cropping process...")
        
        # Scale image data for better visualization
        display_data = np.abs(focused_image)
        display_data = 20 * np.log10(display_data / np.max(display_data) + 1e-10)
        display_data = np.clip(display_data, -50, 0)
        display_data = (display_data + 50) / 50
        
        # Store selected region
        selected_region = None
        
        # Callback function for selection
        def onselect(eclick, erelease):
            """Store the coordinates of the selected region."""
            nonlocal selected_region
            
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Ensure coordinates are in bounds
            x1 = max(0, min(x1, focused_image.shape[1] - 1))
            y1 = max(0, min(y1, focused_image.shape[0] - 1))
            x2 = max(0, min(x2, focused_image.shape[1] - 1))
            y2 = max(0, min(y2, focused_image.shape[0] - 1))
            
            # Swap if needed to ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
                
            # Store the selected region
            selected_region = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2 - x1 + 1, 'height': y2 - y1 + 1
            }
            
            # Draw rectangle on the plot
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # Display region information
            plt.text(x1, y1-5, f"Region: {x1},{y1} to {x2},{y2}", 
                    color='red', fontsize=10, backgroundcolor='white')
            
            plt.draw()
            self.logger.info(f"Selected region at ({x1},{y1})-({x2},{y2})")
        
        # Create figure for selection
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.imshow(display_data, cmap='gray')
        plt.title('Select region to crop from CPHD\nPress Enter when finished, Escape to cancel')
        plt.colorbar(label='Normalized Amplitude (dB)')
        
        # Add instructions text
        plt.figtext(0.5, 0.01, 
                    'Click and drag to select region. Press Enter when done.', 
                    ha='center', fontsize=12, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # Create RectangleSelector
        rect_selector = RectangleSelector(
            ax, onselect, useblit=True,
            button=[1],  # Left mouse button only
            minspanx=20, minspany=20,  # Minimum selection size
            spancoords='pixels',
            interactive=True
        )
        
        # Function to handle key press events
        def on_key_press(event):
            if event.key == 'enter':
                plt.close()
            elif event.key == 'escape':
                nonlocal selected_region
                selected_region = None
                plt.close()
        
        # Connect key press event
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        
        # Show the plot and wait for user interaction
        plt.tight_layout()
        plt.show()
        
        # Check if user selected a region
        if selected_region is None:
            self.logger.info("Cropping cancelled by user")
            return None
        
        # Get region boundaries
        x1, y1 = selected_region['x1'], selected_region['y1']
        x2, y2 = selected_region['x2'], selected_region['y2']
        
        # Crop the signal data
        # Note: The relationship between image coordinates and signal data depends on the processing chain
        # For a basic FFT-based processor, there's a direct correspondence
        cropped_signal_data = signal_data[y1:y2+1, x1:x2+1]
        
        # Filter PVP data as needed (depends on the exact format)
        # This is a placeholder - actual implementation depends on PVP data format
        cropped_pvp_data = {}
        for key, value in pvp_data.items():
            if isinstance(value, np.ndarray):
                if value.shape[0] == signal_data.shape[0]:  # If PVP data is per row
                    cropped_pvp_data[key] = value[y1:y2+1]
                else:
                    # Just copy if we don't know how to crop
                    cropped_pvp_data[key] = value
            else:
                cropped_pvp_data[key] = value
        
        # Update metadata for the cropped region
        # Simply preserve the original metadata since we can't safely modify CPHDType objects
        cropped_metadata = metadata  # Don't attempt to modify, just store reference
        
        # Create a simple metadata dictionary with crop dimensions for reference
        crop_info = {
            'original_shape': signal_data.shape,
            'cropped_shape': (y2-y1+1, x2-x1+1),
            'crop_region': {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'width': x2-x1+1, 'height': y2-y1+1
            }
        }
        
        # Generate output filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        output_filename = f"{base_name}_cropped_{timestamp}.npy"
        output_path = os.path.join(self.output_dir, output_filename)
        
        # Save the cropped data
        # This is a simplified approach using NumPy's save functionality
        # In a production environment, you'd want to save in the original format
        data_to_save = {
            'signal_data': cropped_signal_data,
            'pvp_data': cropped_pvp_data,
            'crop_info': crop_info,
            'crop_region': selected_region
        }
        
        # Convert the metadata to a string representation if possible
        try:
            if hasattr(metadata, '__dict__'):
                data_to_save['metadata_dict'] = metadata.__dict__
            elif hasattr(metadata, 'to_dict'):
                data_to_save['metadata_dict'] = metadata.to_dict()
            else:
                data_to_save['metadata_str'] = str(metadata)
        except Exception as e:
            self.logger.warning(f"Could not serialize metadata: {str(e)}")
            
        np.save(output_path, data_to_save)
        
        self.logger.info(f"Cropped CPHD data saved to {output_path}")
        
        # Also save a preview image
        preview_path = os.path.join(self.output_dir, f"{base_name}_cropped_preview_{timestamp}.png")
        cropped_display = display_data[y1:y2+1, x1:x2+1]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cropped_display, cmap='gray')
        plt.colorbar(label='Normalized Amplitude (dB)')
        plt.title(f"Cropped Region ({x1},{y1}) to ({x2},{y2})")
        plt.tight_layout()
        plt.savefig(preview_path)
        plt.close()
        
        self.logger.info(f"Preview image saved to {preview_path}")
        
        return output_path

    def load_cropped_cphd(self, cropped_file: str) -> Dict[str, Any]:
        """
        Load cropped CPHD data from a file.
        
        Parameters
        ----------
        cropped_file : str
            Path to the cropped data file.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing the loaded data.
        """
        self.logger.info(f"Loading cropped CPHD data from {cropped_file}")
        
        # Load the data
        try:
            data = np.load(cropped_file, allow_pickle=True).item()
            
            # Extract components
            signal_data = data['signal_data']
            pvp_data = data['pvp_data']
            crop_region = data['crop_region']
            crop_info = data.get('crop_info', {})
            
            # For metadata, use what we have available
            metadata = None
            if 'metadata_dict' in data:
                metadata = data['metadata_dict']
            elif 'metadata_str' in data:
                metadata = data['metadata_str']
            
            # Create focused image through basic focusing
            focused_image = np.fft.fftshift(np.fft.fft2(signal_data))
            
            self.logger.info(f"Loaded cropped CPHD data with shape {signal_data.shape}")
            self.logger.info(f"Crop region: {crop_info.get('crop_region', crop_region)}")
            
            return {
                'type': 'cphd',
                'metadata': metadata,
                'signal_data': signal_data,
                'pvp_data': pvp_data,
                'focused_image': focused_image,
                'crop_region': crop_region,
                'crop_info': crop_info
            }
            
        except Exception as e:
            self.logger.error(f"Error loading cropped CPHD data: {str(e)}")
            raise

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
        
        # If the file is not already a cropped file, ask if we should load one
        if not (self.input_file.endswith('.npy') and '_cropped_' in self.input_file):
            # Check if we should load a cropped file instead
            use_cropped = input("Do you want to load a previously cropped file? (y/n): ").lower().strip() == 'y'
            if use_cropped:
                cropped_file = input("Enter the path to the cropped file: ")
                try:
                    read_results = self.load_cropped_cphd(cropped_file)
                except Exception as e:
                    self.logger.error(f"Failed to load cropped file: {str(e)}")
                    print(f"Failed to load cropped file: {str(e)}")
                    return {
                        'status': 'error',
                        'message': f'Failed to load cropped file: {str(e)}'
                    }
        
        # Check data type
        if read_results['type'] == 'cphd':
            # For CPHD data, we need to detect ships on the focused image and analyze vibrations
            focused_image = read_results['focused_image']
            signal_data = read_results['signal_data']
            pvp_data = read_results['pvp_data']
            metadata = read_results['metadata']
            
            # Ask if the user wants to crop the data first
            crop_first = input("Do you want to crop the CPHD data first? (y/n): ").lower().strip() == 'y'
            
            if crop_first:
                cropped_file = self.crop_and_save_cphd(
                    focused_image, signal_data, pvp_data, metadata
                )
                if cropped_file:
                    self.logger.info(f"Cropped data saved to {cropped_file}")
                    print(f"Cropped data saved to {cropped_file}. Use this file for further processing.")
                    return {
                        'status': 'success',
                        'message': 'CPHD data cropped and saved',
                        'cropped_file': cropped_file
                    }
            
            # Ask user whether to use automatic or manual detection
            use_manual = input("Use manual ship selection? (y/n): ").lower().strip() == 'y'
            
            # Detect ships (either automatically or manually)
            if use_manual:
                self.detection_results = self.manually_select_ships(focused_image)
            else:
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
            
            # Ask user whether to use automatic or manual detection
            use_manual = input("Use manual ship selection? (y/n): ").lower().strip() == 'y'
            
            # Detect ships (either automatically or manually)
            if use_manual:
                self.detection_results = self.manually_select_ships(image_data)
            else:
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