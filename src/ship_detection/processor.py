"""
Main processor module for ship detection and micro-motion analysis.
Enhanced with pixel-level phase history analysis and component segmentation.
"""

import os
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy import signal, ndimage

from src.ship_detection.io.readers import SARDataReader
from src.ship_detection.processing.ship_detector import ShipDetector
from src.ship_detection.processing.doppler_subaperture import DopplerSubapertureProcessor
from src.ship_detection.visualization.heatmaps import VibrationHeatmapVisualizer
from src.ship_detection.utils.helpers import setup_logging, save_results

logger = logging.getLogger(__name__)

class SARImagePreprocessor:
    """Handles SAR image preprocessing including calibration and enhancement."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def scale_for_display(self, image_data: np.ndarray) -> np.ndarray:
        """
        Scale complex image data for better visualization.
        
        Parameters
        ----------
        image_data : np.ndarray
            Complex image data.
            
        Returns
        -------
        np.ndarray
            Scaled image data suitable for display.
        """
        display_data = np.abs(image_data)
        display_data = 20 * np.log10(display_data / np.max(display_data) + 1e-10)
        display_data = np.clip(display_data, -50, 0)
        display_data = (display_data + 50) / 50
        return display_data
    
    def apply_speckle_filter(self, image_data: np.ndarray, filter_size: int = 5) -> np.ndarray:
        """
        Apply speckle reduction filter while preserving edges.
        
        Parameters
        ----------
        image_data : np.ndarray
            SAR image data.
        filter_size : int, optional
            Size of filter kernel, by default 5
            
        Returns
        -------
        np.ndarray
            Filtered image.
        """
        self.logger.info(f"Applying speckle filter with kernel size {filter_size}")
        # Lee filter implementation
        mean = ndimage.uniform_filter(np.abs(image_data), filter_size)
        mean_sqr = ndimage.uniform_filter(np.abs(image_data)**2, filter_size)
        var = mean_sqr - mean**2
        
        # Compute weights
        noise_var = np.mean(var)
        weights = var / (var + noise_var)
        
        # Apply filter
        filtered = mean + weights * (np.abs(image_data) - mean)
        
        # Preserve phase information
        phase = np.angle(image_data)
        filtered_complex = filtered * np.exp(1j * phase)
        
        return filtered_complex


class PixelPhaseHistoryExtractor:
    """Extracts pixel-specific phase history from SAR subapertures."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.subapertures = None
        self.sampling_freq = None
        
    def create_subapertures(self, signal_data: np.ndarray, num_subapertures: int = 200) -> np.ndarray:
        """
        Create subapertures from signal data using FFT-based approach.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw signal data.
        num_subapertures : int, optional
            Number of subapertures to create, by default 200
            
        Returns
        -------
        np.ndarray
            Array of subaperture images.
        """
        self.logger.info(f"Creating {num_subapertures} subapertures")
        
        # Get signal dimensions
        rows, cols = signal_data.shape
        
        # Create subapertures array
        subapertures = np.zeros((num_subapertures, rows, cols), dtype=complex)
        
        # Calculate subaperture spacing and width
        aperture_width = cols // 2  # Use half of the full aperture
        step_size = (cols - aperture_width) // (num_subapertures - 1) if num_subapertures > 1 else 1
        
        # Generate subapertures
        for i in range(num_subapertures):
            start_col = i * step_size
            end_col = start_col + aperture_width
            
            if end_col > cols:
                break
                
            # Extract subaperture
            subaperture = np.zeros((rows, cols), dtype=complex)
            subaperture[:, start_col:end_col] = signal_data[:, start_col:end_col]
            
            # Focus subaperture
            subapertures[i] = np.fft.fftshift(np.fft.fft2(subaperture))
        
        self.subapertures = subapertures
        self.sampling_freq = 1.0  # Normalized frequency, can be adjusted based on PVP data
        
        return subapertures
    
    def extract_pixel_phase_history(self, pixel_row: int, pixel_col: int) -> np.ndarray:
        """
        Extract phase history for specific pixel across all subapertures.
        
        Parameters
        ----------
        pixel_row : int
            Pixel row coordinate.
        pixel_col : int
            Pixel column coordinate.
            
        Returns
        -------
        np.ndarray
            Time series of phase values.
        """
        if self.subapertures is None:
            raise ValueError("Subapertures not created. Call create_subapertures first.")
        
        # Extract complex values for this pixel
        complex_values = self.subapertures[:, pixel_row, pixel_col]
        
        # Extract phase information
        phase_history = np.angle(complex_values)
        
        # Unwrap phase to avoid jumps
        phase_history = np.unwrap(phase_history)
        
        return phase_history
    
    def extract_ship_phase_histories(self, ship_region: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Extract phase histories for all pixels in ship region.
        
        Parameters
        ----------
        ship_region : Dict[str, Any]
            Ship region dictionary.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with phase history arrays.
        """
        # Get ship location
        x1, y1, x2, y2 = ship_region['bbox']
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        
        # Create phase history matrix
        phase_histories = np.zeros((height, width, len(self.subapertures)))
        
        # Extract phase history for each pixel
        for i in range(height):
            for j in range(width):
                row, col = y1 + i, x1 + j
                phase_histories[i, j, :] = self.extract_pixel_phase_history(row, col)
        
        return {
            'phase_histories': phase_histories,
            'ship_bbox': ship_region['bbox'],
            'dimensions': (height, width)
        }


class TimeFrequencyAnalyzer:
    """Performs advanced time-frequency analysis on phase histories."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_single_pixel(self, phase_history: np.ndarray, sampling_freq: float) -> Dict[str, Any]:
        """
        Apply multiple time-frequency analysis methods to single pixel.
        
        Parameters
        ----------
        phase_history : np.ndarray
            Phase history time series.
        sampling_freq : float
            Sampling frequency.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with analysis results.
        """
        results = {}
        
        # Standard FFT with windowing
        windowed_signal = phase_history * np.hanning(len(phase_history))
        fft_result = np.abs(np.fft.fft(windowed_signal))
        fft_freqs = np.fft.fftfreq(len(phase_history), 1/sampling_freq)
        
        # Get positive frequencies only
        pos_mask = fft_freqs >= 0
        results['fft'] = {
            'freqs': fft_freqs[pos_mask],
            'spectrum': fft_result[pos_mask]
        }
        
        # Short-Time Fourier Transform
        window_size = min(len(phase_history) // 4, 128)
        window_size = max(window_size, 16)  # Ensure minimum size
        
        # Only compute STFT if we have enough data points
        if len(phase_history) >= window_size * 2:
            f, t, stft_result = signal.stft(
                phase_history, sampling_freq, 
                nperseg=window_size, noverlap=window_size//2
            )
            results['stft'] = {
                'freqs': f,
                'times': t,
                'coefficients': np.abs(stft_result)
            }
        
        # Find dominant frequencies (peaks in spectrum)
        spectrum = results['fft']['spectrum']
        freqs = results['fft']['freqs']
        
        # Exclude DC component
        if len(freqs) > 1:
            start_idx = 1  # Skip DC component
            peaks, _ = signal.find_peaks(spectrum[start_idx:], height=0.1*np.max(spectrum[start_idx:]))
            
            # Adjust peak indices to account for start_idx offset
            peak_indices = peaks + start_idx
            peak_freqs = freqs[peak_indices]
            peak_amps = spectrum[peak_indices]
            
            # Sort by amplitude
            sort_idx = np.argsort(peak_amps)[::-1]  # Descending order
            results['dominant_frequencies'] = {
                'frequencies': peak_freqs[sort_idx],
                'amplitudes': peak_amps[sort_idx]
            }
        else:
            results['dominant_frequencies'] = {
                'frequencies': np.array([]),
                'amplitudes': np.array([])
            }
        
        return results
    
    def analyze_ship_region(self, phase_histories: np.ndarray, sampling_freq: float) -> Dict[str, Any]:
        """
        Analyze phase histories for all pixels in a ship region.
        
        Parameters
        ----------
        phase_histories : np.ndarray
            3D array of phase histories (height x width x time).
        sampling_freq : float
            Sampling frequency.
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with analysis results.
        """
        height, width, _ = phase_histories.shape
        
        # Create result containers
        dominant_freqs = np.zeros((height, width))
        dominant_amps = np.zeros((height, width))
        all_spectra = np.zeros((height, width, phase_histories.shape[2]//2 + 1))
        
        # Process each pixel
        for i in range(height):
            for j in range(width):
                phase_history = phase_histories[i, j, :]
                analysis = self.analyze_single_pixel(phase_history, sampling_freq)
                
                # Store dominant frequency (if any)
                dom_freqs = analysis['dominant_frequencies']['frequencies']
                dom_amps = analysis['dominant_frequencies']['amplitudes']
                
                if len(dom_freqs) > 0:
                    dominant_freqs[i, j] = dom_freqs[0]
                    dominant_amps[i, j] = dom_amps[0]
                
                # Store full spectrum
                spectrum_len = min(len(analysis['fft']['spectrum']), all_spectra.shape[2])
                all_spectra[i, j, :spectrum_len] = analysis['fft']['spectrum'][:spectrum_len]
        
        return {
            'dominant_frequencies': dominant_freqs,
            'dominant_amplitudes': dominant_amps,
            'all_spectra': all_spectra,
            'frequency_axis': analysis['fft']['freqs'][:all_spectra.shape[2]]
        }


class ShipComponentClassifier:
    """Classifies ship pixels into different structural components."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def classify_components(self, ship_region: np.ndarray, intensity_mask: np.ndarray = None) -> np.ndarray:
        """
        Segment a ship into components based on image features.
        
        Parameters
        ----------
        ship_region : np.ndarray
            Ship region image data.
        intensity_mask : np.ndarray, optional
            Binary mask of ship pixels, by default None
            
        Returns
        -------
        np.ndarray
            Component map with component IDs.
        """
        if intensity_mask is None:
            # Create a simple mask if none provided
            intensity_mask = np.abs(ship_region) > np.mean(np.abs(ship_region))
        
        # Create empty component map
        component_map = np.zeros_like(intensity_mask, dtype=int)
        
        # Extract features for classification
        gradient_x = ndimage.sobel(np.abs(ship_region), axis=1)
        gradient_y = ndimage.sobel(np.abs(ship_region), axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Get ship dimensions
        rows, cols = ship_region.shape
        
        # Define regions (simplified approach)
        # 1: Hull, 2: Deck, 3: Superstructure, 4: Bow, 5: Stern
        
        # Bow and stern (front and back quarters)
        left_quarter = int(cols * 0.25)
        right_quarter = int(cols * 0.75)
        
        # Check gradient patterns to determine which end is bow
        left_gradients = np.sum(gradient_magnitude[:, :left_quarter])
        right_gradients = np.sum(gradient_magnitude[:, right_quarter:])
        
        # Initialize different component areas
        if left_gradients > right_gradients:
            # Bow is on the left
            component_map[:, :left_quarter][intensity_mask[:, :left_quarter]] = 4  # Bow
            component_map[:, right_quarter:][intensity_mask[:, right_quarter:]] = 5  # Stern
        else:
            # Bow is on the right
            component_map[:, right_quarter:][intensity_mask[:, right_quarter:]] = 4  # Bow
            component_map[:, :left_quarter][intensity_mask[:, :left_quarter]] = 5  # Stern
        
        # Central hull area
        middle_section = intensity_mask.copy()
        middle_section[:, :left_quarter] = False
        middle_section[:, right_quarter:] = False
        
        # Split into hull (bottom), deck (middle) and superstructure (top)
        top_third = int(rows * 0.33)
        bottom_third = int(rows * 0.67)
        
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
        
        return component_map


class PhysicsConstrainedAnalyzer:
    """Applies physical constraints to vibration analysis."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Define expected frequency ranges for different components
        self.component_freq_ranges = {
            1: (5, 15),    # Hull: 5-15 Hz
            2: (8, 20),    # Deck: 8-20 Hz
            3: (10, 30),   # Superstructure: 10-30 Hz
            4: (12, 25),   # Bow: 12-25 Hz
            5: (8, 18)     # Stern (engine area): 8-18 Hz
        }
    
    def apply_physical_constraints(
        self, 
        component_map: np.ndarray, 
        vibration_frequencies: np.ndarray,
        vibration_amplitudes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply physical constraints to detected vibrations based on ship components.
        
        Parameters
        ----------
        component_map : np.ndarray
            Map of ship components.
        vibration_frequencies : np.ndarray
            Detected vibration frequencies.
        vibration_amplitudes : np.ndarray
            Detected vibration amplitudes.
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Constrained frequencies and amplitudes.
        """
        self.logger.info("Applying physical constraints to vibration analysis")
        
        # Create copies of input arrays
        constrained_freqs = vibration_frequencies.copy()
        constrained_amps = vibration_amplitudes.copy()
        
        # Apply component-specific constraints
        for component_id, freq_range in self.component_freq_ranges.items():
            # Create mask for this component
            component_mask = component_map == component_id
            
            if not np.any(component_mask):
                continue
                
            min_freq, max_freq = freq_range
            self.logger.debug(f"Processing component {component_id} with freq range {min_freq}-{max_freq} Hz")
            
            # For each pixel in this component
            invalid_mask = (
                (vibration_frequencies < min_freq) | 
                (vibration_frequencies > max_freq)
            ) & component_mask
            
            # Zero out frequencies outside expected range
            constrained_freqs[invalid_mask] = 0
            constrained_amps[invalid_mask] = 0
            
            # For pixels with no valid frequencies, find nearest valid frequency in component
            zero_mask = (constrained_amps == 0) & component_mask
            if np.any(zero_mask):
                # Calculate mean of valid frequencies for this component
                valid_freqs = constrained_freqs[component_mask & ~zero_mask]
                if len(valid_freqs) > 0:
                    mean_freq = np.mean(valid_freqs)
                    # Assign mean frequency to zeros
                    constrained_freqs[zero_mask] = mean_freq
                    # Assign small amplitude
                    constrained_amps[zero_mask] = 0.1 * np.mean(constrained_amps[constrained_amps > 0])
        
        return constrained_freqs, constrained_amps


class MicroMotionVisualizer:
    """Creates enhanced visualizations for micromotion analysis."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_component_overlay(
        self, 
        ship_image: np.ndarray, 
        component_map: np.ndarray
    ) -> plt.Figure:
        """
        Create visualization of ship components.
        
        Parameters
        ----------
        ship_image : np.ndarray
            Ship region image.
        component_map : np.ndarray
            Component classification map.
            
        Returns
        -------
        plt.Figure
            Matplotlib figure.
        """
        # Component names for legend
        component_names = {
            1: 'Hull',
            2: 'Deck',
            3: 'Superstructure',
            4: 'Bow',
            5: 'Stern'
        }
        
        # Create colormap for components
        cmap = plt.cm.get_cmap('tab10', 6)  # 6 colors (0-5)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display ship image in grayscale
        ship_display = np.abs(ship_image)
        ship_display = ship_display / np.max(ship_display)
        ax.imshow(ship_display, cmap='gray', alpha=0.7)
        
        # Overlay components with transparency
        component_overlay = ax.imshow(
            component_map, cmap=cmap, alpha=0.5,
            vmin=0, vmax=5
        )
        
        # Create legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, color=cmap(i), alpha=0.5, label=name)
            for i, name in component_names.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title('Ship Component Classification')
        
        return fig
    
    def create_micromotion_heatmap(
        self, 
        ship_image: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        component_map: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Create heatmap visualization of micromotion.
        
        Parameters
        ----------
        ship_image : np.ndarray
            Ship region image.
        frequencies : np.ndarray
            Frequency map.
        amplitudes : np.ndarray
            Amplitude map.
        component_map : Optional[np.ndarray], optional
            Component classification map, by default None
            
        Returns
        -------
        plt.Figure
            Matplotlib figure.
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3 if component_map is not None else 2, 
                               figsize=(15 if component_map is not None else 10, 5))
        
        # Display ship image
        ship_display = np.abs(ship_image)
        ship_display = ship_display / np.max(ship_display)
        axes[0].imshow(ship_display, cmap='gray')
        axes[0].set_title('Ship Image')
        
        # Create mask for valid vibrations
        valid_mask = amplitudes > 0
        
        # Display frequency heatmap
        freq_display = np.zeros_like(frequencies)
        freq_display[valid_mask] = frequencies[valid_mask]
        
        freq_im = axes[1].imshow(freq_display, cmap='jet', 
                               vmin=np.min(freq_display[valid_mask]) if np.any(valid_mask) else 0,
                               vmax=np.max(freq_display[valid_mask]) if np.any(valid_mask) else 1)
        axes[1].set_title('Vibration Frequency (Hz)')
        plt.colorbar(freq_im, ax=axes[1])
        
        # Display component map if provided
        if component_map is not None:
            # Component names for legend
            component_names = {
                1: 'Hull',
                2: 'Deck',
                3: 'Superstructure',
                4: 'Bow',
                5: 'Stern'
            }
            
            # Create colormap for components
            cmap = plt.cm.get_cmap('tab10', 6)  # 6 colors (0-5)
            
            comp_im = axes[2].imshow(component_map, cmap=cmap, vmin=0, vmax=5)
            axes[2].set_title('Ship Components')
            
            # Create legend
            legend_elements = [
                plt.Rectangle((0, 0), 1, 1, color=cmap(i), alpha=0.8, label=name)
                for i, name in component_names.items()
            ]
            axes[2].legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_vibration_spectrogram(self, phase_history: np.ndarray, sampling_freq: float) -> plt.Figure:
        """
        Create spectrogram visualization for a single pixel.
        
        Parameters
        ----------
        phase_history : np.ndarray
            Phase history time series.
        sampling_freq : float
            Sampling frequency.
            
        Returns
        -------
        plt.Figure
            Matplotlib figure.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute spectrogram
        nperseg = min(len(phase_history) // 4, 128)
        nperseg = max(nperseg, 16)  # Ensure minimum size
        
        f, t, Sxx = signal.spectrogram(
            phase_history, 
            fs=sampling_freq, 
            nperseg=nperseg,
            noverlap=nperseg//2
        )
        
        # Plot spectrogram
        im = ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='inferno')
        fig.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)')
        
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [s]')
        ax.set_title('Vibration Spectrogram')
        
        return fig


class EnhancedShipDetectionProcessor:
    """
    Enhanced processor for ship detection and micro-motion analysis.
    Implements pixel-level phase analysis and component classification.
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
        self.logger.info(f"Initializing enhanced processor for file: {input_file}")
        
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Initialize components
        self.reader = None
        self.preprocessor = SARImagePreprocessor(self.logger)
        self.ship_detector = None
        self.phase_extractor = PixelPhaseHistoryExtractor(self.logger)
        self.time_freq_analyzer = TimeFrequencyAnalyzer(self.logger)
        self.component_classifier = ShipComponentClassifier(self.logger)
        self.physics_analyzer = PhysicsConstrainedAnalyzer(self.logger)
        self.visualizer = MicroMotionVisualizer(self.logger)
        
        # Results storage
        self.read_results = None
        self.detection_results = None
        self.phase_history_results = None
        self.vibration_results = None
        self.component_results = None
        self.physics_results = None
        self.visualization_results = None
    
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
                self.read_results = self.load_cropped_cphd(self.input_file)
                return self.read_results
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
            
            self.read_results = {
                'type': 'cphd',
                'metadata': self.reader.get_metadata(),
                'signal_data': signal_data,
                'pvp_data': pvp_data,
                'focused_image': focused_image
            }
            
            return self.read_results
            
        # For SICD or similar complex data
        elif hasattr(self.reader.reader, 'sicd_meta'):
            self.logger.info("Processing SICD or similar complex data")
            
            # Read complex image data
            image_data = self.reader.read_sicd_data()
            
            self.read_results = {
                'type': 'sicd',
                'metadata': self.reader.get_metadata(),
                'image_data': image_data
            }
            
            return self.read_results
            
        else:
            self.logger.warning("Unsupported data type")
            
            self.read_results = {
                'type': 'unsupported',
                'metadata': self.reader.get_metadata()
            }
            
            return self.read_results
    
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
        
        # Apply speckle filter for better detection
        filtered_image = self.preprocessor.apply_speckle_filter(image_data)
        
        # Initialize ship detector
        self.ship_detector = ShipDetector(filtered_image)
        
        # Run detection pipeline
        detection_results = self.ship_detector.process_all()
        
        num_ships = len(detection_results['filtered_ships'])
        self.logger.info(f"Detected {num_ships} ships")
        
        # Store results
        self.detection_results = detection_results
        
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
        display_data = self.preprocessor.scale_for_display(image_data)
        
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
        self.detection_results = {
            'filtered_ships': selected_regions,
            'ship_regions': selected_regions,
            'num_ships': len(selected_regions)
        }
        
        return self.detection_results
    
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
        display_data = self.preprocessor.scale_for_display(focused_image)
        
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
        cropped_signal_data = signal_data[y1:y2+1, x1:x2+1]
        
        # Filter PVP data as needed (depends on the exact format)
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

    def analyze_ship_micromotion(
        self, 
        signal_data: np.ndarray, 
        pvp_data: Dict[str, np.ndarray],
        ship_regions: List[Dict[str, Any]],
        num_subapertures: int = 200
    ) -> Dict[str, Any]:
        """
        Analyze ship micro-motion using pixel-specific phase history.
        
        Parameters
        ----------
        signal_data : np.ndarray
            SAR signal data.
        pvp_data : Dict[str, np.ndarray]
            PVP data.
        ship_regions : List[Dict[str, Any]]
            List of detected ship regions.
        num_subapertures : int, optional
            Number of subapertures to use, by default 200
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of micro-motion analysis results.
        """
        self.logger.info(f"Analyzing ship micro-motion with {num_subapertures} subapertures...")
        
        # Create subapertures from signal data
        subapertures = self.phase_extractor.create_subapertures(
            signal_data, num_subapertures)
        
        # Calculate sampling frequency from PVP data if available
        sampling_freq = 1.0  # Default normalized frequency
        
        if 'SIGNAL' in pvp_data and 'TxTime' in pvp_data['SIGNAL']:
            # Extract time values (assuming they are in seconds)
            time_values = pvp_data['SIGNAL']['TxTime']
            
            # Calculate sampling period and frequency
            if len(time_values) >= 2:
                sampling_period = np.mean(np.diff(time_values))
                sampling_freq = 1.0 / sampling_period
                self.logger.info(f"Calculated sampling frequency: {sampling_freq} Hz")
        
        # Store results for each ship
        ship_results = []
        
        for i, ship in enumerate(ship_regions):
            self.logger.info(f"Processing ship {i+1} of {len(ship_regions)}")
            
            # Extract phase histories for ship region
            ship_phase_data = self.phase_extractor.extract_ship_phase_histories(ship)
            
            # Get ship region image (for visualization and component classification)
            x1, y1, x2, y2 = ship['bbox']
            ship_image = subapertures[0, y1:y2+1, x1:x2+1]  # Use first subaperture
            
            # Classify ship components
            component_map = self.component_classifier.classify_components(
                ship_image, ship['mask'])
            
            # Analyze phase histories
            vibration_analysis = self.time_freq_analyzer.analyze_ship_region(
                ship_phase_data['phase_histories'], sampling_freq)
            
            # Apply physical constraints
            constrained_freqs, constrained_amps = self.physics_analyzer.apply_physical_constraints(
                component_map,
                vibration_analysis['dominant_frequencies'],
                vibration_analysis['dominant_amplitudes']
            )
            
            # Collect results for this ship
            ship_result = {
                'ship_index': i,
                'bbox': ship['bbox'],
                'phase_histories': ship_phase_data,
                'vibration_analysis': vibration_analysis,
                'component_map': component_map,
                'constrained_frequencies': constrained_freqs,
                'constrained_amplitudes': constrained_amps,
                'sampling_freq': sampling_freq
            }
            
            ship_results.append(ship_result)
        
        # Store overall results
        results = {
            'num_ships': len(ship_regions),
            'ship_results': ship_results,
            'sampling_freq': sampling_freq,
            'num_subapertures': num_subapertures,
            'subaperture_timestamps': np.arange(num_subapertures) / sampling_freq
        }
        
        self.vibration_results = results
        self.logger.info("Ship micro-motion analysis complete")
        
        return results
    
    def create_visualizations(self) -> Dict[str, plt.Figure]:
        """
        Create visualizations for detection and micro-motion results.
        
        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary of figures.
        """
        self.logger.info("Creating visualizations...")
        
        if not self.detection_results or not self.vibration_results:
            self.logger.warning("No results to visualize")
            return {}
        
        figures = {}
        
        # Get appropriate image data
        if self.read_results['type'] == 'cphd':
            image_data = self.read_results['focused_image']
        else:
            image_data = self.read_results['image_data']
            
        # Create ship detection overview
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Display the image
            display_data = self.preprocessor.scale_for_display(image_data)
            ax.imshow(display_data, cmap='gray')
            
            # Overlay ships
            ship_regions = self.detection_results['filtered_ships']
            for i, ship in enumerate(ship_regions):
                x1, y1, x2, y2 = ship['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"Ship {i+1}", 
                        color='red', fontsize=10, backgroundcolor='white')
                
            ax.set_title(f"Detected Ships ({len(ship_regions)})")
            
            figures['ship_detection'] = fig
        except Exception as e:
            self.logger.error(f"Error creating ship detection overview: {str(e)}")
        
        # Process each ship for detailed visualizations
        for i, ship_result in enumerate(self.vibration_results['ship_results']):
            ship_index = ship_result['ship_index']
            
            try:
                # Get ship region image
                x1, y1, x2, y2 = ship_result['bbox']
                ship_image = image_data[y1:y2+1, x1:x2+1]
                
                # Create component classification visualization
                component_fig = self.visualizer.create_component_overlay(
                    ship_image, ship_result['component_map']
                )
                figures[f'ship_{ship_index}_components'] = component_fig
                
                # Create micromotion heatmap
                heatmap_fig = self.visualizer.create_micromotion_heatmap(
                    ship_image,
                    ship_result['constrained_frequencies'],
                    ship_result['constrained_amplitudes'],
                    ship_result['component_map']
                )
                figures[f'ship_{ship_index}_micromotion'] = heatmap_fig
                
                # Create vibration spectrograms for selected points
                # Choose central points from each component
                component_ids = np.unique(ship_result['component_map'])
                
                for comp_id in component_ids:
                    if comp_id == 0:  # Skip background
                        continue
                        
                    # Find central point of this component
                    comp_mask = ship_result['component_map'] == comp_id
                    if not np.any(comp_mask):
                        continue
                        
                    # Calculate centroid
                    coords = np.where(comp_mask)
                    center_y = int(np.mean(coords[0]))
                    center_x = int(np.mean(coords[1]))
                    
                    # Get phase history for this point
                    phase_history = ship_result['phase_histories']['phase_histories'][center_y, center_x, :]
                    
                    # Create spectrogram
                    spec_fig = self.visualizer.create_vibration_spectrogram(
                        phase_history, ship_result['sampling_freq']
                    )
                    
                    comp_name = {1: 'hull', 2: 'deck', 3: 'superstructure', 
                               4: 'bow', 5: 'stern'}.get(comp_id, f'comp_{comp_id}')
                    
                    figures[f'ship_{ship_index}_{comp_name}_spectrogram'] = spec_fig
                
            except Exception as e:
                self.logger.error(f"Error creating visualizations for ship {ship_index}: {str(e)}")
        
        self.visualization_results = figures
        self.logger.info(f"Created {len(figures)} visualization figures")
        
        return figures

    def save_results(self) -> List[str]:
        """
        Save analysis results and visualizations.
        
        Returns
        -------
        List[str]
            Paths to saved files.
        """
        self.logger.info("Saving analysis results and visualizations...")
        
        if not self.detection_results or not self.visualization_results:
            self.logger.warning("No results to save")
            return []
        
        # Generate timestamp for results
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{os.path.splitext(os.path.basename(self.input_file))[0]}_{timestamp}"
        
        # Prepare data to save
        data_to_save = {}
        
        # Detection results
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
        
        # Vibration results
        if self.vibration_results:
            # Create a serializable version of vibration results
            serializable_vibration = {
                'num_ships': self.vibration_results['num_ships'],
                'sampling_freq': self.vibration_results['sampling_freq'],
                'num_subapertures': self.vibration_results['num_subapertures'],
                'ship_results': []
            }
            
            # Process each ship
            for ship_result in self.vibration_results['ship_results']:
                # Remove large arrays
                simplified_ship = {
                    'ship_index': ship_result['ship_index'],
                    'bbox': ship_result['bbox'],
                    'sampling_freq': ship_result['sampling_freq']
                }
                
                # Keep component map (small enough)
                simplified_ship['component_map'] = ship_result['component_map']
                
                # Keep constrained frequencies and amplitudes (important results)
                simplified_ship['constrained_frequencies'] = ship_result['constrained_frequencies']
                simplified_ship['constrained_amplitudes'] = ship_result['constrained_amplitudes']
                
                serializable_vibration['ship_results'].append(simplified_ship)
            
            data_to_save['vibration'] = serializable_vibration
        
        # Save results using helper function
        saved_files = save_results(
            self.output_dir,
            base_filename,
            self.visualization_results,
            data_to_save
        )
        
        self.logger.info(f"Saved {len(saved_files)} result files to {self.output_dir}")
        
        return saved_files
    
    def process(self) -> Dict[str, Any]:
        """
        Run the complete processing pipeline.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all processing results.
        """
        self.logger.info("Starting enhanced processing pipeline...")
        
        # Read data
        self.read_data()
        
        # If the file is not already a cropped file, ask if we should load one
        if not (self.input_file.endswith('.npy') and '_cropped_' in self.input_file):
            # Check if we should load a cropped file instead
            use_cropped = input("Do you want to load a previously cropped file? (y/n): ").lower().strip() == 'y'
            if use_cropped:
                cropped_file = input("Enter the path to the cropped file: ")
                try:
                    self.read_results = self.load_cropped_cphd(cropped_file)
                except Exception as e:
                    self.logger.error(f"Failed to load cropped file: {str(e)}")
                    print(f"Failed to load cropped file: {str(e)}")
                    return {
                        'status': 'error',
                        'message': f'Failed to load cropped file: {str(e)}'
                    }
        
        # Check data type
        if self.read_results['type'] == 'cphd':
            # For CPHD data, we need to detect ships on the focused image and analyze vibrations
            focused_image = self.read_results['focused_image']
            signal_data = self.read_results['signal_data']
            pvp_data = self.read_results['pvp_data']
            metadata = self.read_results['metadata']
            
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
                self.manually_select_ships(focused_image)
            else:
                self.detect_ships(focused_image)
            
            # Analyze micro-motion
            self.analyze_ship_micromotion(
                signal_data, 
                pvp_data, 
                self.detection_results['filtered_ships']
            )
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            saved_files = self.save_results()
            
        elif self.read_results['type'] == 'sicd':
            # For SICD data, we can only detect ships since we need raw signal data for vibration analysis
            image_data = self.read_results['image_data']
            
            # Ask user whether to use automatic or manual detection
            use_manual = input("Use manual ship selection? (y/n): ").lower().strip() == 'y'
            
            # Detect ships (either automatically or manually)
            if use_manual:
                self.manually_select_ships(image_data)
            else:
                self.detect_ships(image_data)
            
            # Skip vibration analysis
            self.logger.warning("Skipping micro-motion analysis - requires CPHD data")
            
            # Create limited visualizations (only ship detection)
            figures = {}
            
            # Display the image
            display_data = self.preprocessor.scale_for_display(image_data)
            
            # Create ship detection overview
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(display_data, cmap='gray')
            
            # Overlay ships
            ship_regions = self.detection_results['filtered_ships']
            for i, ship in enumerate(ship_regions):
                x1, y1, x2, y2 = ship['bbox']
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                    fill=False, edgecolor='red', linewidth=2)
                ax.add_patch(rect)
                ax.text(x1, y1-5, f"Ship {i+1}", 
                        color='red', fontsize=10, backgroundcolor='white')
                
            ax.set_title(f"Detected Ships ({len(ship_regions)})")
            
            figures['ship_detection'] = fig
            
            self.visualization_results = figures
            
            # Save results
            saved_files = self.save_results()
            
        else:
            self.logger.error("Unsupported data type for processing")
            return {
                'status': 'error',
                'message': 'Unsupported data type'
            }
        
        # Close reader
        if self.reader:
            self.reader.close()
        
        self.logger.info("Enhanced processing complete")
        
        return {
            'status': 'success',
            'read_results': self.read_results,
            'detection_results': self.detection_results,
            'vibration_results': self.vibration_results,
            'visualization_results': self.visualization_results,
            'saved_files': saved_files
        }


# Legacy compatibility shim
class ShipDetectionProcessor(EnhancedShipDetectionProcessor):
    """Legacy compatibility class for backward compatibility."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with a warning about legacy usage."""
        import warnings
        warnings.warn(
            "ShipDetectionProcessor is deprecated. Use EnhancedShipDetectionProcessor instead.",
            DeprecationWarning, stacklevel=2
        )
        super().__init__(*args, **kwargs)
        
        # Override analyze_vibrations with a compatibility method
        self.analyze_vibrations = self._legacy_analyze_vibrations
    
    def _legacy_analyze_vibrations(
        self, 
        signal_data: np.ndarray, 
        pvp_data: Dict[str, np.ndarray],
        num_subapertures: int = 200
    ) -> Dict[str, Any]:
        """Legacy compatibility method for backward compatibility."""
        self.logger.warning(
            "Using legacy analyze_vibrations method. Consider using analyze_ship_micromotion instead."
        )
        
        # Run the new analysis if we have detection results
        if self.detection_results and 'filtered_ships' in self.detection_results:
            return self.analyze_ship_micromotion(
                signal_data, pvp_data, self.detection_results['filtered_ships'], num_subapertures
            )
        
        # Legacy fallback using old DopplerSubapertureProcessor
        from src.ship_detection.processing.doppler_subaperture import DopplerSubapertureProcessor
        
        self.logger.info(f"Analyzing vibrations using {num_subapertures} sub-apertures...")
        
        # Initialize vibration processor
        self.vibration_processor = DopplerSubapertureProcessor(
            signal_data, pvp_data, num_subapertures)
        
        # Run vibration analysis
        vibration_results = self.vibration_processor.process_all()
        
        self.logger.info("Vibration analysis complete")
        
        return vibration_results 