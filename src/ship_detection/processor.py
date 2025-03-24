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
from concurrent.futures import ThreadPoolExecutor
# Import Dask for pipeline parallelism (optional dependency)
try:
    import dask
    from dask.delayed import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False

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
    
    def downsample_image(self, image_data: np.ndarray, factor: int = 4) -> np.ndarray:
        """
        Create a downsampled version of the image data for faster initial processing.
        
        Parameters
        ----------
        image_data : np.ndarray
            Original image data.
        factor : int, optional
            Downsampling factor, by default 4
            
        Returns
        -------
        np.ndarray
            Downsampled image.
        """
        rows, cols = image_data.shape
        
        if factor <= 1:
            return image_data
            
        # Calculate new dimensions
        new_rows = rows // factor
        new_cols = cols // factor
        
        self.logger.info(f"Downsampling image from {rows}x{cols} to {new_rows}x{new_cols}")
        
        # Create downsampled array
        downsampled = np.zeros((new_rows, new_cols), dtype=image_data.dtype)
        
        # Use block averaging for downsampling to preserve energy
        for i in range(new_rows):
            for j in range(new_cols):
                r_start = i * factor
                r_end = min((i + 1) * factor, rows)
                c_start = j * factor
                c_end = min((j + 1) * factor, cols)
                
                block = image_data[r_start:r_end, c_start:c_end]
                downsampled[i, j] = np.mean(block)
        
        return downsampled
    
    def scale_coordinates(self, 
                         coords: Tuple[int, int, int, int], 
                         original_shape: Tuple[int, int], 
                         downsampled_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Scale coordinates from downsampled image back to original image.
        
        Parameters
        ----------
        coords : Tuple[int, int, int, int]
            Coordinates in downsampled image (x1, y1, x2, y2).
        original_shape : Tuple[int, int]
            Shape of original image (rows, cols).
        downsampled_shape : Tuple[int, int]
            Shape of downsampled image (rows, cols).
            
        Returns
        -------
        Tuple[int, int, int, int]
            Scaled coordinates in original image.
        """
        orig_rows, orig_cols = original_shape
        down_rows, down_cols = downsampled_shape
        
        x1, y1, x2, y2 = coords
        
        # Calculate scaling factors
        row_scale = orig_rows / down_rows
        col_scale = orig_cols / down_cols
        
        # Scale coordinates
        x1_orig = int(x1 * col_scale)
        y1_orig = int(y1 * row_scale)
        x2_orig = int(x2 * col_scale)
        y2_orig = int(y2 * row_scale)
        
        # Ensure coordinates are within bounds
        x1_orig = max(0, min(x1_orig, orig_cols - 1))
        y1_orig = max(0, min(y1_orig, orig_rows - 1))
        x2_orig = max(0, min(x2_orig, orig_cols - 1))
        y2_orig = max(0, min(y2_orig, orig_rows - 1))
        
        return (x1_orig, y1_orig, x2_orig, y2_orig)


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
        self.logger.info(f"Creating {num_subapertures} subapertures using parallel processing")
        
        # Get signal dimensions
        rows, cols = signal_data.shape
        
        # Create subapertures array
        subapertures = np.zeros((num_subapertures, rows, cols), dtype=complex)
        
        # Calculate subaperture spacing and width
        aperture_width = cols // 2  # Use half of the full aperture
        step_size = (cols - aperture_width) // (num_subapertures - 1) if num_subapertures > 1 else 1
        
        # Process subapertures in parallel
        def process_subaperture(i):
            start_col = i * step_size
            end_col = start_col + aperture_width
            
            if end_col > cols:
                return None
                
            # Extract subaperture
            subaperture = np.zeros((rows, cols), dtype=complex)
            subaperture[:, start_col:end_col] = signal_data[:, start_col:end_col]
            
            # Focus subaperture
            return np.fft.fftshift(np.fft.fft2(subaperture))
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_subaperture, i) for i in range(num_subapertures)]
            
            for i, future in enumerate(futures):
                result = future.result()
                if result is not None:
                    subapertures[i] = result
        
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
        
        # Try to use GPU acceleration if available
        try:
            import cupy as cp
            self.logger.info("Using GPU acceleration with CuPy for phase history extraction")
            
            # Transfer subapertures to GPU
            gpu_subapertures = cp.asarray(self.subapertures)
            
            # Extract region and compute phase histories on GPU
            region_subapertures = gpu_subapertures[:, y1:y2+1, x1:x2+1]
            phase_histories = cp.angle(region_subapertures)
            
            # Unwrap phase on GPU
            phase_histories = cp.unwrap(phase_histories, axis=0)
            
            # Transfer result back to CPU
            phase_histories_cpu = cp.asnumpy(phase_histories)
            
            # Reshape to match expected format (height, width, time)
            phase_histories_cpu = np.transpose(phase_histories_cpu, (1, 2, 0))
            
            return {
                'phase_histories': phase_histories_cpu,
                'ship_bbox': ship_region['bbox'],
                'dimensions': (height, width)
            }
            
        except (ImportError, ModuleNotFoundError):
            self.logger.info("CuPy not available, using CPU implementation")
            
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
        self.logger.info("Performing vectorized FFT analysis for all pixels")
        height, width, time_samples = phase_histories.shape
        
        # Create window function once
        window = np.hanning(time_samples)
        
        # Apply windowing to all pixel phase histories at once
        # Shape: height x width x time
        windowed_signals = phase_histories * window[np.newaxis, np.newaxis, :]
        
        # Perform FFT on all pixels at once
        # Shape: height x width x time
        fft_results = np.abs(np.fft.fft(windowed_signals, axis=2))
        
        # Calculate frequency axis once
        fft_freqs = np.fft.fftfreq(time_samples, 1/sampling_freq)
        
        # Get positive frequencies only
        pos_mask = fft_freqs >= 0
        pos_freqs = fft_freqs[pos_mask]
        pos_spectra = fft_results[:, :, pos_mask]
        
        # Find dominant frequencies (peaks in spectrum) for each pixel
        # Skip the DC component (index 0)
        start_idx = 1
        dominant_freqs = np.zeros((height, width))
        dominant_amps = np.zeros((height, width))
        
        # This is still a loop, but we've reduced complexity by calculating FFTs vectorially
        for i in range(height):
            for j in range(width):
                spectrum = pos_spectra[i, j, start_idx:]
                if len(spectrum) > 0:
                    # Find peaks above threshold
                    peaks, _ = signal.find_peaks(spectrum, height=0.1*np.max(spectrum))
                    
                    if len(peaks) > 0:
                        # Adjust indices to account for start_idx offset
                        peak_indices = peaks + start_idx
                        peak_freqs = pos_freqs[peak_indices]
                        peak_amps = pos_spectra[i, j, peak_indices]
                        
                        # Find highest amplitude peak
                        max_idx = np.argmax(peak_amps)
                        dominant_freqs[i, j] = peak_freqs[max_idx]
                        dominant_amps[i, j] = peak_amps[max_idx]
        
        return {
            'dominant_frequencies': dominant_freqs,
            'dominant_amplitudes': dominant_amps,
            'all_spectra': pos_spectra,
            'frequency_axis': pos_freqs
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
        
        # Try to use GPU acceleration if available
        try:
            import cupy as cp
            self.logger.info("Using GPU acceleration with CuPy for component classification")
            
            # Transfer data to GPU
            gpu_ship_region = cp.asarray(np.abs(ship_region))
            gpu_intensity_mask = cp.asarray(intensity_mask)
            
            # Extract features for classification using GPU
            gradient_x = cp.zeros_like(gpu_ship_region)
            gradient_y = cp.zeros_like(gpu_ship_region)
            
            # Implement Sobel filter
            for i in range(1, gpu_ship_region.shape[0] - 1):
                for j in range(1, gpu_ship_region.shape[1] - 1):
                    # X gradient
                    gradient_x[i, j] = (
                        (gpu_ship_region[i-1, j+1] + 2*gpu_ship_region[i, j+1] + gpu_ship_region[i+1, j+1]) -
                        (gpu_ship_region[i-1, j-1] + 2*gpu_ship_region[i, j-1] + gpu_ship_region[i+1, j-1])
                    )
                    
                    # Y gradient
                    gradient_y[i, j] = (
                        (gpu_ship_region[i+1, j-1] + 2*gpu_ship_region[i+1, j] + gpu_ship_region[i+1, j+1]) -
                        (gpu_ship_region[i-1, j-1] + 2*gpu_ship_region[i-1, j] + gpu_ship_region[i-1, j+1])
                    )
            
            gradient_magnitude = cp.sqrt(gradient_x**2 + gradient_y**2)
            
            # Get ship dimensions
            rows, cols = gpu_ship_region.shape
            
            # Retrieve array from GPU for further processing
            gradient_magnitude_cpu = cp.asnumpy(gradient_magnitude)
            intensity_mask_cpu = cp.asnumpy(gpu_intensity_mask)
            
            # Define regions (simplified approach)
            # 1: Hull, 2: Deck, 3: Superstructure, 4: Bow, 5: Stern
            
            # Bow and stern (front and back quarters)
            left_quarter = int(cols * 0.25)
            right_quarter = int(cols * 0.75)
            
            # Check gradient patterns to determine which end is bow
            left_gradients = np.sum(gradient_magnitude_cpu[:, :left_quarter])
            right_gradients = np.sum(gradient_magnitude_cpu[:, right_quarter:])
            
            # Initialize different component areas
            if left_gradients > right_gradients:
                # Bow is on the left
                component_map[:, :left_quarter][intensity_mask_cpu[:, :left_quarter]] = 4  # Bow
                component_map[:, right_quarter:][intensity_mask_cpu[:, right_quarter:]] = 5  # Stern
            else:
                # Bow is on the right
                component_map[:, right_quarter:][intensity_mask_cpu[:, right_quarter:]] = 4  # Bow
                component_map[:, :left_quarter][intensity_mask_cpu[:, :left_quarter]] = 5  # Stern
            
            # Central hull area
            middle_section = intensity_mask_cpu.copy()
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
            
        except (ImportError, ModuleNotFoundError):
            self.logger.info("CuPy not available, using CPU implementation for component classification")
            
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
                log_file: Optional[str] = None,
                use_gpu: bool = True,
                use_parallel: bool = True,
                use_pipeline: bool = True,
                tile_processing: bool = False,
                tile_size: int = 512):
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
        use_gpu : bool, optional
            Whether to use GPU acceleration if available, by default True
        use_parallel : bool, optional
            Whether to use parallel processing, by default True
        use_pipeline : bool, optional
            Whether to use pipeline parallelism with Dask, by default True
        tile_processing : bool, optional
            Whether to use tile-based processing for large datasets, by default False
        tile_size : int, optional
            Size of tiles for tile-based processing, by default 512
        """
        # Setup logging
        self.logger = setup_logging(log_file)
        self.logger.info(f"Initializing enhanced processor for file: {input_file}")
        
        self.input_file = input_file
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Parallelization parameters
        self.use_gpu = use_gpu
        self.use_parallel = use_parallel
        self.use_pipeline = use_pipeline and DASK_AVAILABLE
        self.tile_processing = tile_processing
        self.tile_size = tile_size
        
        # Check for GPU availability
        if self.use_gpu:
            try:
                import cupy
                self.logger.info("GPU acceleration is available")
            except ImportError:
                self.logger.info("CuPy not found, GPU acceleration disabled")
                self.use_gpu = False
        
        # Check for Dask availability
        if self.use_pipeline and not DASK_AVAILABLE:
            self.logger.info("Dask not found, pipeline parallelism disabled")
            self.use_pipeline = False
        
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
        
        # Processing parameters
        self.num_subapertures = 200  # Default number of subapertures
        self.speckle_filter_size = 5  # Default speckle filter kernel size
        self.skip_constraints = False  # Whether to skip physical constraints
        self.use_manual_selection = False  # Whether to use manual ship selection
        self.crop_only = False  # Whether to only perform cropping
        self.component_analysis = True  # Whether to perform component-specific analysis
        self.apply_normalization = True  # Whether to apply image normalization
        self.preview_downsample_factor = 4  # Default downsampling factor for preview
        
        # Log memory usage
        self._log_memory_usage("initialization")
    
    def _log_memory_usage(self, stage_name: str) -> None:
        """
        Log current memory usage.
        
        Parameters
        ----------
        stage_name : str
            Name of the current processing stage.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_usage_gb = process.memory_info().rss / (1024 * 1024 * 1024)
            self.logger.info(f"Memory usage after {stage_name}: {memory_usage_gb:.2f} GB")
        except ImportError:
            self.logger.debug("psutil not installed, skipping memory usage tracking")
    
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
                
                # Create downsampled preview for ship detection
                if 'focused_image' in self.read_results:
                    orig_image = self.read_results['focused_image']
                    self.read_results['preview_image'] = self.preprocessor.downsample_image(
                        orig_image, self.preview_downsample_factor)
                    self.read_results['preview_factor'] = self.preview_downsample_factor
                    self.read_results['original_shape'] = orig_image.shape
                    self.read_results['preview_shape'] = self.read_results['preview_image'].shape
                
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
            
            # Create downsampled preview for ship detection
            preview_image = self.preprocessor.downsample_image(
                focused_image, self.preview_downsample_factor)
            
            self.read_results = {
                'type': 'cphd',
                'metadata': self.reader.get_metadata(),
                'signal_data': signal_data,
                'pvp_data': pvp_data,
                'focused_image': focused_image,
                'preview_image': preview_image,
                'preview_factor': self.preview_downsample_factor,
                'original_shape': focused_image.shape,
                'preview_shape': preview_image.shape
            }
            
            # Memory usage info for monitoring
            full_size_mb = focused_image.nbytes / (1024 * 1024)
            preview_size_mb = preview_image.nbytes / (1024 * 1024)
            self.logger.info(f"Full image size: {full_size_mb:.2f} MB, Preview size: {preview_size_mb:.2f} MB")
            self.logger.info(f"Memory reduction: {100 * (1 - preview_size_mb/full_size_mb):.1f}%")
            
            return self.read_results
            
        # For SICD or similar complex data
        elif hasattr(self.reader.reader, 'sicd_meta'):
            self.logger.info("Processing SICD or similar complex data")
            
            # Read complex image data
            image_data = self.reader.read_sicd_data()
            
            # Create downsampled preview for ship detection
            preview_image = self.preprocessor.downsample_image(
                image_data, self.preview_downsample_factor)
            
            self.read_results = {
                'type': 'sicd',
                'metadata': self.reader.get_metadata(),
                'image_data': image_data,
                'preview_image': preview_image,
                'preview_factor': self.preview_downsample_factor,
                'original_shape': image_data.shape,
                'preview_shape': preview_image.shape
            }
            
            # Memory usage info for monitoring
            full_size_mb = image_data.nbytes / (1024 * 1024)
            preview_size_mb = preview_image.nbytes / (1024 * 1024)
            self.logger.info(f"Full image size: {full_size_mb:.2f} MB, Preview size: {preview_size_mb:.2f} MB")
            self.logger.info(f"Memory reduction: {100 * (1 - preview_size_mb/full_size_mb):.1f}%")
            
            return self.read_results
            
        else:
            self.logger.warning("Unsupported data type")
            
            self.read_results = {
                'type': 'unsupported',
                'metadata': self.reader.get_metadata()
            }
            
            return self.read_results
    
    def detect_ships(self, image_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect ships in the image using downsampled preview for efficiency.
        
        Parameters
        ----------
        image_data : Optional[np.ndarray], optional
            Custom image data to use, by default None (uses preview image)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing detection results.
        """
        self.logger.info("Detecting ships using low-resolution preview...")
        
        if image_data is None and 'preview_image' in self.read_results:
            # Use the downsampled preview image by default
            image_data = self.read_results['preview_image']
            self.logger.info(f"Using preview image with shape {image_data.shape}")
        elif image_data is None and 'focused_image' in self.read_results:
            # Fallback to full resolution if preview not available
            image_data = self.read_results['focused_image']
            self.logger.warning("Preview image not found, using full resolution")
        elif image_data is None and 'image_data' in self.read_results:
            # Fallback for SICD
            image_data = self.read_results['image_data']
            self.logger.warning("Preview image not found, using full resolution")
        
        # Apply speckle filter for better detection if enabled
        if self.speckle_filter_size > 0:
            start_time = datetime.datetime.now()
            filtered_image = self.preprocessor.apply_speckle_filter(
                image_data, self.speckle_filter_size)
            end_time = datetime.datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            self.logger.info(f"Applied speckle filter with kernel size {self.speckle_filter_size} in {elapsed:.2f} seconds")
        else:
            filtered_image = image_data
            self.logger.info("Speckle filtering disabled")
        
        # Initialize ship detector
        start_time = datetime.datetime.now()
        self.logger.info("Initializing ship detector...")
        self.ship_detector = ShipDetector(filtered_image)
        
        # Run detection pipeline
        self.logger.info("Running ship detection pipeline...")
        detection_results = self.ship_detector.process_all()
        end_time = datetime.datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Count detected ships
        num_ships = len(detection_results['filtered_ships'])
        self.logger.info(f"Detected {num_ships} ships in {elapsed:.2f} seconds")
        
        # If we're using a preview image, we need to scale coordinates back to original
        if 'preview_factor' in self.read_results and self.read_results['preview_factor'] > 1:
            self.logger.info("Scaling detection coordinates to original resolution...")
            orig_shape = self.read_results['original_shape']
            preview_shape = self.read_results['preview_shape']
            
            # Scale ship regions
            for region in detection_results['filtered_ships']:
                x1, y1, x2, y2 = region['bbox']
                
                # Scale back to original coordinates
                x1_orig, y1_orig, x2_orig, y2_orig = self.preprocessor.scale_coordinates(
                    (x1, y1, x2, y2), orig_shape, preview_shape)
                
                # Update bounding box
                region['bbox'] = (x1_orig, y1_orig, x2_orig, y2_orig)
                region['width'] = x2_orig - x1_orig + 1
                region['height'] = y2_orig - y1_orig + 1
                region['area'] = region['width'] * region['height']
                region['center'] = ((x1_orig + x2_orig) // 2, (y1_orig + y2_orig) // 2)
                region['centroid'] = ((y1_orig + y2_orig) // 2, (x1_orig + x2_orig) // 2)
                
                # We can't update the region or mask here, they'll be extracted from the original later
                # Don't delete them as they're needed for some internal processing
        
        # Store results
        self.detection_results = detection_results
        
        return detection_results
    
    def manually_select_ships(self, image_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Allow manual selection of ships in the image using downsampled preview.
        
        Parameters
        ----------
        image_data : Optional[np.ndarray], optional
            Custom image data to use, by default None (uses preview image)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing ship regions.
        """
        self.logger.info("Starting manual ship selection using low-resolution preview...")
        
        # Use the downsampled preview image for display and selection
        if image_data is None and 'preview_image' in self.read_results:
            image_data = self.read_results['preview_image']
            using_preview = True
            self.logger.info(f"Using preview image with shape {image_data.shape}")
        elif image_data is None and 'focused_image' in self.read_results:
            image_data = self.read_results['focused_image']
            using_preview = False
            self.logger.warning("Preview image not found, using full resolution")
        elif image_data is None and 'image_data' in self.read_results:
            image_data = self.read_results['image_data']
            using_preview = False
            self.logger.warning("Preview image not found, using full resolution")
        else:
            using_preview = False
        
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
            
            # Add region with preview coordinates
            region_data = {
                'bbox': (x1, y1, x2, y2),
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
        if using_preview:
            info_text = f'Using downsampled preview ({self.read_results["preview_factor"]}x) for efficiency. '
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
        
        self.logger.info(f"Manual selection complete. {len(selected_regions)} ships selected.")
        
        # If using preview, scale coordinates back to original
        if using_preview and self.read_results['preview_factor'] > 1:
            self.logger.info("Scaling selection coordinates to original resolution...")
            orig_shape = self.read_results['original_shape']
            preview_shape = self.read_results['preview_shape']
            
            # Get the full resolution image data
            if 'focused_image' in self.read_results:
                full_image = self.read_results['focused_image']
            else:
                full_image = self.read_results['image_data']
            
            # Scale and extract from original for each region
            original_regions = []
            for region in selected_regions:
                x1, y1, x2, y2 = region['bbox']
                
                # Scale back to original coordinates
                x1_orig, y1_orig, x2_orig, y2_orig = self.preprocessor.scale_coordinates(
                    (x1, y1, x2, y2), orig_shape, preview_shape)
                
                # Extract region from full resolution image
                orig_region = full_image[y1_orig:y2_orig+1, x1_orig:x2_orig+1]
                
                # Create mask
                mask = np.ones((y2_orig-y1_orig+1, x2_orig-x1_orig+1), dtype=bool)
                
                # Calculate center
                center_y = (y1_orig + y2_orig) // 2
                center_x = (x1_orig + x2_orig) // 2
                
                # Add to original regions
                orig_region_data = {
                    'bbox': (x1_orig, y1_orig, x2_orig, y2_orig),
                    'center': (center_x, center_y),
                    'centroid': (center_y, center_x),
                    'region': orig_region,
                    'mask': mask,
                    'width': x2_orig - x1_orig + 1,
                    'height': y2_orig - y1_orig + 1,
                    'area': (x2_orig - x1_orig + 1) * (y2_orig - y1_orig + 1)
                }
                
                original_regions.append(orig_region_data)
            
            selected_regions = original_regions
                
        else:
            # Extract regions from full image if not using preview
            for region in selected_regions:
                x1, y1, x2, y2 = region['bbox']
                
                # Extract region
                region['region'] = image_data[y1:y2+1, x1:x2+1]
                
                # Create mask
                region['mask'] = np.ones((y2-y1+1, x2-y1+1), dtype=bool)
                
                # Calculate center
                center_y = (y1 + y2) // 2
                center_x = (x1 + x2) // 2
                
                region['center'] = (center_x, center_y)
                region['centroid'] = (center_y, center_x)
        
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
        num_subapertures: Optional[int] = None
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
        num_subapertures : Optional[int], optional
            Number of subapertures to use, by default None (uses self.num_subapertures)
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of micro-motion analysis results.
        """
        # Use instance parameter if not explicitly provided
        if num_subapertures is None:
            num_subapertures = self.num_subapertures
            
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
            
            # Classify ship components if component analysis is enabled
            if self.component_analysis:
                component_map = self.component_classifier.classify_components(
                    ship_image, ship['mask'])
                self.logger.info("Performed component classification")
            else:
                # Create a simple component map with all pixels as one component
                component_map = np.ones_like(ship['mask'], dtype=int)
                self.logger.info("Component analysis disabled, using uniform component map")
            
            # Analyze phase histories
            vibration_analysis = self.time_freq_analyzer.analyze_ship_region(
                ship_phase_data['phase_histories'], sampling_freq)
            
            # Apply physical constraints if not disabled
            if not self.skip_constraints:
                constrained_freqs, constrained_amps = self.physics_analyzer.apply_physical_constraints(
                    component_map,
                    vibration_analysis['dominant_frequencies'],
                    vibration_analysis['dominant_amplitudes']
                )
            else:
                self.logger.info("Skipping physical constraints as requested")
                constrained_freqs = vibration_analysis['dominant_frequencies']
                constrained_amps = vibration_analysis['dominant_amplitudes']
            
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
            'subaperture_timestamps': np.arange(num_subapertures) / sampling_freq,
            'physical_constraints_applied': not self.skip_constraints,
            'component_analysis_enabled': self.component_analysis
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
        
        # Apply normalization if enabled
        if self.apply_normalization:
            display_data = self.preprocessor.scale_for_display(image_data)
            self.logger.info("Applied normalization for visualization")
        else:
            # Just use absolute value for display
            display_data = np.abs(image_data)
            display_data = display_data / np.max(display_data)
            self.logger.info("Using simple absolute value for visualization")
            
        # Create ship detection overview
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Display the image
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
                
                # Create component classification visualization if component analysis is enabled
                if self.component_analysis:
                    component_fig = self.visualizer.create_component_overlay(
                        ship_image, ship_result['component_map']
                    )
                    figures[f'ship_{ship_index}_components'] = component_fig
                
                # Create micromotion heatmap
                heatmap_fig = self.visualizer.create_micromotion_heatmap(
                    ship_image,
                    ship_result['constrained_frequencies'],
                    ship_result['constrained_amplitudes'],
                    ship_result['component_map'] if self.component_analysis else None
                )
                figures[f'ship_{ship_index}_micromotion'] = heatmap_fig
                
                # Create vibration spectrograms for selected points
                # If component analysis is enabled, create spectrograms for each component
                if self.component_analysis:
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
                else:
                    # Create a single spectrogram for the center of the ship
                    height, width = ship_result['phase_histories']['dimensions']
                    center_y, center_x = height // 2, width // 2
                    phase_history = ship_result['phase_histories']['phase_histories'][center_y, center_x, :]
                    
                    spec_fig = self.visualizer.create_vibration_spectrogram(
                        phase_history, ship_result['sampling_freq']
                    )
                    
                    figures[f'ship_{ship_index}_spectrogram'] = spec_fig
                
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
        
        # Choose the appropriate processing method based on configuration
        if self.tile_processing:
            self.logger.info("Using tile-based processing for large datasets")
            return self.process_large_dataset(tile_size=self.tile_size)
        elif self.use_pipeline:
            self.logger.info("Using pipeline parallelism with Dask")
            return self.pipeline_process()
        
        # If we're here, we're using regular processing with potential 
        # thread/GPU acceleration within individual processing steps
        
        # Read data
        self.read_data()
        
        # Memory usage info for monitoring
        self._log_memory_usage("loading data")
        
        # If the file is not already a cropped file, ask if we should load one
        if not (self.input_file.endswith('.npy') and '_cropped_' in self.input_file):
            # Check if we should load a cropped file instead
            use_cropped = input("Do you want to load a previously cropped file? (y/n): ").lower().strip() == 'y'
            if use_cropped:
                cropped_file = input("Enter the path to the cropped file: ")
                try:
                    self.read_results = self.load_cropped_cphd(cropped_file)
                    
                    # Create preview if it doesn't exist
                    if 'preview_image' not in self.read_results and 'focused_image' in self.read_results:
                        orig_image = self.read_results['focused_image']
                        self.read_results['preview_image'] = self.preprocessor.downsample_image(
                            orig_image, self.preview_downsample_factor)
                        self.read_results['preview_factor'] = self.preview_downsample_factor
                        self.read_results['original_shape'] = orig_image.shape
                        self.read_results['preview_shape'] = self.read_results['preview_image'].shape
                        
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
            
            # Check if we're only cropping
            if self.crop_only:
                self.logger.info("Crop-only mode activated, skipping detection and analysis")
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
                else:
                    return {
                        'status': 'error',
                        'message': 'Cropping cancelled or failed'
                    }
            
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
            
            # Detect ships using the preview image (either automatically or manually)
            try:
                if self.use_manual_selection:
                    self.logger.info("Using manual ship selection as requested")
                    self.manually_select_ships()  # Uses preview by default
                else:
                    self.detect_ships()  # Uses preview by default
                
                # Log memory usage after detection
                self._log_memory_usage("ship detection")
                
                # Extract full resolution regions for detected ships
                if 'preview_factor' in self.read_results and self.read_results['preview_factor'] > 1:
                    self.logger.info("Extracting full resolution regions for detected ships...")
                    
                    # For each ship, extract the region from the full resolution image
                    for ship in self.detection_results['filtered_ships']:
                        if 'region' not in ship:  # Skip if region already extracted
                            x1, y1, x2, y2 = ship['bbox']
                            ship['region'] = focused_image[y1:y2+1, x1:x2+1]
                            
                            # Create or update mask if needed
                            if 'mask' not in ship or ship['mask'].shape != (y2-y1+1, x2-x1+1):
                                ship['mask'] = np.ones((y2-y1+1, x2-x1+1), dtype=bool)
                
                # Analyze micro-motion
                self.analyze_ship_micromotion(
                    signal_data, 
                    pvp_data, 
                    self.detection_results['filtered_ships'],
                    self.num_subapertures
                )
                
                # Log memory usage after micromotion analysis
                self._log_memory_usage("micromotion analysis")
                
                # Create visualizations
                self.create_visualizations()
                
                # Save results
                saved_files = self.save_results()
                
            except MemoryError as e:
                self.logger.error(f"Memory error during processing: {str(e)}")
                print(f"Memory error during processing. Try using a smaller preview or crop the data first.")
                return {
                    'status': 'error',
                    'message': f'Memory error: {str(e)}'
                }
            
        elif self.read_results['type'] == 'sicd':
            # For SICD data, we can only detect ships since we need raw signal data for vibration analysis
            image_data = self.read_results['image_data']
            
            try:
                # Detect ships using the preview image (either automatically or manually)
                if self.use_manual_selection:
                    self.logger.info("Using manual ship selection as requested")
                    self.manually_select_ships()  # Uses preview by default
                else:
                    self.detect_ships()  # Uses preview by default
                
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
                
            except MemoryError as e:
                self.logger.error(f"Memory error during processing: {str(e)}")
                print(f"Memory error during processing. Try using a smaller preview or crop the data first.")
                return {
                    'status': 'error',
                    'message': f'Memory error: {str(e)}'
                }
            
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

    def process_large_dataset(self, tile_size: int = 512, overlap: int = 64) -> Dict[str, Any]:
        """
        Process large SAR datasets using tile-based processing to minimize memory usage.
        This method divides the image into overlapping tiles, processes each tile separately,
        and then combines the results.
        
        Parameters
        ----------
        tile_size : int, optional
            Size of each processing tile, by default 512
        overlap : int, optional
            Overlap between adjacent tiles to avoid edge artifacts, by default 64
            
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processing results.
        """
        self.logger.info(f"Starting tile-based processing with tile size {tile_size}x{tile_size} and overlap {overlap}")
        
        # Check if data is loaded
        if not self.read_results:
            self.read_data()
        
        # Get the image data
        if self.read_results['type'] == 'cphd':
            full_image = self.read_results['focused_image']
            signal_data = self.read_results['signal_data']
            pvp_data = self.read_results['pvp_data']
        elif self.read_results['type'] == 'sicd':
            full_image = self.read_results['image_data']
            signal_data = None
            pvp_data = None
        else:
            self.logger.error("Unsupported data type for tile-based processing")
            return {'status': 'error', 'message': 'Unsupported data type'}
        
        # Get image dimensions
        img_height, img_width = full_image.shape
        
        # Calculate number of tiles
        n_tiles_h = max(1, (img_height - overlap) // (tile_size - overlap))
        n_tiles_w = max(1, (img_width - overlap) // (tile_size - overlap))
        
        self.logger.info(f"Processing image of size {img_height}x{img_width} using {n_tiles_h}x{n_tiles_w} tiles")
        
        # Initialize result containers
        all_ship_regions = []
        tile_metadata = []
        
        # Process each tile
        for i in range(n_tiles_h):
            for j in range(n_tiles_w):
                # Calculate tile boundaries with overlap
                start_h = i * (tile_size - overlap)
                end_h = min(start_h + tile_size, img_height)
                start_w = j * (tile_size - overlap)
                end_w = min(start_w + tile_size, img_width)
                
                # Adjust start positions to ensure tile size doesn't exceed image boundaries
                if end_h - start_h < tile_size:
                    start_h = max(0, end_h - tile_size)
                if end_w - start_w < tile_size:
                    start_w = max(0, end_w - tile_size)
                
                self.logger.info(f"Processing tile ({i},{j}) at position ({start_h}:{end_h}, {start_w}:{end_w})")
                
                # Extract tile data
                tile_image = full_image[start_h:end_h, start_w:end_w]
                
                if signal_data is not None:
                    tile_signal = signal_data[start_h:end_h, start_w:end_w]
                else:
                    tile_signal = None
                
                # Create a temporary processor for this tile
                tile_processor = SARImagePreprocessor(self.logger)
                
                # Apply speckle filter if enabled
                if self.speckle_filter_size > 0:
                    tile_image = tile_processor.apply_speckle_filter(
                        tile_image, self.speckle_filter_size)
                
                # Detect ships in the tile
                ship_detector = ShipDetector(tile_image)
                tile_detection = ship_detector.process_all()
                
                # Adjust ship coordinates to global image coordinates
                for ship in tile_detection['filtered_ships']:
                    x1, y1, x2, y2 = ship['bbox']
                    
                    # Convert to global coordinates
                    global_x1 = x1 + start_w
                    global_y1 = y1 + start_h
                    global_x2 = x2 + start_w
                    global_y2 = y2 + start_h
                    
                    # Create global ship region
                    global_ship = ship.copy()
                    global_ship['bbox'] = (global_x1, global_y1, global_x2, global_y2)
                    global_ship['center'] = (
                        (global_x1 + global_x2) // 2, 
                        (global_y1 + global_y2) // 2
                    )
                    global_ship['centroid'] = (
                        (global_y1 + global_y2) // 2,
                        (global_x1 + global_x2) // 2
                    )
                    global_ship['tile_coords'] = (i, j)
                    global_ship['tile_bounds'] = (start_h, end_h, start_w, end_w)
                    
                    # Only keep ships that are not on the overlap border
                    # (to avoid duplicate detections)
                    border_margin = overlap // 2
                    if (i == 0 or y1 >= border_margin) and \
                       (j == 0 or x1 >= border_margin) and \
                       (i == n_tiles_h-1 or y2 <= tile_size - border_margin) and \
                       (j == n_tiles_w-1 or x2 <= tile_size - border_margin):
                        all_ship_regions.append(global_ship)
                
                # Store tile metadata
                tile_metadata.append({
                    'tile_id': (i, j),
                    'bounds': (start_h, end_h, start_w, end_w),
                    'ships_detected': len(tile_detection['filtered_ships'])
                })
                
                # Free memory
                del tile_image, ship_detector, tile_detection
                if tile_signal is not None:
                    del tile_signal
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
        
        # Remove duplicate ship detections
        # Two ships are considered duplicates if their IoU > 0.5
        filtered_ships = []
        ship_added = [False] * len(all_ship_regions)
        
        def calculate_iou(box1, box2):
            # Calculate Intersection over Union for two bounding boxes
            x1_1, y1_1, x2_1, y2_1 = box1
            x1_2, y1_2, x2_2, y2_2 = box2
            
            # Calculate intersection area
            x_left = max(x1_1, x1_2)
            y_top = max(y1_1, y1_2)
            x_right = min(x2_1, x2_2)
            y_bottom = min(y2_1, y2_2)
            
            if x_right < x_left or y_bottom < y_top:
                return 0.0
                
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            
            # Calculate union area
            box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
            box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
            union_area = box1_area + box2_area - intersection_area
            
            return intersection_area / union_area
        
        for i in range(len(all_ship_regions)):
            if ship_added[i]:
                continue
                
            current_ship = all_ship_regions[i]
            filtered_ships.append(current_ship)
            ship_added[i] = True
            
            # Mark all similar ships as added
            for j in range(i+1, len(all_ship_regions)):
                if ship_added[j]:
                    continue
                    
                # Calculate IoU between ships
                iou = calculate_iou(
                    current_ship['bbox'],
                    all_ship_regions[j]['bbox']
                )
                
                if iou > 0.5:  # Threshold for considering as duplicate
                    ship_added[j] = True
        
        self.logger.info(f"Filtered {len(all_ship_regions)} initial detections to {len(filtered_ships)} unique ships")
        
        # Extract regions from full image for each ship
        for ship in filtered_ships:
            x1, y1, x2, y2 = ship['bbox']
            ship['region'] = full_image[y1:y2+1, x1:x2+1]
            ship['mask'] = np.ones((y2-y1+1, x2-x1+1), dtype=bool)
        
        # Store detection results
        self.detection_results = {
            'filtered_ships': filtered_ships,
            'ship_regions': filtered_ships,
            'num_ships': len(filtered_ships),
            'tile_metadata': tile_metadata
        }
        
        # If signal data is available, analyze micro-motion for each ship
        if signal_data is not None and pvp_data is not None:
            # For micro-motion analysis, we need to extract separate signal data
            # for each ship to reduce memory usage
            ship_results = []
            
            for ship_idx, ship in enumerate(filtered_ships):
                self.logger.info(f"Analyzing micro-motion for ship {ship_idx+1}/{len(filtered_ships)}")
                
                # Extract ship region plus margin for analysis
                x1, y1, x2, y2 = ship['bbox']
                
                # Add margin for better analysis
                margin = 10
                x1_margin = max(0, x1 - margin)
                y1_margin = max(0, y1 - margin)
                x2_margin = min(img_width - 1, x2 + margin)
                y2_margin = min(img_height - 1, y2 + margin)
                
                # Extract ship signal data
                ship_signal = signal_data[y1_margin:y2_margin+1, x1_margin:x2_margin+1]
                
                # Create subapertures for this ship
                phase_extractor = PixelPhaseHistoryExtractor(self.logger)
                subapertures = phase_extractor.create_subapertures(
                    ship_signal, self.num_subapertures)
                
                # Adjust ship coordinates for the extracted region
                ship_bbox_local = (
                    x1 - x1_margin,
                    y1 - y1_margin,
                    x2 - x1_margin,
                    y2 - y1_margin
                )
                
                ship_local = {
                    'bbox': ship_bbox_local,
                    'mask': ship['mask']
                }
                
                # Extract phase histories
                ship_phase_data = phase_extractor.extract_ship_phase_histories(ship_local)
                
                # Get ship region image for component classification
                ship_image = subapertures[0, 
                                        ship_bbox_local[1]:ship_bbox_local[3]+1, 
                                        ship_bbox_local[0]:ship_bbox_local[2]+1]
                
                # Classify ship components if enabled
                if self.component_analysis:
                    component_classifier = ShipComponentClassifier(self.logger)
                    component_map = component_classifier.classify_components(
                        ship_image, ship['mask'])
                else:
                    component_map = np.ones_like(ship['mask'], dtype=int)
                
                # Analyze phase histories
                time_freq_analyzer = TimeFrequencyAnalyzer(self.logger)
                vibration_analysis = time_freq_analyzer.analyze_ship_region(
                    ship_phase_data['phase_histories'], 1.0)  # Default sampling freq
                
                # Apply physical constraints if not disabled
                if not self.skip_constraints:
                    physics_analyzer = PhysicsConstrainedAnalyzer(self.logger)
                    constrained_freqs, constrained_amps = physics_analyzer.apply_physical_constraints(
                        component_map,
                        vibration_analysis['dominant_frequencies'],
                        vibration_analysis['dominant_amplitudes']
                    )
                else:
                    constrained_freqs = vibration_analysis['dominant_frequencies']
                    constrained_amps = vibration_analysis['dominant_amplitudes']
                
                # Collect results for this ship
                ship_result = {
                    'ship_index': ship_idx,
                    'bbox': ship['bbox'],
                    'phase_histories': ship_phase_data,
                    'vibration_analysis': vibration_analysis,
                    'component_map': component_map,
                    'constrained_frequencies': constrained_freqs,
                    'constrained_amplitudes': constrained_amps,
                    'sampling_freq': 1.0  # Default
                }
                
                ship_results.append(ship_result)
                
                # Free memory
                del ship_signal, subapertures, phase_extractor
                import gc
                gc.collect()
            
            # Store vibration results
            self.vibration_results = {
                'num_ships': len(filtered_ships),
                'ship_results': ship_results,
                'sampling_freq': 1.0,  # Default
                'num_subapertures': self.num_subapertures,
                'subaperture_timestamps': np.arange(self.num_subapertures),
                'physical_constraints_applied': not self.skip_constraints,
                'component_analysis_enabled': self.component_analysis
            }
            
            # Create visualizations
            self.create_visualizations()
        
        # Save results
        saved_files = self.save_results()
        
        return {
            'status': 'success',
            'read_results': self.read_results,
            'detection_results': self.detection_results,
            'vibration_results': self.vibration_results if signal_data is not None else None,
            'visualization_results': self.visualization_results,
            'saved_files': saved_files
        }

    def pipeline_process(self) -> Dict[str, Any]:
        """
        Run the processing pipeline with task parallelism using Dask.
        This implementation creates a directed acyclic graph (DAG) of computations,
        allowing for parallel execution of independent tasks.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing processing results.
        """
        if not DASK_AVAILABLE:
            self.logger.warning("Dask not available, falling back to sequential processing")
            return self.process()
            
        self.logger.info("Starting pipeline processing with Dask task parallelism")
        
        # Define task functions for pipeline stages
        @delayed
        def read_data_task():
            """Task for reading SAR data."""
            self.logger.info("Task: Reading SAR data")
            return self.read_data()
        
        @delayed
        def detect_ships_task(read_results):
            """Task for detecting ships."""
            self.logger.info("Task: Detecting ships")
            self.read_results = read_results
            
            if self.use_manual_selection:
                return self.manually_select_ships()
            else:
                return self.detect_ships()
        
        @delayed
        def analyze_micromotion_task(read_results, detection_results):
            """Task for analyzing ship micro-motion."""
            self.logger.info("Task: Analyzing ship micro-motion")
            self.read_results = read_results
            self.detection_results = detection_results
            
            if read_results['type'] == 'cphd':
                return self.analyze_ship_micromotion(
                    read_results['signal_data'], 
                    read_results['pvp_data'], 
                    detection_results['filtered_ships'],
                    self.num_subapertures
                )
            else:
                self.logger.warning("Micro-motion analysis not possible - requires CPHD data")
                return None
        
        @delayed
        def create_visualizations_task(read_results, detection_results, vibration_results):
            """Task for creating visualizations."""
            self.logger.info("Task: Creating visualizations")
            self.read_results = read_results
            self.detection_results = detection_results
            self.vibration_results = vibration_results
            
            return self.create_visualizations()
        
        @delayed
        def save_results_task(visualization_results):
            """Task for saving results."""
            self.logger.info("Task: Saving results")
            self.visualization_results = visualization_results
            
            return self.save_results()
        
        # Set up pipeline DAG
        # Stage 1: Read data
        read_stage = read_data_task()
        
        # Stage 2: Detect ships
        detection_stage = detect_ships_task(read_stage)
        
        # Stage 3: Analyze micro-motion (if applicable)
        micromotion_stage = analyze_micromotion_task(read_stage, detection_stage)
        
        # Stage 4: Create visualizations
        visualization_stage = create_visualizations_task(read_stage, detection_stage, micromotion_stage)
        
        # Stage 5: Save results
        save_stage = save_results_task(visualization_stage)
        
        # Execute the pipeline with progress reporting
        self.logger.info("Executing pipeline...")
        
        # Monitor computation progress
        try:
            from dask.diagnostics import ProgressBar
            with ProgressBar():
                saved_files = save_stage.compute()
        except ImportError:
            saved_files = save_stage.compute()
        
        self.logger.info("Pipeline processing complete")
        
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