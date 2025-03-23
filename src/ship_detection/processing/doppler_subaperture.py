"""
Module for Doppler sub-aperture processing to analyze micro-motions in SAR data.

This implements the approach described in the paper:
"Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in
Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment"
"""

import logging
from typing import Dict, Tuple, List, Optional, Union, Any

import numpy as np
from scipy import signal, fft
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate2d
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class DopplerSubapertureProcessor:
    """
    Processor for Doppler sub-aperture analysis of SAR data to detect micro-motions.
    """
    
    def __init__(self, signal_data: np.ndarray, pvp_data: Dict[str, np.ndarray], 
                 num_subapertures: int = 200):
        """
        Initialize the Doppler sub-aperture processor.
        
        Parameters
        ----------
        signal_data : np.ndarray
            The signal data from the SAR file.
        pvp_data : Dict[str, np.ndarray]
            Dictionary containing Per Vector Parameter (PVP) data.
        num_subapertures : int, optional
            Number of sub-apertures to create, by default 200
        """
        self.signal_data = signal_data
        self.pvp_data = pvp_data
        self.num_subapertures = num_subapertures
        self.subapertures = []
        self.focused_images = []
        self.tx_times = pvp_data.get('TxTime')
        
        # Calculate acquisition time details if TxTime is available
        if self.tx_times is not None:
            self.acquisition_duration = self.tx_times[-1] - self.tx_times[0]
            self.time_per_subaperture = self.acquisition_duration / self.num_subapertures
            logger.info(f"Acquisition duration: {self.acquisition_duration:.2f}s")
            logger.info(f"Time per sub-aperture: {self.time_per_subaperture:.4f}s")
            logger.info(f"Maximum observable frequency: {1/(2*self.time_per_subaperture):.2f}Hz")
        else:
            logger.warning("TxTime not available in PVP data. Using default timing.")
            self.acquisition_duration = 12.0  # Default based on paper
            self.time_per_subaperture = self.acquisition_duration / self.num_subapertures
    
    def create_subapertures(self) -> List[np.ndarray]:
        """
        Split the signal data into Doppler sub-apertures.
        
        Returns
        -------
        List[np.ndarray]
            List of sub-aperture data arrays.
        """
        # Get dimensions of the signal data
        num_pulses, num_samples = self.signal_data.shape
        
        # Calculate pulses per sub-aperture
        pulses_per_subaperture = num_pulses // self.num_subapertures
        
        # Create sub-apertures
        self.subapertures = []
        for i in range(self.num_subapertures):
            start_idx = i * pulses_per_subaperture
            end_idx = (i + 1) * pulses_per_subaperture
            
            # Handle the last sub-aperture to include remaining pulses
            if i == self.num_subapertures - 1:
                end_idx = num_pulses
                
            subaperture = self.signal_data[start_idx:end_idx, :]
            self.subapertures.append(subaperture)
            
        logger.info(f"Created {len(self.subapertures)} sub-apertures")
        return self.subapertures
    
    def focus_subapertures(self) -> List[np.ndarray]:
        """
        Focus each sub-aperture to create a series of SAR images.
        This is a simplified focusing process.
        
        Returns
        -------
        List[np.ndarray]
            List of focused sub-aperture images.
        """
        self.focused_images = []
        
        for i, subaperture in enumerate(self.subapertures):
            # Apply simple Range-Doppler Algorithm (simplified)
            # 1. Range compression (FFT in range dimension)
            range_compressed = fft.fft(subaperture, axis=1)
            
            # 2. Azimuth FFT
            azimuth_fft = fft.fft(range_compressed, axis=0)
            
            # 3. Range Cell Migration Correction would be here in a full processor
            # (simplified - just using the FFT result)
            
            # 4. Azimuth compression (inverse FFT)
            focused = fft.ifft(azimuth_fft, axis=0)
            
            # Convert to magnitude for visualization
            focused_mag = np.abs(focused)
            
            # Apply Gaussian smoothing to reduce noise
            focused_smoothed = gaussian_filter(focused_mag, sigma=1.0)
            
            self.focused_images.append(focused_smoothed)
            
            if i % 20 == 0:
                logger.info(f"Focused sub-aperture {i+1}/{len(self.subapertures)}")
                
        logger.info(f"Focused {len(self.focused_images)} sub-aperture images")
        return self.focused_images
    
    def compute_coregistration_shifts(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sub-pixel shifts between consecutive sub-aperture images using 
        normalized cross-correlation.
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays of range and azimuth shifts for each consecutive pair of images.
        """
        if not self.focused_images:
            logger.warning("No focused images available. Running focusing.")
            self.focus_subapertures()
            
        num_images = len(self.focused_images)
        range_shifts = np.zeros(num_images - 1)
        azimuth_shifts = np.zeros(num_images - 1)
        
        # Window size for correlation (adjust based on your image size)
        window_size = 128
        
        for i in range(num_images - 1):
            # Get windows from the center of consecutive images
            img1 = self.focused_images[i]
            img2 = self.focused_images[i + 1]
            
            # Get the image dimensions
            rows, cols = img1.shape
            
            # Calculate window position (center of the image)
            start_row = max(0, rows // 2 - window_size // 2)
            start_col = max(0, cols // 2 - window_size // 2)
            
            # Extract windows (ensure they don't go beyond image boundaries)
            end_row = min(rows, start_row + window_size)
            end_col = min(cols, start_col + window_size)
            
            window1 = img1[start_row:end_row, start_col:end_col]
            window2 = img2[start_row:end_row, start_col:end_col]
            
            # Compute normalized cross-correlation
            correlation = correlate2d(window1, window2, mode='same', boundary='symm')
            
            # Find the peak in the correlation
            max_idx = np.unravel_index(np.argmax(correlation), correlation.shape)
            
            # Calculate shifts (subtracting the center position)
            dy = max_idx[0] - window_size // 2
            dx = max_idx[1] - window_size // 2
            
            # Store the shifts
            range_shifts[i] = dx
            azimuth_shifts[i] = dy
            
            if i % 20 == 0:
                logger.info(f"Computed shifts for images {i+1}/{num_images-1}")
        
        logger.info("Completed coregistration shift computation")
        return range_shifts, azimuth_shifts
    
    def estimate_vibration_parameters(self, range_shifts: np.ndarray, 
                                     azimuth_shifts: np.ndarray,
                                     oversample_factor: int = 4,
                                     max_freq: Optional[float] = None) -> Dict[str, Any]:
        """
        Estimate vibration parameters from the shifts.
        
        Parameters
        ----------
        range_shifts : np.ndarray
            Array of range shifts.
        azimuth_shifts : np.ndarray
            Array of azimuth shifts.
        oversample_factor : int, optional
            Oversampling factor for frequency analysis, by default 4
        max_freq : Optional[float], optional
            Maximum frequency to analyze (Hz), by default None
            
        Returns
        -------
        Dict[str, Any]
            Dictionary of vibration parameters including:
            - 'times': time points
            - 'range_shifts': interpolated range shifts
            - 'azimuth_shifts': interpolated azimuth shifts
            - 'frequencies': frequency array
            - 'range_spectrum': range vibration spectrum
            - 'azimuth_spectrum': azimuth vibration spectrum
            - 'dominant_freq_range': dominant frequency in range direction
            - 'dominant_freq_azimuth': dominant frequency in azimuth direction
        """
        # Create time array
        times = np.arange(len(range_shifts)) * self.time_per_subaperture
        
        # Interpolate shifts for smoother analysis
        interp_times = np.linspace(times[0], times[-1], len(times) * oversample_factor)
        interp_range_shifts = np.interp(interp_times, times, range_shifts)
        interp_azimuth_shifts = np.interp(interp_times, times, azimuth_shifts)
        
        # Compute the FFT
        n_samples = len(interp_times)
        sampling_freq = 1 / (interp_times[1] - interp_times[0])
        
        # Apply a Hanning window to reduce spectral leakage
        window = np.hanning(n_samples)
        windowed_range = interp_range_shifts * window
        windowed_azimuth = interp_azimuth_shifts * window
        
        # Calculate the FFT
        range_fft = fft.fft(windowed_range)
        azimuth_fft = fft.fft(windowed_azimuth)
        
        # Calculate magnitude spectrum
        range_spectrum = np.abs(range_fft[:n_samples//2]) / n_samples
        azimuth_spectrum = np.abs(azimuth_fft[:n_samples//2]) / n_samples
        
        # Double amplitude to account for negative frequencies
        range_spectrum[1:] *= 2
        azimuth_spectrum[1:] *= 2
        
        # Frequency array
        frequencies = np.linspace(0, sampling_freq/2, n_samples//2)
        
        # Limit to max_freq if specified
        if max_freq is not None and max_freq < frequencies[-1]:
            valid_idx = frequencies <= max_freq
            frequencies = frequencies[valid_idx]
            range_spectrum = range_spectrum[valid_idx]
            azimuth_spectrum = azimuth_spectrum[valid_idx]
        
        # Find dominant frequencies
        dominant_freq_range_idx = np.argmax(range_spectrum[1:]) + 1
        dominant_freq_azimuth_idx = np.argmax(azimuth_spectrum[1:]) + 1
        
        dominant_freq_range = frequencies[dominant_freq_range_idx]
        dominant_freq_azimuth = frequencies[dominant_freq_azimuth_idx]
        
        logger.info(f"Dominant range vibration frequency: {dominant_freq_range:.2f} Hz")
        logger.info(f"Dominant azimuth vibration frequency: {dominant_freq_azimuth:.2f} Hz")
        
        # Return results
        return {
            'times': interp_times,
            'range_shifts': interp_range_shifts,
            'azimuth_shifts': interp_azimuth_shifts,
            'frequencies': frequencies,
            'range_spectrum': range_spectrum,
            'azimuth_spectrum': azimuth_spectrum,
            'dominant_freq_range': dominant_freq_range,
            'dominant_freq_azimuth': dominant_freq_azimuth,
        }
    
    def bandpass_filter_vibrations(self, shifts: np.ndarray, 
                                  sampling_freq: float, 
                                  low_freq: float = 10.0, 
                                  high_freq: float = 30.0) -> np.ndarray:
        """
        Apply a bandpass filter to the shifts to isolate specific vibration frequencies.
        
        Parameters
        ----------
        shifts : np.ndarray
            Array of shifts to filter.
        sampling_freq : float
            Sampling frequency of the shifts.
        low_freq : float, optional
            Low cutoff frequency in Hz, by default 10.0
        high_freq : float, optional
            High cutoff frequency in Hz, by default 30.0
            
        Returns
        -------
        np.ndarray
            Filtered shifts.
        """
        # Normalize frequencies to Nyquist
        nyquist = sampling_freq / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        # Design bandpass filter
        b, a = signal.butter(4, [low, high], btype='band')
        
        # Apply filter
        filtered_shifts = signal.filtfilt(b, a, shifts)
        
        return filtered_shifts
    
    def process_all(self) -> Dict[str, Any]:
        """
        Run the full processing chain.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all processing results.
        """
        # Create sub-apertures
        self.create_subapertures()
        
        # Focus sub-apertures
        self.focus_subapertures()
        
        # Compute shifts
        range_shifts, azimuth_shifts = self.compute_coregistration_shifts()
        
        # Estimate vibration parameters
        vibration_params = self.estimate_vibration_parameters(range_shifts, azimuth_shifts)
        
        # Apply bandpass filtering to focus on typical ship vibration frequencies (10-30 Hz)
        sampling_freq = 1 / (vibration_params['times'][1] - vibration_params['times'][0])
        filtered_range = self.bandpass_filter_vibrations(
            vibration_params['range_shifts'], sampling_freq, 10.0, 30.0)
        filtered_azimuth = self.bandpass_filter_vibrations(
            vibration_params['azimuth_shifts'], sampling_freq, 10.0, 30.0)
        
        # Add filtered results to the parameters
        vibration_params['filtered_range_shifts'] = filtered_range
        vibration_params['filtered_azimuth_shifts'] = filtered_azimuth
        
        return {
            'subapertures': self.subapertures,
            'focused_images': self.focused_images,
            'range_shifts': range_shifts,
            'azimuth_shifts': azimuth_shifts,
            'vibration_params': vibration_params,
        } 