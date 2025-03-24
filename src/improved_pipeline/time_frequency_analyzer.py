#!/usr/bin/env python3
"""
Time-frequency analysis module for the improved ship micro-motion analysis pipeline.

This module implements advanced time-frequency analysis techniques for the
pixel-level phase histories extracted from SAR data. It provides multiple
analysis methods including:
- Short-Time Fourier Transform (STFT)
- Continuous Wavelet Transform (CWT)
- Empirical Mode Decomposition (EMD)
- Adaptive window analysis
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from .utils import (
    setup_logging, save_results, load_step_output, check_gpu_availability,
    scale_for_display
)


class TimeFrequencyAnalyzer:
    """
    Performs advanced time-frequency analysis on phase histories.
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the time-frequency analyzer.
        
        Parameters
        ----------
        use_gpu : bool, optional
            Whether to use GPU acceleration if available, by default False
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        """
        self.use_gpu = use_gpu and check_gpu_availability()
        self.logger = logger or logging.getLogger(__name__)
        
        if self.use_gpu:
            self.logger.info("Using GPU acceleration for time-frequency analysis")
    
    def analyze_single_pixel(
        self, 
        phase_history: np.ndarray, 
        sampling_freq: float
    ) -> Dict[str, Any]:
        """
        Apply multiple time-frequency analysis methods to a single pixel.
        
        Parameters
        ----------
        phase_history : np.ndarray
            Phase history time series
        sampling_freq : float
            Sampling frequency
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with analysis results
        """
        results = {}
        
        # Ensure phase history is 1D
        phase_history = phase_history.flatten()
        
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
        
        # Short-Time Fourier Transform with adaptive window
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
        
        # Try to compute Continuous Wavelet Transform if scipy version supports it
        try:
            import scipy.signal
            
            # Check if cwt method exists in scipy.signal
            if hasattr(scipy.signal, 'cwt'):
                # Set scales based on data length
                max_scale = min(len(phase_history) // 4, 64)
                scales = np.arange(1, max_scale + 1)
                
                # Compute CWT
                wavelet = signal.morlet2
                cwtm = signal.cwt(phase_history, wavelet, scales)
                
                # Convert scales to frequencies
                # Approximate conversion for Morlet wavelets
                frequencies = sampling_freq / (scales * 2 * np.pi)
                
                results['cwt'] = {
                    'scales': scales,
                    'freqs': frequencies,
                    'coefficients': np.abs(cwtm)
                }
        except (ImportError, AttributeError) as e:
            self.logger.debug(f"CWT computation not available: {str(e)}")
        
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
    
    def analyze_ship_region(
        self, 
        phase_histories: np.ndarray, 
        sampling_freq: float
    ) -> Dict[str, Any]:
        """
        Analyze phase histories for all pixels in a ship region.
        
        Parameters
        ----------
        phase_histories : np.ndarray
            3D array of phase histories (height x width x time)
        sampling_freq : float
            Sampling frequency
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with analysis results
        """
        self.logger.info("Performing vectorized FFT analysis for all pixels")
        height, width, time_samples = phase_histories.shape
        
        # Check for GPU acceleration
        if self.use_gpu:
            try:
                import cupy as cp
                self.logger.info("Using GPU for vectorized FFT analysis")
                
                # Transfer data to GPU
                gpu_phases = cp.asarray(phase_histories)
                
                # Create window function
                window = cp.hanning(time_samples).reshape(1, 1, -1)  # 1 x 1 x time_samples
                
                # Apply windowing to all pixel phase histories at once
                windowed_signals = gpu_phases * window
                
                # Perform FFT on all pixels at once
                fft_results = cp.abs(cp.fft.fft(windowed_signals, axis=2))
                
                # Calculate frequency axis
                fft_freqs = cp.fft.fftfreq(time_samples, 1/sampling_freq)
                
                # Get positive frequencies only
                pos_mask = fft_freqs >= 0
                pos_freqs = fft_freqs[pos_mask]
                pos_spectra = fft_results[:, :, pos_mask]
                
                # Find dominant frequencies for each pixel
                # Skip the DC component (index 0)
                start_idx = 1
                dominant_freqs = cp.zeros((height, width))
                dominant_amps = cp.zeros((height, width))
                
                # Process each pixel (GPU implementation could be optimized further)
                for i in range(height):
                    for j in range(width):
                        spectrum = pos_spectra[i, j, start_idx:]
                        
                        if len(spectrum) > 0:
                            # Find peak (simplification - just taking max)
                            max_idx = cp.argmax(spectrum)
                            max_idx += start_idx  # Adjust for offset
                            
                            dominant_freqs[i, j] = pos_freqs[max_idx]
                            dominant_amps[i, j] = pos_spectra[i, j, max_idx]
                
                # Transfer results back to CPU
                return {
                    'dominant_frequencies': cp.asnumpy(dominant_freqs),
                    'dominant_amplitudes': cp.asnumpy(dominant_amps),
                    'all_spectra': cp.asnumpy(pos_spectra),
                    'frequency_axis': cp.asnumpy(pos_freqs)
                }
                
            except ImportError:
                self.logger.warning("CuPy not available, falling back to CPU implementation")
        
        # CPU implementation
        self.logger.info("Using CPU implementation for vectorized FFT analysis")
        
        # Create window function once
        window = np.hanning(time_samples).reshape(1, 1, -1)  # 1 x 1 x time_samples
        
        # Apply windowing to all pixel phase histories at once
        windowed_signals = phase_histories * window
        
        # Perform FFT on all pixels at once
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
        
        # Process each pixel
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
    
    def compute_stft_maps(
        self, 
        phase_histories: np.ndarray, 
        sampling_freq: float
    ) -> Dict[str, np.ndarray]:
        """
        Compute STFT maps for all pixels in a ship region.
        
        Parameters
        ----------
        phase_histories : np.ndarray
            3D array of phase histories (height x width x time)
        sampling_freq : float
            Sampling frequency
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with STFT results
        """
        self.logger.info("Computing STFT maps for entire ship region")
        height, width, time_samples = phase_histories.shape
        
        # Determine window size (adaptive)
        window_size = min(time_samples // 4, 128)
        window_size = max(window_size, 16)  # Ensure minimum size
        
        # Only compute if we have enough data points
        if time_samples < window_size * 2:
            self.logger.warning(f"Not enough time samples ({time_samples}) for STFT with window size {window_size}")
            return {
                'computed': False,
                'reason': f"Not enough time samples ({time_samples}) for STFT with window size {window_size}"
            }
        
        # Compute STFT for a single time series to get dimensions
        f, t, _ = signal.stft(
            phase_histories[0, 0, :], sampling_freq, 
            nperseg=window_size, noverlap=window_size//2
        )
        
        # Initialize STFT coefficients array
        stft_coeffs = np.zeros((height, width, len(f), len(t)), dtype=np.complex128)
        
        # Compute STFT for each pixel
        for i in range(height):
            for j in range(width):
                _, _, coeffs = signal.stft(
                    phase_histories[i, j, :], sampling_freq, 
                    nperseg=window_size, noverlap=window_size//2
                )
                stft_coeffs[i, j, :, :] = coeffs
        
        # Compute magnitude
        stft_magnitude = np.abs(stft_coeffs)
        
        return {
            'computed': True,
            'stft_coefficients': stft_coeffs,
            'stft_magnitude': stft_magnitude,
            'frequencies': f,
            'times': t,
            'window_size': window_size
        }


def analyze_time_frequency(
    input_file: str,
    output_file: str,
    use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Run time-frequency analysis on phase history data.
    
    Parameters
    ----------
    input_file : str
        Path to input file (phase history data)
    output_file : str
        Path to output file
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing time-frequency analysis results
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Analyzing time-frequency relationships in {input_file}")
    
    # Load phase history data
    phase_data = load_step_output(input_file)
    logger.info(f"Loaded phase history data with keys: {phase_data.keys()}")
    
    # Check for required data
    if 'ship_results' not in phase_data:
        logger.error("No ship_results found in phase history data")
        raise ValueError("Invalid phase history data format")
    
    # Initialize time-frequency analyzer
    analyzer = TimeFrequencyAnalyzer(
        use_gpu=use_gpu,
        logger=logger
    )
    
    # Process each ship
    ship_results = []
    for i, ship in enumerate(phase_data['ship_results']):
        logger.info(f"Processing ship {i+1}/{len(phase_data['ship_results'])}")
        
        # Check for required data
        if 'phase_histories' not in ship:
            logger.warning(f"No phase histories found for ship {i+1}, skipping")
            continue
        
        # Extract necessary data
        phase_histories = ship['phase_histories']
        sampling_freq = ship.get('sampling_freq', phase_data.get('sampling_freq', 1.0))
        
        # Analyze ship region
        tf_analysis = analyzer.analyze_ship_region(phase_histories, sampling_freq)
        
        # Compute STFT maps (if dataset is not too large)
        height, width, _ = phase_histories.shape
        if height * width <= 10000:  # Arbitrary limit to avoid memory issues
            stft_maps = analyzer.compute_stft_maps(phase_histories, sampling_freq)
            tf_analysis['stft_maps'] = stft_maps
        
        # Add metadata
        tf_analysis['ship_index'] = ship.get('ship_index', i)
        tf_analysis['ship_bbox'] = ship.get('ship_bbox')
        tf_analysis['dimensions'] = ship.get('dimensions')
        
        # Store results
        ship_results.append(tf_analysis)
    
    # Create complete results dictionary
    results = {
        'ship_results': ship_results,
        'num_ships': len(ship_results),
        'sampling_freq': phase_data.get('sampling_freq', 1.0),
        'timestamp': phase_data.get('timestamp', None),
        'input_file': input_file
    }
    
    # Save results
    save_results(output_file, results)
    logger.info(f"Saved time-frequency analysis results to {output_file}")
    
    # Create visualizations
    create_tf_visualizations(ship_results, os.path.splitext(output_file)[0])
    
    return results


def create_tf_visualizations(
    ship_results: List[Dict[str, Any]],
    output_prefix: str
) -> None:
    """
    Create visualizations of time-frequency analysis results.
    
    Parameters
    ----------
    ship_results : List[Dict[str, Any]]
        List of time-frequency analysis results per ship
    output_prefix : str
        Prefix for output file paths
    """
    for i, ship in enumerate(ship_results):
        # Skip if no data available
        if 'dominant_frequencies' not in ship or 'dominant_amplitudes' not in ship:
            continue
        
        # Create frequency map visualization
        dominant_freqs = ship['dominant_frequencies']
        dominant_amps = ship['dominant_amplitudes']
        
        height, width = dominant_freqs.shape
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create valid mask
        valid_mask = dominant_amps > 0
        
        # Create normalized colormap for frequencies
        if np.any(valid_mask):
            vmin = np.min(dominant_freqs[valid_mask])
            vmax = np.max(dominant_freqs[valid_mask])
            
            # Create frequency heatmap
            im = ax.imshow(dominant_freqs, cmap='jet', vmin=vmin, vmax=vmax)
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Frequency (Hz)')
            
            # Set title
            ax.set_title(f"Ship {i+1} - Dominant Vibration Frequencies")
            
            # Save figure
            freq_file = f"{output_prefix}_ship{i+1}_freq_map.png"
            plt.tight_layout()
            plt.savefig(freq_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create amplitude map
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Normalize amplitudes
            norm_amps = dominant_amps / np.max(dominant_amps) if np.max(dominant_amps) > 0 else dominant_amps
            
            # Create amplitude heatmap
            im = ax.imshow(norm_amps, cmap='inferno')
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label('Normalized Amplitude')
            
            # Set title
            ax.set_title(f"Ship {i+1} - Vibration Amplitudes")
            
            # Save figure
            amp_file = f"{output_prefix}_ship{i+1}_amp_map.png"
            plt.tight_layout()
            plt.savefig(amp_file, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Check if we have STFT maps
            if 'stft_maps' in ship and ship['stft_maps'].get('computed', False):
                # Extract a representative pixel for STFT visualization
                center_y, center_x = height // 2, width // 2
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Get STFT data
                stft_mag = ship['stft_maps']['stft_magnitude'][center_y, center_x]
                freqs = ship['stft_maps']['frequencies']
                times = ship['stft_maps']['times']
                
                # Plot spectrogram
                im = ax.pcolormesh(times, freqs, 10 * np.log10(stft_mag + 1e-10), 
                                 shading='gouraud', cmap='inferno')
                
                # Add colorbar
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Power/Frequency (dB/Hz)')
                
                # Set labels
                ax.set_ylabel('Frequency [Hz]')
                ax.set_xlabel('Time [s]')
                ax.set_title(f"Ship {i+1} - Central Pixel Spectrogram")
                
                # Save figure
                stft_file = f"{output_prefix}_ship{i+1}_stft.png"
                plt.tight_layout()
                plt.savefig(stft_file, dpi=300, bbox_inches='tight')
                plt.close(fig)


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Time-Frequency Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input file (phase history data)')
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
        # Run time-frequency analysis
        result = analyze_time_frequency(
            args.input,
            args.output,
            args.use_gpu,
            logger
        )
        logger.info("Time-frequency analysis completed successfully")
    except Exception as e:
        logger.error(f"Error during time-frequency analysis: {str(e)}")
        sys.exit(1) 