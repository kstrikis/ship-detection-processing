#!/usr/bin/env python3
"""
Pixel-level phase history extraction module for improved ship micro-motion analysis.

This module implements the core improvement of the new pipeline - extracting phase history 
for each individual pixel rather than applying global shift analysis. This enables 
detailed micro-motion analysis of different parts of a ship.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from utils import (
    setup_logging, save_results, load_step_output, check_gpu_availability,
    scale_for_display
)


class PixelPhaseHistoryExtractor:
    """
    Extracts pixel-specific phase histories from SAR subapertures.
    """
    
    def __init__(
        self, 
        signal_data: np.ndarray,
        num_subapertures: int = 200,
        aperture_width_ratio: float = 0.5,
        use_gpu: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the phase history extractor.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw signal data
        num_subapertures : int, optional
            Number of subapertures to create, by default 200
        aperture_width_ratio : float, optional
            Ratio of subaperture width to full aperture width, by default 0.5
        use_gpu : bool, optional
            Whether to use GPU acceleration if available, by default False
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        """
        self.signal_data = signal_data
        self.num_subapertures = num_subapertures
        self.aperture_width_ratio = aperture_width_ratio
        self.use_gpu = use_gpu and check_gpu_availability()
        self.logger = logger or logging.getLogger(__name__)
        
        # Calculated properties
        self.subapertures = None
        self.sampling_freq = 1.0  # Normalized frequency by default
        self.rows, self.cols = signal_data.shape
        
        self.logger.info(f"Phase history extractor initialized with signal data shape {signal_data.shape}")
        if self.use_gpu:
            self.logger.info("Using GPU acceleration for phase history extraction")
    
    def create_subapertures(self) -> np.ndarray:
        """
        Create subapertures from signal data using FFT-based approach.
        
        Returns
        -------
        np.ndarray
            Array of subaperture images, shape (num_subapertures, rows, cols)
        """
        self.logger.info(f"Creating {self.num_subapertures} subapertures")
        rows, cols = self.signal_data.shape
        
        # Calculate subaperture width
        aperture_width = int(cols * self.aperture_width_ratio)
        
        # Ensure sufficient data for creating subapertures
        if aperture_width >= cols:
            self.logger.warning(f"Aperture width {aperture_width} is too large for data width {cols}")
            aperture_width = cols // 2
            self.logger.info(f"Using reduced aperture width: {aperture_width}")
        
        # Calculate step size between subapertures
        step_size = (cols - aperture_width) // (self.num_subapertures - 1) if self.num_subapertures > 1 else 1
        
        if step_size < 1:
            step_size = 1
            self.logger.warning(f"Step size too small, using minimum step size of 1")
            self.num_subapertures = cols - aperture_width + 1
            self.logger.info(f"Adjusted number of subapertures to {self.num_subapertures}")
        
        # Use GPU acceleration if available
        if self.use_gpu:
            try:
                import cupy as cp
                self.logger.info("Using GPU acceleration for subaperture creation")
                
                # Transfer signal data to GPU
                gpu_signal = cp.asarray(self.signal_data)
                
                # Allocate memory for subapertures
                gpu_subapertures = cp.zeros((self.num_subapertures, rows, cols), dtype=cp.complex128)
                
                # Process each subaperture
                for i in range(self.num_subapertures):
                    start_col = i * step_size
                    end_col = start_col + aperture_width
                    
                    if end_col > cols:
                        break
                    
                    # Create window function (Hamming)
                    window = cp.hamming(aperture_width).reshape(1, -1)  # Make 2D: 1 x aperture_width
                    
                    # Extract and apply window
                    subaperture = cp.zeros((rows, cols), dtype=cp.complex128)
                    subaperture[:, start_col:end_col] = gpu_signal[:, start_col:end_col] * window
                    
                    # Focus subaperture
                    gpu_subapertures[i] = cp.fft.fftshift(cp.fft.fft2(subaperture))
                
                # Transfer result back to CPU
                self.subapertures = cp.asnumpy(gpu_subapertures)
                
            except ImportError:
                self.logger.warning("CuPy not available, falling back to CPU implementation")
                self.subapertures = self._cpu_create_subapertures(aperture_width, step_size)
        else:
            # Use CPU implementation with parallel processing
            self.subapertures = self._cpu_create_subapertures(aperture_width, step_size)
        
        self.logger.info(f"Created {self.num_subapertures} subapertures with shape {self.subapertures.shape}")
        
        return self.subapertures
    
    def _cpu_create_subapertures(self, aperture_width: int, step_size: int) -> np.ndarray:
        """
        Create subapertures using CPU with parallel processing.
        
        Parameters
        ----------
        aperture_width : int
            Width of each subaperture
        step_size : int
            Step size between consecutive subapertures
            
        Returns
        -------
        np.ndarray
            Array of subaperture images
        """
        rows, cols = self.signal_data.shape
        subapertures = np.zeros((self.num_subapertures, rows, cols), dtype=complex)
        
        # Process subapertures in parallel
        def process_subaperture(i):
            start_col = i * step_size
            end_col = start_col + aperture_width
            
            if end_col > cols:
                return None
            
            # Create window function (Hamming)
            window = np.hamming(aperture_width).reshape(1, -1)  # Make 2D: 1 x aperture_width
            
            # Extract subaperture
            subaperture = np.zeros((rows, cols), dtype=complex)
            subaperture[:, start_col:end_col] = self.signal_data[:, start_col:end_col] * window
            
            # Focus subaperture
            return np.fft.fftshift(np.fft.fft2(subaperture))
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_subaperture, i) for i in range(self.num_subapertures)]
            
            for i, future in enumerate(futures):
                result = future.result()
                if result is not None:
                    subapertures[i] = result
        
        return subapertures
    
    def extract_pixel_phase_history(
        self, 
        pixel_row: int, 
        pixel_col: int
    ) -> np.ndarray:
        """
        Extract phase history for a specific pixel across all subapertures.
        
        Parameters
        ----------
        pixel_row : int
            Pixel row coordinate (y)
        pixel_col : int
            Pixel column coordinate (x)
            
        Returns
        -------
        np.ndarray
            Time series of phase values
        """
        if self.subapertures is None:
            self.create_subapertures()
        
        # Extract complex values for this pixel across all subapertures
        complex_values = self.subapertures[:, pixel_row, pixel_col]
        
        # Extract phase information
        phase_history = np.angle(complex_values)
        
        # Unwrap phase to avoid discontinuities
        phase_history = np.unwrap(phase_history)
        
        return phase_history
    
    def extract_ship_phase_histories(
        self, 
        ship_region: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        Extract phase histories for all pixels in a ship region.
        
        Parameters
        ----------
        ship_region : Dict[str, Any]
            Ship region dictionary with bbox, mask, etc.
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with phase history arrays
        """
        # Get ship bounding box
        x1, y1, x2, y2 = ship_region['bbox']
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        
        self.logger.info(f"Extracting phase histories for ship region at ({x1},{y1})-({x2},{y2})")
        
        # Ensure subapertures are created
        if self.subapertures is None:
            self.create_subapertures()
        
        # Check if GPU acceleration is available and requested
        if self.use_gpu:
            try:
                import cupy as cp
                self.logger.info("Using GPU acceleration for phase history extraction")
                
                # Transfer subapertures to GPU
                gpu_subapertures = cp.asarray(self.subapertures)
                
                # Extract region for all subapertures at once
                # Shape: num_subapertures x height x width
                region_subapertures = gpu_subapertures[:, y1:y2+1, x1:x2+1]
                
                # Extract phase histories
                # Shape: num_subapertures x height x width
                complex_values = region_subapertures
                phase_histories = cp.angle(complex_values)
                
                # Unwrap phase along time dimension (axis 0)
                # This is more complex on GPU, we might need to do it on CPU
                # For now, transfer back to CPU for unwrapping
                phase_histories_cpu = cp.asnumpy(phase_histories)
                
                # Create output array
                unwrapped_histories = np.zeros_like(phase_histories_cpu)
                
                # Unwrap each pixel's phase history
                for i in range(height):
                    for j in range(width):
                        unwrapped_histories[:, i, j] = np.unwrap(phase_histories_cpu[:, i, j])
                
                # Transpose to get height x width x time (subapertures)
                # This is the expected format for further processing
                phase_histories_final = np.transpose(unwrapped_histories, (1, 2, 0))
                
            except ImportError:
                self.logger.warning("CuPy not available, falling back to CPU implementation")
                phase_histories_final = self._cpu_extract_ship_phase_histories(x1, y1, x2, y2)
        else:
            # Use CPU implementation
            phase_histories_final = self._cpu_extract_ship_phase_histories(x1, y1, x2, y2)
        
        self.logger.info(f"Extracted phase histories with shape {phase_histories_final.shape}")
        
        return {
            'phase_histories': phase_histories_final,
            'ship_bbox': ship_region['bbox'],
            'dimensions': (height, width),
            'num_subapertures': self.num_subapertures,
            'sampling_freq': self.sampling_freq
        }
    
    def _cpu_extract_ship_phase_histories(
        self, 
        x1: int, 
        y1: int, 
        x2: int, 
        y2: int
    ) -> np.ndarray:
        """
        Extract phase histories using CPU implementation.
        
        Parameters
        ----------
        x1, y1, x2, y2 : int
            Ship region bounding box coordinates
            
        Returns
        -------
        np.ndarray
            Array of phase histories, shape (height, width, num_subapertures)
        """
        height = y2 - y1 + 1
        width = x2 - x1 + 1
        
        # Create phase history matrix: height x width x num_subapertures
        phase_histories = np.zeros((height, width, self.num_subapertures))
        
        # Vectorized extraction for efficiency
        # First, extract the complex values for the entire region
        # Shape: num_subapertures x height x width
        region_complex = self.subapertures[:, y1:y2+1, x1:x2+1]
        
        # Convert to phase
        # Shape: num_subapertures x height x width
        phase_values = np.angle(region_complex)
        
        # Transpose to get height x width x num_subapertures
        phase_values = np.transpose(phase_values, (1, 2, 0))
        
        # Unwrap each pixel's phase history
        for i in range(height):
            for j in range(width):
                phase_histories[i, j, :] = np.unwrap(phase_values[i, j, :])
        
        return phase_histories


def extract_phase_history(
    input_file: str,
    ship_file: str,
    output_file: str,
    num_subapertures: int = 200,
    aperture_width_ratio: float = 0.5,
    use_gpu: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Extract phase histories for all ships in a preprocessed SAR image.
    
    Parameters
    ----------
    input_file : str
        Path to input file (preprocessed SAR data)
    ship_file : str
        Path to ship detection results
    output_file : str
        Path to output file
    num_subapertures : int, optional
        Number of subapertures to create, by default 200
    aperture_width_ratio : float, optional
        Ratio of subaperture width to full aperture width, by default 0.5
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing phase extraction results
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Extracting phase histories for ships in {ship_file} using data from {input_file}")
    
    # Load preprocessed data
    preprocessed = load_step_output(input_file)
    logger.info(f"Loaded preprocessed data with keys: {preprocessed.keys()}")
    
    # Load ship detection results
    ships = load_step_output(ship_file)
    logger.info(f"Loaded ship detection data with keys: {ships.keys()}")
    logger.info(f"Found {ships.get('num_ships', 0)} ships")
    
    # Check that we have signal data available
    if 'signal_data' not in preprocessed:
        logger.error("No signal data found in preprocessed data, required for phase extraction")
        raise ValueError("No signal data found in preprocessed data")
    
    # Get raw signal data
    signal_data = preprocessed['signal_data']
    
    # Get PVP data if available for sampling frequency calculation
    if 'pvp_data' in preprocessed:
        pvp_data = preprocessed['pvp_data']
        
        # Try to calculate sampling frequency from PVP data
        sampling_freq = 1.0  # Default normalized frequency
        
        for channel in pvp_data:
            if isinstance(pvp_data[channel], dict) and 'TxTime' in pvp_data[channel]:
                # Extract time values (assuming they are in seconds)
                time_values = pvp_data[channel]['TxTime']
                
                # Calculate sampling period and frequency
                if len(time_values) >= 2:
                    sampling_period = np.mean(np.diff(time_values))
                    sampling_freq = 1.0 / sampling_period
                    logger.info(f"Calculated sampling frequency: {sampling_freq} Hz")
                    break
    else:
        sampling_freq = 1.0
        logger.info("No PVP data available, using normalized frequency")
    
    # Initialize phase extractor
    extractor = PixelPhaseHistoryExtractor(
        signal_data=signal_data,
        num_subapertures=num_subapertures,
        aperture_width_ratio=aperture_width_ratio,
        use_gpu=use_gpu,
        logger=logger
    )
    
    # Create subapertures (this is expensive, so we only do it once)
    extractor.sampling_freq = sampling_freq
    subapertures = extractor.create_subapertures()
    
    # Process each ship
    ship_results = []
    filtered_ships = ships.get('filtered_ships', [])
    
    for i, ship in enumerate(filtered_ships):
        logger.info(f"Processing ship {i+1}/{len(filtered_ships)}")
        
        # Extract phase histories for ship region
        ship_phase_data = extractor.extract_ship_phase_histories(ship)
        
        # Add ship index for reference
        ship_phase_data['ship_index'] = i
        
        # Store results
        ship_results.append(ship_phase_data)
    
    # Create complete results dictionary
    results = {
        'ship_results': ship_results,
        'num_ships': len(filtered_ships),
        'num_subapertures': num_subapertures,
        'aperture_width_ratio': aperture_width_ratio,
        'sampling_freq': sampling_freq,
        'timestamp': preprocessed.get('timestamp', None),
        'input_file': input_file,
        'ship_file': ship_file
    }
    
    # Save results
    save_results(output_file, results)
    logger.info(f"Saved phase history extraction results to {output_file}")
    
    # Create a visualization
    create_phase_visualization(
        subapertures[0],  # Use first subaperture image for visualization
        ship_results,
        os.path.splitext(output_file)[0] + '_phase.png'
    )
    
    return results


def create_phase_visualization(
    image_data: np.ndarray, 
    ship_results: List[Dict[str, Any]],
    output_path: str
) -> None:
    """
    Create a visualization of ships with phase history extraction.
    
    Parameters
    ----------
    image_data : np.ndarray
        SAR image data
    ship_results : List[Dict[str, Any]]
        List of ship phase extraction results
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
    for i, ship in enumerate(ship_results):
        x1, y1, x2, y2 = ship['ship_bbox']
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1-5, f"Ship {i+1}", 
               color='red', fontsize=10, backgroundcolor='white')
    
    ax.set_title(f"Phase History Extraction for {len(ship_results)} Ships")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Create phase history visualization for each ship
    for i, ship in enumerate(ship_results):
        # Select central pixel for visualization
        height, width = ship['dimensions']
        center_y, center_x = height // 2, width // 2
        
        # Get phase history for central pixel
        phase_history = ship['phase_histories'][center_y, center_x, :]
        
        # Create figure for phase history
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot phase history
        ax.plot(phase_history)
        ax.set_title(f"Ship {i+1} - Central Pixel Phase History")
        ax.set_xlabel("Subaperture Index")
        ax.set_ylabel("Phase (rad)")
        ax.grid(True)
        
        # Save figure
        ship_path = os.path.splitext(output_path)[0] + f"_ship{i+1}_phase.png"
        plt.tight_layout()
        plt.savefig(ship_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Pixel-Level Phase History Extractor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input file (preprocessed SAR data)')
    parser.add_argument('--ship-file', type=str, required=True,
                      help='Ship detection results file')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--num-subapertures', type=int, default=200,
                      help='Number of subapertures to create')
    parser.add_argument('--aperture-width-ratio', type=float, default=0.5,
                      help='Ratio of subaperture width to full aperture width')
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
        # Run phase history extraction
        result = extract_phase_history(
            args.input,
            args.ship_file,
            args.output,
            args.num_subapertures,
            args.aperture_width_ratio,
            args.use_gpu,
            logger
        )
        logger.info("Phase history extraction completed successfully")
    except Exception as e:
        logger.error(f"Error during phase history extraction: {str(e)}")
        sys.exit(1) 