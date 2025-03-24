#!/usr/bin/env python3
"""
Utility functions for the improved ship micro-motion analysis pipeline.
Contains common functionality such as logging setup, file I/O, and data handling.
"""

import os
import sys
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple


def setup_logging(log_file: Optional[str] = None, log_level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration.
    
    Parameters
    ----------
    log_file : Optional[str], optional
        Path to log file, by default None (console only)
    log_level : str, optional
        Logging level, by default 'INFO'
        
    Returns
    -------
    logging.Logger
        Configured logger object
    """
    # Create logger
    logger = logging.getLogger('ship_micromotion')
    logger.setLevel(getattr(logging, log_level))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"Logging initialized at level {log_level}")
    
    return logger


def save_results(output_file: str, data: Dict[str, Any], compress: bool = True) -> str:
    """
    Save processing results to file.
    
    Parameters
    ----------
    output_file : str
        Path to output file
    data : Dict[str, Any]
        Data to save
    compress : bool, optional
        Whether to use compression, by default True
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save data to .npz file
    if compress:
        np.savez_compressed(output_file, **data)
    else:
        np.savez(output_file, **data)
    
    return output_file


def save_figure(fig: plt.Figure, output_path: str, dpi: int = 300) -> str:
    """
    Save matplotlib figure to file.
    
    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure
    output_path : str
        Path to output file
    dpi : int, optional
        Resolution in dots per inch, by default 300
        
    Returns
    -------
    str
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


def load_step_output(file_path: str) -> Dict[str, Any]:
    """
    Load output data from a previous pipeline step.
    
    Parameters
    ----------
    file_path : str
        Path to input file
        
    Returns
    -------
    Dict[str, Any]
        Loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    # Load .npz file
    data = np.load(file_path, allow_pickle=True)
    
    # Convert to dictionary
    result = {}
    for key in data.files:
        result[key] = data[key]
    
    return result


def check_gpu_availability() -> bool:
    """
    Check if GPU acceleration is available.
    
    Returns
    -------
    bool
        True if GPU acceleration is available, False otherwise
    """
    try:
        import cupy
        return True
    except ImportError:
        return False


def scale_for_display(image_data: np.ndarray) -> np.ndarray:
    """
    Scale complex image data for better visualization.
    
    Parameters
    ----------
    image_data : np.ndarray
        Complex image data
        
    Returns
    -------
    np.ndarray
        Scaled image data suitable for display
    """
    # Take absolute value (magnitude)
    display_data = np.abs(image_data)
    
    # Convert to dB scale (log scale)
    display_data = 20 * np.log10(display_data / np.max(display_data) + 1e-10)
    
    # Clip and normalize to [0, 1]
    display_data = np.clip(display_data, -50, 0)
    display_data = (display_data + 50) / 50
    
    return display_data


def apply_speckle_filter(image_data: np.ndarray, filter_size: int = 5) -> np.ndarray:
    """
    Apply speckle reduction filter while preserving edges.
    
    Parameters
    ----------
    image_data : np.ndarray
        SAR image data
    filter_size : int, optional
        Size of filter kernel, by default 5
        
    Returns
    -------
    np.ndarray
        Filtered image
    """
    from scipy import ndimage
    
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


def downsample_image(image_data: np.ndarray, factor: int = 4) -> np.ndarray:
    """
    Create a downsampled version of the image data for faster initial processing.
    
    Parameters
    ----------
    image_data : np.ndarray
        Original image data
    factor : int, optional
        Downsampling factor, by default 4
        
    Returns
    -------
    np.ndarray
        Downsampled image
    """
    if factor <= 1:
        return image_data
        
    rows, cols = image_data.shape
    
    # Calculate new dimensions
    new_rows = rows // factor
    new_cols = cols // factor
    
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


def scale_coordinates(
    coords: Tuple[int, int, int, int], 
    original_shape: Tuple[int, int], 
    downsampled_shape: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Scale coordinates from downsampled image back to original image.
    
    Parameters
    ----------
    coords : Tuple[int, int, int, int]
        Coordinates in downsampled image (x1, y1, x2, y2)
    original_shape : Tuple[int, int]
        Shape of original image (rows, cols)
    downsampled_shape : Tuple[int, int]
        Shape of downsampled image (rows, cols)
        
    Returns
    -------
    Tuple[int, int, int, int]
        Scaled coordinates in original image
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