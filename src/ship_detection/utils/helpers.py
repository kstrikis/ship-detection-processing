"""
Helper functions for the ship detection project.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Iterator

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

# Configure logging
def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    file_level: int = logging.DEBUG
) -> logging.Logger:
    """
    Set up logging for the project.
    
    Parameters
    ----------
    log_file : Optional[str], optional
        Path to the log file, by default None
    level : int, optional
        Logging level for console output, by default logging.INFO
    file_level : int, optional
        Logging level for file output, by default logging.DEBUG
        
    Returns
    -------
    logging.Logger
        Configured logger.
    """
    # Create logger
    logger = logging.getLogger('ship_detection')
    logger.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def save_results(
    output_dir: str, 
    filename_base: str, 
    figures: Dict[str, plt.Figure],
    data: Dict[str, Any]
) -> List[str]:
    """
    Save results to files.
    
    Parameters
    ----------
    output_dir : str
        Directory to save results.
    filename_base : str
        Base name for output files.
    figures : Dict[str, plt.Figure]
        Dictionary of figures to save.
    data : Dict[str, Any]
        Dictionary of data to save.
        
    Returns
    -------
    List[str]
        List of saved files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []
    
    # Save figures
    for name, fig in figures.items():
        figure_path = os.path.join(output_dir, f"{filename_base}_{name}.png")
        fig.savefig(figure_path, dpi=300, bbox_inches='tight')
        saved_files.append(figure_path)
    
    # Save data as NPZ file
    data_path = os.path.join(output_dir, f"{filename_base}_data.npz")
    np.savez_compressed(data_path, **data)
    saved_files.append(data_path)
    
    return saved_files

def calculate_signal_statistics(signal_data: np.ndarray) -> Dict[str, float]:
    """
    Calculate statistics for the signal data.
    
    Parameters
    ----------
    signal_data : np.ndarray
        Input signal data.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of statistics.
    """
    if np.iscomplexobj(signal_data):
        magnitude = np.abs(signal_data)
    else:
        magnitude = signal_data
    
    return {
        'mean': np.mean(magnitude),
        'median': np.median(magnitude),
        'std': np.std(magnitude),
        'min': np.min(magnitude),
        'max': np.max(magnitude),
        'dynamic_range_db': 10 * np.log10(np.max(magnitude) / (np.min(magnitude) + 1e-10)),
    }

def sliding_window(
    image: np.ndarray, 
    window_size: Tuple[int, int], 
    step_size: Tuple[int, int]
) -> Iterator[Tuple[Tuple[int, int], np.ndarray]]:
    """
    Sliding window generator over an image.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    window_size : Tuple[int, int]
        Size of the window (height, width).
    step_size : Tuple[int, int]
        Step size (height_step, width_step).
        
    Yields
    ------
    Iterator[Tuple[Tuple[int, int], np.ndarray]]
        Iterator of (position, window) tuples.
    """
    # Get image dimensions
    height, width = image.shape[:2]
    window_height, window_width = window_size
    step_height, step_width = step_size
    
    # Iterate through positions
    for y in range(0, height - window_height + 1, step_height):
        for x in range(0, width - window_width + 1, step_width):
            yield (y, x), image[y:y+window_height, x:x+window_width]

def enhance_image_contrast(image: np.ndarray, percentile: float = 2.0) -> np.ndarray:
    """
    Enhance image contrast by clipping to percentiles.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    percentile : float, optional
        Percentile to clip, by default 2.0
        
    Returns
    -------
    np.ndarray
        Enhanced image.
    """
    # Calculate percentiles
    p_low = np.percentile(image, percentile)
    p_high = np.percentile(image, 100 - percentile)
    
    # Clip the image
    clipped = np.clip(image, p_low, p_high)
    
    # Normalize to [0, 1]
    normalized = (clipped - p_low) / (p_high - p_low)
    
    return normalized

def align_images(reference: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Align target image to reference using cross-correlation.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference image.
    target : np.ndarray
        Target image to be aligned.
        
    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int]]
        Aligned image and shift values (y_shift, x_shift).
    """
    # Ensure both images have the same dimensions
    if reference.shape != target.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # Compute cross-correlation
    correlation = ndimage.correlate(reference, target, mode='constant')
    
    # Find the peak in the correlation
    y_shift, x_shift = np.unravel_index(np.argmax(correlation), correlation.shape)
    
    # Center the shift
    y_shift -= reference.shape[0] // 2
    x_shift -= reference.shape[1] // 2
    
    # Shift the target image
    aligned = ndimage.shift(target, (y_shift, x_shift), mode='constant')
    
    return aligned, (y_shift, x_shift) 