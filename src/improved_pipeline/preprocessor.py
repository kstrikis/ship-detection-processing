#!/usr/bin/env python3
"""
SAR data preprocessing module for the improved ship micro-motion analysis pipeline.

This module handles:
1. Loading SAR data from various formats (CPHD, SICD, etc.)
2. Calibration and normalization
3. Speckle reduction
4. Optional cropping for region of interest
5. Saving preprocessed data for subsequent stages
"""

import os
import sys
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from utils import (
    setup_logging, save_results, load_step_output, check_gpu_availability,
    scale_for_display, apply_speckle_filter, downsample_image
)

# Try to import sarpy for SAR data reading
try:
    import sarpy
    from sarpy.io.complex.sicd import SICDReader
    from sarpy.io.complex.sidd import SIDDReader
    from sarpy.io.complex.cphd import CPHDReader
    SARPY_AVAILABLE = True
except ImportError:
    SARPY_AVAILABLE = False


class SARDataReader:
    """Reader for various SAR data formats using sarpy."""
    
    def __init__(self, input_file: str):
        """
        Initialize the reader.
        
        Parameters
        ----------
        input_file : str
            Path to input SAR data file
        """
        if not SARPY_AVAILABLE:
            raise ImportError("sarpy library required for SAR data reading is not available")
        
        self.input_file = input_file
        self.reader = None
        self.open_file()
    
    def open_file(self):
        """Open the input file and initialize appropriate reader."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        
        # Try to open with various readers
        try:
            # First try CPHD
            self.reader = CPHDReader(self.input_file)
            self.data_type = 'cphd'
        except Exception:
            try:
                # Try SICD
                self.reader = SICDReader(self.input_file)
                self.data_type = 'sicd'
            except Exception:
                try:
                    # Try SIDD
                    self.reader = SIDDReader(self.input_file)
                    self.data_type = 'sidd'
                except Exception:
                    # Check if it's a NumPy file (for cropped data)
                    if self.input_file.endswith('.npy') or self.input_file.endswith('.npz'):
                        self.reader = None
                        self.data_type = 'numpy'
                    else:
                        raise ValueError(f"Unsupported file format: {self.input_file}")
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata from the SAR data file.
        
        Returns
        -------
        Dict[str, Any]
            Metadata dictionary
        """
        if self.data_type == 'numpy':
            # For NumPy files, we'll handle metadata separately
            return {'type': 'numpy'}
        
        if hasattr(self.reader, 'cphd_meta'):
            return {'type': 'cphd', 'meta': self.reader.cphd_meta}
        elif hasattr(self.reader, 'sicd_meta'):
            return {'type': 'sicd', 'meta': self.reader.sicd_meta}
        elif hasattr(self.reader, 'sidd_meta'):
            return {'type': 'sidd', 'meta': self.reader.sidd_meta}
        else:
            return {'type': 'unknown'}
    
    def read_cphd_signal_data(self) -> np.ndarray:
        """
        Read CPHD signal data.
        
        Returns
        -------
        np.ndarray
            Signal data array
        """
        if self.data_type != 'cphd':
            raise ValueError("Not a CPHD file")
        
        # Read signal data from CPHD
        signal_data = self.reader.read_signal_block()
        return signal_data
    
    def read_pvp_data(self) -> Dict[str, np.ndarray]:
        """
        Read per-vector parameters (PVP) data from CPHD.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary of PVP data
        """
        if self.data_type != 'cphd':
            raise ValueError("Not a CPHD file")
        
        pvp_data = {}
        for channel_id in self.reader.cphd_meta.Data.Channels:
            pvp_data[channel_id] = {}
            for param_name in self.reader.cphd_meta.PVP.get_parameter_names():
                try:
                    param_data = self.reader.read_pvp_block(param_name, channel_id=channel_id)
                    pvp_data[channel_id][param_name] = param_data
                except Exception as e:
                    # Some parameters might not be available
                    continue
        
        return pvp_data
    
    def read_sicd_data(self) -> np.ndarray:
        """
        Read SICD complex image data.
        
        Returns
        -------
        np.ndarray
            Complex image data array
        """
        if self.data_type != 'sicd':
            raise ValueError("Not a SICD file")
        
        # Read full resolution complex image data
        image_data = self.reader.read_chip()
        return image_data
    
    def close(self):
        """Close the reader and free resources."""
        self.reader = None


def load_cropped_data(file_path: str) -> Dict[str, Any]:
    """
    Load previously cropped SAR data.
    
    Parameters
    ----------
    file_path : str
        Path to cropped data file
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded data
    """
    # Check if it's a NumPy file
    if file_path.endswith('.npy'):
        # .npy files store a single array, but we expect a dict
        try:
            data = np.load(file_path, allow_pickle=True).item()
            return data
        except:
            raise ValueError(f"Invalid .npy file format: {file_path}")
    elif file_path.endswith('.npz'):
        # .npz files store multiple arrays
        try:
            data = dict(np.load(file_path, allow_pickle=True))
            return data
        except:
            raise ValueError(f"Invalid .npz file format: {file_path}")
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def crop_and_save_cphd(
    focused_image: np.ndarray, 
    signal_data: np.ndarray, 
    pvp_data: Dict[str, np.ndarray], 
    metadata: Any,
    output_file: str,
    logger: Optional[logging.Logger] = None
) -> str:
    """
    Display focused image and allow user to select a region to crop.
    Then save the cropped CPHD data to a new file.
    
    Parameters
    ----------
    focused_image : np.ndarray
        Focused SAR image for visualization
    signal_data : np.ndarray
        Original CPHD signal data
    pvp_data : Dict[str, np.ndarray]
        Original PVP data
    metadata : Any
        Original metadata
    output_file : str
        Path to output file
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    str
        Path to the cropped output file
    """
    if logger is None:
        logger = logging.getLogger('ship_micromotion')
    
    logger.info("Starting CPHD cropping process...")
    
    # Scale image data for better visualization
    display_data = scale_for_display(focused_image)
    
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
        logger.info(f"Selected region at ({x1},{y1})-({x2},{y2})")
    
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
        logger.info("Cropping cancelled by user")
        return None
    
    # Get region boundaries
    x1, y1 = selected_region['x1'], selected_region['y1']
    x2, y2 = selected_region['x2'], selected_region['y2']
    
    # Crop the signal data
    cropped_signal_data = signal_data[y1:y2+1, x1:x2+1]
    
    # Filter PVP data as needed (depends on the exact format)
    cropped_pvp_data = {}
    for key, value in pvp_data.items():
        if isinstance(value, dict):
            cropped_pvp_data[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    if subvalue.shape[0] == signal_data.shape[0]:  # If PVP data is per row
                        cropped_pvp_data[key][subkey] = subvalue[y1:y2+1]
                    else:
                        # Just copy if we don't know how to crop
                        cropped_pvp_data[key][subkey] = subvalue
                else:
                    cropped_pvp_data[key][subkey] = subvalue
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
    
    # Create the focused image from cropped signal data
    cropped_focused_image = np.fft.fftshift(np.fft.fft2(cropped_signal_data))
    
    # Data to save
    data_to_save = {
        'signal_data': cropped_signal_data,
        'pvp_data': cropped_pvp_data,
        'focused_image': cropped_focused_image,
        'crop_info': crop_info,
        'crop_region': selected_region,
        'type': 'cphd',
        'timestamp': datetime.datetime.now().isoformat()
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
        logger.warning(f"Could not serialize metadata: {str(e)}")
    
    # Save data
    np.savez_compressed(output_file, **data_to_save)
    logger.info(f"Cropped CPHD data saved to {output_file}")
    
    # Also save a preview image
    preview_path = os.path.splitext(output_file)[0] + '_preview.png'
    cropped_display = scale_for_display(cropped_focused_image)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cropped_display, cmap='gray')
    plt.colorbar(label='Normalized Amplitude (dB)')
    plt.title(f"Cropped Region ({x1},{y1}) to ({x2},{y2})")
    plt.tight_layout()
    plt.savefig(preview_path)
    plt.close()
    
    logger.info(f"Preview image saved to {preview_path}")
    
    return output_file


def preprocess_sar_data(
    input_file: str,
    output_file: str,
    speckle_filter_size: int = 5,
    use_gpu: bool = False,
    crop_only: bool = False,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Preprocess SAR data: load, calibrate, filter, and optionally crop.
    
    Parameters
    ----------
    input_file : str
        Path to input SAR data file
    output_file : str
        Path to output file
    speckle_filter_size : int, optional
        Kernel size for speckle filtering (0 to disable), by default 5
    use_gpu : bool, optional
        Whether to use GPU acceleration if available, by default False
    crop_only : bool, optional
        Whether to only perform cropping, by default False
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing preprocessing results
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Preprocessing SAR data from {input_file}")
    
    # Check if GPU acceleration is requested and available
    if use_gpu:
        gpu_available = check_gpu_availability()
        if gpu_available:
            logger.info("GPU acceleration is available and will be used")
        else:
            logger.warning("GPU acceleration requested but not available, using CPU")
            use_gpu = False
    
    # Check if the file is one of our previously saved NumPy files
    if input_file.endswith('.npy') or input_file.endswith('.npz'):
        logger.info("Detected NumPy file, loading directly...")
        try:
            # Load already processed or cropped data
            read_results = load_cropped_data(input_file)
            logger.info(f"Loaded NumPy data with keys: {read_results.keys()}")
            
            # If we're only doing cropping and this is already a cropped file, we're done
            if crop_only and 'crop_region' in read_results:
                logger.info("File is already cropped, no further cropping needed")
                # Just copy to the output file
                if input_file != output_file:
                    import shutil
                    shutil.copy2(input_file, output_file)
                    logger.info(f"Copied to {output_file}")
                return read_results
            
            # If already preprocessed and not doing crop-only, we can use it directly
            if not crop_only and 'focused_image' in read_results:
                # Apply speckle filter if requested
                if speckle_filter_size > 0:
                    logger.info(f"Applying speckle filter with kernel size {speckle_filter_size}")
                    focused_image = read_results['focused_image']
                    if use_gpu:
                        # Use GPU for filtering
                        try:
                            import cupy as cp
                            # Transfer to GPU
                            gpu_image = cp.asarray(focused_image)
                            # Apply filter (need to implement on GPU)
                            # For now, transfer back to CPU for filtering
                            filtered_image = apply_speckle_filter(
                                cp.asnumpy(gpu_image), speckle_filter_size)
                            # No need to transfer back since we're already on CPU
                            read_results['filtered_image'] = filtered_image
                        except ImportError:
                            logger.warning("CuPy not available, falling back to CPU filtering")
                            read_results['filtered_image'] = apply_speckle_filter(
                                focused_image, speckle_filter_size)
                    else:
                        # Use CPU for filtering
                        read_results['filtered_image'] = apply_speckle_filter(
                            focused_image, speckle_filter_size)
                
                # Create downsampled preview for ship detection
                logger.info("Creating downsampled preview for ship detection")
                if 'filtered_image' in read_results:
                    source_image = read_results['filtered_image']
                else:
                    source_image = read_results['focused_image']
                    
                preview_factor = 4  # Default downsampling factor
                read_results['preview_image'] = downsample_image(source_image, preview_factor)
                read_results['preview_factor'] = preview_factor
                read_results['original_shape'] = source_image.shape
                read_results['preview_shape'] = read_results['preview_image'].shape
                
                # Save preprocessed results
                save_results(output_file, read_results)
                logger.info(f"Saved preprocessed data to {output_file}")
                
                return read_results
        except Exception as e:
            logger.error(f"Error loading NumPy data: {str(e)}")
            raise
    
    # For new data files, use SARDataReader
    logger.info("Initializing SAR data reader")
    reader = SARDataReader(input_file)
    
    # Get metadata
    metadata = reader.get_metadata()
    logger.info(f"Detected data type: {metadata['type']}")
    
    # For CPHD data
    if metadata['type'] == 'cphd':
        logger.info("Processing CPHD data")
        
        # Read signal data
        signal_data = reader.read_cphd_signal_data()
        
        # Read PVP data
        pvp_data = reader.read_pvp_data()
        
        # Convert signal data to complex image through basic focusing
        focused_image = np.fft.fftshift(np.fft.fft2(signal_data))
        
        # Check if we're only cropping
        if crop_only:
            logger.info("Crop-only mode activated")
            return crop_and_save_cphd(
                focused_image, signal_data, pvp_data, metadata['meta'], output_file, logger
            )
        
        # Apply speckle filter if requested
        if speckle_filter_size > 0:
            logger.info(f"Applying speckle filter with kernel size {speckle_filter_size}")
            filtered_image = apply_speckle_filter(focused_image, speckle_filter_size)
        else:
            filtered_image = focused_image
        
        # Create downsampled preview for ship detection
        logger.info("Creating downsampled preview for ship detection")
        preview_factor = 4  # Default downsampling factor
        preview_image = downsample_image(filtered_image, preview_factor)
        
        # Create results dictionary
        results = {
            'type': 'cphd',
            'metadata': metadata,
            'signal_data': signal_data,
            'pvp_data': pvp_data,
            'focused_image': focused_image,
            'filtered_image': filtered_image,
            'preview_image': preview_image,
            'preview_factor': preview_factor,
            'original_shape': focused_image.shape,
            'preview_shape': preview_image.shape,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    # For SICD data
    elif metadata['type'] == 'sicd':
        logger.info("Processing SICD data")
        
        # Read complex image data
        image_data = reader.read_sicd_data()
        
        # For SICD, the image is already focused
        focused_image = image_data
        
        # Apply speckle filter if requested
        if speckle_filter_size > 0:
            logger.info(f"Applying speckle filter with kernel size {speckle_filter_size}")
            filtered_image = apply_speckle_filter(focused_image, speckle_filter_size)
        else:
            filtered_image = focused_image
        
        # Create downsampled preview for ship detection
        logger.info("Creating downsampled preview for ship detection")
        preview_factor = 4  # Default downsampling factor
        preview_image = downsample_image(filtered_image, preview_factor)
        
        # Create results dictionary
        results = {
            'type': 'sicd',
            'metadata': metadata,
            'focused_image': focused_image,
            'filtered_image': filtered_image,
            'preview_image': preview_image,
            'preview_factor': preview_factor,
            'original_shape': focused_image.shape,
            'preview_shape': preview_image.shape,
            'timestamp': datetime.datetime.now().isoformat()
        }
    
    else:
        logger.error(f"Unsupported data type: {metadata['type']}")
        raise ValueError(f"Unsupported data type: {metadata['type']}")
    
    # Close reader
    reader.close()
    
    # Save results
    save_results(output_file, results)
    logger.info(f"Saved preprocessed data to {output_file}")
    
    return results


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SAR Data Preprocessor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input SAR data file')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--speckle-filter-size', type=int, default=5,
                      help='Kernel size for speckle filtering (0 to disable)')
    parser.add_argument('--use-gpu', action='store_true',
                      help='Use GPU acceleration if available')
    parser.add_argument('--crop-only', action='store_true',
                      help='Only perform data cropping')
    parser.add_argument('--log-file', type=str,
                      help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    try:
        # Run preprocessing
        result = preprocess_sar_data(
            args.input,
            args.output,
            args.speckle_filter_size,
            args.use_gpu,
            args.crop_only,
            logger
        )
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        sys.exit(1) 