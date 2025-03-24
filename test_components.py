"""
Test script to diagnose each component of the ship detection pipeline.
This script tests each component individually and visualizes the results.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Make sure we can import from src
sys.path.append(str(Path(__file__).parent))

from src.ship_detection.io.readers import SARDataReader
from src.ship_detection.processing.ship_detector import ShipDetector
from src.ship_detection.processing.doppler_subaperture import DopplerSubapertureProcessor
from src.ship_detection.visualization.heatmaps import VibrationHeatmapVisualizer

def log_time(message):
    """Log a message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def test_reader(input_file, skip_large_operations=False):
    """Test the SAR data reader component."""
    log_time(f"\n===== Testing SARDataReader with {input_file} =====")
    
    reader = SARDataReader(input_file)
    metadata = reader.get_metadata()
    
    log_time(f"File type: {reader.extension}")
    log_time(f"Reader type: {type(reader.reader)}")
    
    # Print some basic metadata
    log_time("\nMetadata summary:")
    if hasattr(reader.reader, 'cphd_meta'):
        log_time("CPHD metadata available")
        if hasattr(metadata, 'CollectionInfo'):
            log_time(f"Collection info: {metadata.CollectionInfo.CoreName}")
    elif hasattr(reader.reader, 'sicd_meta'):
        log_time("SICD metadata available")
        if hasattr(metadata, 'CollectionInfo'):
            log_time(f"Collection info: {metadata.CollectionInfo.CoreName}")
    
    # Try to read data
    try:
        if hasattr(reader.reader, 'cphd_meta'):
            log_time("\nReading CPHD signal data...")
            start_time = time.time()
            
            if skip_large_operations:
                log_time("SKIPPING large signal data read operation (use --full to include)")
                signal_data = None
                pvp_data = None
            else:
                signal_data = reader.read_cphd_signal_data()
                read_time = time.time() - start_time
                log_time(f"Signal data read completed in {read_time:.2f} seconds")
                log_time(f"Signal data shape: {signal_data.shape}")
                log_time(f"Signal data type: {signal_data.dtype}")
                
                # Display signal data statistics
                magnitude = np.abs(signal_data)
                log_time(f"Signal magnitude min: {np.min(magnitude)}")
                log_time(f"Signal magnitude max: {np.max(magnitude)}")
                log_time(f"Signal magnitude mean: {np.mean(magnitude)}")
                
                # Only plot a smaller subsection for visualization
                log_time("Creating signal data visualization (downsampled)...")
                plt.figure(figsize=(10, 8))
                # Downsample for visualization
                downsample_factor = max(1, signal_data.shape[0] // 1000, signal_data.shape[1] // 1000)
                downsampled = signal_data[::downsample_factor, ::downsample_factor]
                plt.imshow(np.log10(np.abs(downsampled) + 1), cmap='viridis')
                plt.colorbar(label='Log Magnitude')
                plt.title('CPHD Signal Data (Log Magnitude, Downsampled)')
                plt.savefig('signal_data.png')
                plt.close()
                log_time("Saved signal data visualization to signal_data.png")
                
                # Read PVP data
                log_time("\nReading PVP data...")
                start_time = time.time()
                pvp_data = reader.read_pvp_data()
                read_time = time.time() - start_time
                log_time(f"PVP data read completed in {read_time:.2f} seconds")
                log_time(f"PVP data keys: {list(pvp_data.keys())}")
                for key, value in pvp_data.items():
                    if isinstance(value, np.ndarray):
                        log_time(f"  {key} shape: {value.shape}")
            
            return signal_data, pvp_data, metadata
            
        elif hasattr(reader.reader, 'sicd_meta'):
            log_time("\nReading SICD data...")
            start_time = time.time()
            image_data = reader.read_sicd_data()
            read_time = time.time() - start_time
            log_time(f"Image data read completed in {read_time:.2f} seconds")
            log_time(f"Image data shape: {image_data.shape}")
            log_time(f"Image data type: {image_data.dtype}")
            
            # Display image data statistics
            magnitude = np.abs(image_data)
            log_time(f"Image magnitude min: {np.min(magnitude)}")
            log_time(f"Image magnitude max: {np.max(magnitude)}")
            log_time(f"Image magnitude mean: {np.mean(magnitude)}")
            
            # Create a simple visualization of the image data
            log_time("Creating image data visualization...")
            plt.figure(figsize=(10, 8))
            plt.imshow(np.log10(magnitude + 1), cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.title('SICD Image Data (Log Magnitude)')
            plt.savefig('image_data.png')
            plt.close()
            log_time("Saved image data visualization to image_data.png")
            
            return image_data, None, metadata
            
    except Exception as e:
        log_time(f"Error reading data: {str(e)}")
        import traceback
        traceback.print_exc()
    
    reader.close()
    return None, None, metadata

def test_ship_detector(image_data, skip_large_operations=False):
    """Test the ship detector component."""
    log_time("\n===== Testing ShipDetector =====")
    
    if image_data is None:
        log_time("Image data is None. Skipping ship detection.")
        return None
        
    if skip_large_operations and (image_data.shape[0] > 5000 or image_data.shape[1] > 5000):
        log_time(f"Image too large ({image_data.shape}). Using downsampled version for detection.")
        # Determine downsampling factor
        downsample_factor = max(1, image_data.shape[0] // 5000, image_data.shape[1] // 5000)
        log_time(f"Downsampling by factor of {downsample_factor}")
        
        # Downsample the image
        if np.iscomplexobj(image_data):
            downsampled = image_data[::downsample_factor, ::downsample_factor]
        else:
            downsampled = image_data[::downsample_factor, ::downsample_factor]
            
        image_data = downsampled
    
    # Initialize ship detector
    log_time("Initializing ship detector...")
    start_time = time.time()
    ship_detector = ShipDetector(image_data)
    init_time = time.time() - start_time
    log_time(f"Initialization completed in {init_time:.2f} seconds")
    
    # Test preprocessing
    log_time("Preprocessing image data...")
    start_time = time.time()
    preprocessed = ship_detector.preprocess()
    preprocess_time = time.time() - start_time
    log_time(f"Preprocessing completed in {preprocess_time:.2f} seconds")
    log_time(f"Preprocessed data shape: {preprocessed.shape}")
    log_time(f"Preprocessed data min: {np.min(preprocessed)}")
    log_time(f"Preprocessed data max: {np.max(preprocessed)}")
    
    # Visualize preprocessed data
    log_time("Creating preprocessed data visualization...")
    plt.figure(figsize=(10, 8))
    plt.imshow(preprocessed, cmap='gray')
    plt.colorbar(label='Normalized Value')
    plt.title('Preprocessed Image Data')
    plt.savefig('preprocessed_data.png')
    plt.close()
    log_time("Saved preprocessed data visualization to preprocessed_data.png")
    
    # Test ship detection with adaptive thresholding
    log_time("\nDetecting ships using adaptive thresholding...")
    start_time = time.time()
    ship_mask = ship_detector.detect_ships_by_adaptive_threshold(sensitivity=0.7, min_area=20)
    detection_time = time.time() - start_time
    log_time(f"Ship detection completed in {detection_time:.2f} seconds")
    log_time(f"Ship mask shape: {ship_mask.shape}")
    log_time(f"Number of ship candidates: {np.sum(ship_mask)}")
    
    # Visualize ship mask
    log_time("Creating ship mask visualization...")
    plt.figure(figsize=(10, 8))
    plt.imshow(preprocessed, cmap='gray')
    plt.imshow(ship_mask, cmap='Reds', alpha=0.5)
    plt.title('Ship Detection Results')
    plt.savefig('ship_mask.png')
    plt.close()
    log_time("Saved ship mask visualization to ship_mask.png")
    
    # Extract ship regions
    log_time("\nExtracting ship regions...")
    start_time = time.time()
    ship_regions = ship_detector.extract_ship_regions(ship_mask, padding=10)
    extraction_time = time.time() - start_time
    log_time(f"Region extraction completed in {extraction_time:.2f} seconds")
    log_time(f"Extracted {len(ship_regions)} ship regions")
    
    # Filter ships by size
    log_time("\nFiltering ships by size...")
    start_time = time.time()
    filtered_ships = ship_detector.filter_ships_by_size(min_area=50)
    filter_time = time.time() - start_time
    log_time(f"Ship filtering completed in {filter_time:.2f} seconds")
    log_time(f"Filtered to {len(filtered_ships)} ship regions")
    
    # Visualize ship regions
    if len(filtered_ships) > 0:
        log_time("Creating ship regions visualization...")
        fig, axes = plt.subplots(1, min(len(filtered_ships), 3), figsize=(15, 5))
        if len(filtered_ships) == 1:
            axes = [axes]
        
        for i, (ax, ship) in enumerate(zip(axes, filtered_ships[:3])):
            region = ship['region']
            mask = ship['mask']
            
            # Display the ship region
            if np.iscomplexobj(region):
                display_region = np.log10(np.abs(region) + 1)
            else:
                display_region = region
            
            # Normalize for display
            display_region = (display_region - np.min(display_region)) / (np.max(display_region) - np.min(display_region))
            
            ax.imshow(display_region, cmap='gray')
            ax.imshow(mask, cmap='Reds', alpha=0.5)
            ax.set_title(f'Ship {i}')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            
        plt.tight_layout()
        plt.savefig('ship_regions.png')
        plt.close()
        log_time("Saved ship regions visualization to ship_regions.png")
    
    log_time("\nRunning complete ship detection pipeline...")
    start_time = time.time()
    results = ship_detector.process_all()
    process_time = time.time() - start_time
    log_time(f"Ship detection pipeline completed in {process_time:.2f} seconds")
    
    return results

def test_doppler_subaperture(signal_data, pvp_data, skip_large_operations=False):
    """Test the Doppler subaperture processor component."""
    log_time("\n===== Testing DopplerSubapertureProcessor =====")
    
    if signal_data is None or pvp_data is None:
        log_time("Signal data or PVP data is None. Skipping Doppler subaperture processing.")
        return None
    
    # Check if we should skip or downsample large operations
    if skip_large_operations:
        log_time("SKIPPING Doppler subaperture processing (use --full to include)")
        
        # Return simulated vibration results for testing visualization
        return {
            'vibration_params': {
                'times': np.linspace(0, 10, 100),
                'range_shifts': np.sin(np.linspace(0, 20*np.pi, 100)),
                'azimuth_shifts': np.cos(np.linspace(0, 20*np.pi, 100)),
                'filtered_range_shifts': np.sin(np.linspace(0, 20*np.pi, 100)),
                'filtered_azimuth_shifts': np.cos(np.linspace(0, 20*np.pi, 100)),
                'frequencies': np.linspace(0, 25, 50),
                'range_spectrum': np.zeros(50),
                'azimuth_spectrum': np.zeros(50),
                'dominant_freq_range': 15.0,
                'dominant_freq_azimuth': 12.5
            }
        }
    
    # Initialize Doppler subaperture processor
    log_time("Initializing Doppler subaperture processor...")
    start_time = time.time()
    processor = DopplerSubapertureProcessor(signal_data, pvp_data, num_subapertures=200)
    init_time = time.time() - start_time
    log_time(f"Initialization completed in {init_time:.2f} seconds")
    
    # Create subapertures
    log_time("Creating subapertures...")
    start_time = time.time()
    try:
        subapertures = processor.create_subapertures()
        create_time = time.time() - start_time
        log_time(f"Subaperture creation completed in {create_time:.2f} seconds")
        log_time(f"Created {len(subapertures)} subapertures")
        log_time(f"First subaperture shape: {subapertures[0].shape}")
    except Exception as e:
        log_time(f"Error creating subapertures: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Focus subapertures
    log_time("\nFocusing subapertures (this may take a long time)...")
    start_time = time.time()
    try:
        focused_images = processor.focus_subapertures()
        focus_time = time.time() - start_time
        log_time(f"Subaperture focusing completed in {focus_time:.2f} seconds")
        log_time(f"Created {len(focused_images)} focused images")
        log_time(f"First focused image shape: {focused_images[0].shape}")
        
        # Visualize first few focused images
        log_time("Creating focused subapertures visualization...")
        fig, axes = plt.subplots(1, min(len(focused_images), 3), figsize=(15, 5))
        for i, ax in enumerate(axes):
            idx = i * (len(focused_images) // 3)
            ax.imshow(np.log10(focused_images[idx] + 1), cmap='viridis')
            ax.set_title(f'Subaperture {idx}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('focused_subapertures.png')
        plt.close()
        log_time("Saved focused subapertures visualization to focused_subapertures.png")
    except Exception as e:
        log_time(f"Error focusing subapertures: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Compute coregistration shifts
    log_time("\nComputing coregistration shifts...")
    start_time = time.time()
    try:
        range_shifts, azimuth_shifts = processor.compute_coregistration_shifts()
        shifts_time = time.time() - start_time
        log_time(f"Coregistration shifts computation completed in {shifts_time:.2f} seconds")
        log_time(f"Range shifts shape: {range_shifts.shape}")
        log_time(f"Azimuth shifts shape: {azimuth_shifts.shape}")
        
        # Plot shifts
        log_time("Creating shifts visualization...")
        plt.figure(figsize=(10, 6))
        plt.plot(range_shifts, label='Range Shifts')
        plt.plot(azimuth_shifts, label='Azimuth Shifts')
        plt.xlabel('Subaperture Index')
        plt.ylabel('Shift (pixels)')
        plt.legend()
        plt.title('Coregistration Shifts')
        plt.grid(True)
        plt.savefig('shifts.png')
        plt.close()
        log_time("Saved shifts visualization to shifts.png")
    except Exception as e:
        log_time(f"Error computing coregistration shifts: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Estimate vibration parameters
    log_time("\nEstimating vibration parameters...")
    start_time = time.time()
    try:
        vibration_params = processor.estimate_vibration_parameters(
            range_shifts, azimuth_shifts, oversample_factor=4)
        vib_time = time.time() - start_time
        log_time(f"Vibration parameters estimation completed in {vib_time:.2f} seconds")
        log_time(f"Vibration parameters keys: {list(vibration_params.keys())}")
        
        # Plot spectra
        log_time("Creating vibration spectra visualization...")
        plt.figure(figsize=(10, 6))
        plt.plot(vibration_params['frequencies'], vibration_params['range_spectrum'], 
                label='Range Spectrum')
        plt.plot(vibration_params['frequencies'], vibration_params['azimuth_spectrum'], 
                label='Azimuth Spectrum')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.title('Vibration Spectra')
        plt.grid(True)
        plt.savefig('vibration_spectra.png')
        plt.close()
        log_time("Saved vibration spectra to vibration_spectra.png")
        
        # Print dominant frequencies
        log_time(f"Dominant range frequency: {vibration_params['dominant_freq_range']:.2f} Hz")
        log_time(f"Dominant azimuth frequency: {vibration_params['dominant_freq_azimuth']:.2f} Hz")
    except Exception as e:
        log_time(f"Error estimating vibration parameters: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    
    # Process all
    log_time("\nRunning complete vibration analysis pipeline...")
    start_time = time.time()
    try:
        results = processor.process_all()
        process_time = time.time() - start_time
        log_time(f"Complete vibration analysis completed in {process_time:.2f} seconds")
        log_time(f"Processing complete. Results keys: {list(results.keys())}")
        return results
    except Exception as e:
        log_time(f"Error in complete processing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_visualization(image_data, ship_regions, vibration_results):
    """Test the visualization component."""
    log_time("\n===== Testing VibrationHeatmapVisualizer =====")
    
    if ship_regions is None:
        log_time("No ship regions available. Skipping visualization.")
        return
    
    if not ship_regions:
        log_time("Empty ship regions list. Skipping visualization.")
        return
    
    # Initialize visualizer
    log_time("Initializing visualizer...")
    start_time = time.time()
    visualizer = VibrationHeatmapVisualizer(
        image_data, ship_regions, vibration_results)
    init_time = time.time() - start_time
    log_time(f"Visualizer initialization completed in {init_time:.2f} seconds")
    
    # Test ship detection visualization
    log_time("Creating ship detection visualization...")
    start_time = time.time()
    try:
        fig = visualizer.plot_ship_detection_results()
        vis_time = time.time() - start_time
        log_time(f"Ship detection visualization completed in {vis_time:.2f} seconds")
        fig.savefig('ship_detection_viz.png')
        plt.close(fig)
        log_time("Saved ship detection visualization to ship_detection_viz.png")
    except Exception as e:
        log_time(f"Error creating ship detection visualization: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Test vibration heatmap
    log_time("\nCreating vibration heatmap...")
    for i in range(min(len(ship_regions), 3)):
        start_time = time.time()
        try:
            fig, heatmap = visualizer.create_vibration_heatmap(
                i, freq_range=(10.0, 30.0), window_size=16)
            heatmap_time = time.time() - start_time
            log_time(f"Vibration heatmap for ship {i} created in {heatmap_time:.2f} seconds")
            fig.savefig(f'vibration_heatmap_ship_{i}.png')
            plt.close(fig)
            log_time(f"Saved vibration heatmap for ship {i} to vibration_heatmap_ship_{i}.png")
            
            # Print heatmap statistics
            if heatmap is not None:
                valid_values = heatmap[~np.isnan(heatmap)]
                if len(valid_values) > 0:
                    log_time(f"  Heatmap frequency range: {np.min(valid_values):.2f} - {np.max(valid_values):.2f} Hz")
                    log_time(f"  Mean frequency: {np.mean(valid_values):.2f} Hz")
                else:
                    log_time("  No valid frequency values in heatmap")
        except Exception as e:
            log_time(f"Error creating vibration heatmap for ship {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test vibration spectra
    log_time("\nCreating vibration spectra plots...")
    for i in range(min(len(ship_regions), 3)):
        start_time = time.time()
        try:
            fig = visualizer.plot_vibration_spectra(i, max_freq=50.0)
            spectra_time = time.time() - start_time
            log_time(f"Vibration spectra for ship {i} created in {spectra_time:.2f} seconds")
            fig.savefig(f'vibration_spectra_ship_{i}.png')
            plt.close(fig)
            log_time(f"Saved vibration spectra for ship {i} to vibration_spectra_ship_{i}.png")
        except Exception as e:
            log_time(f"Error creating vibration spectra for ship {i}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test combined visualization
    log_time("\nCreating combined visualizations...")
    for i in range(min(len(ship_regions), 3)):
        start_time = time.time()
        try:
            fig = visualizer.create_combined_visualization(i, freq_range=(10.0, 30.0))
            combined_time = time.time() - start_time
            log_time(f"Combined visualization for ship {i} created in {combined_time:.2f} seconds")
            fig.savefig(f'combined_viz_ship_{i}.png')
            plt.close(fig)
            log_time(f"Saved combined visualization for ship {i} to combined_viz_ship_{i}.png")
        except Exception as e:
            log_time(f"Error creating combined visualization for ship {i}: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run all component tests."""
    parser = argparse.ArgumentParser(
        description="Test individual components of the ship detection pipeline"
    )
    parser.add_argument(
        "input_file", 
        help="Path to input SAR data file (CPHD, SICD, or NITF format)"
    )
    parser.add_argument(
        "--fast", 
        action="store_true",
        help="Skip time-intensive operations (use for quick diagnostics)"
    )
    parser.add_argument(
        "--full", 
        action="store_true",
        help="Run all operations including the most intensive ones"
    )
    
    args = parser.parse_args()
    
    # By default, skip large operations unless --full is specified
    skip_large_operations = not args.full
    
    # If --fast is specified, always skip large operations regardless of --full
    if args.fast:
        skip_large_operations = True
        log_time("Running in FAST mode - skipping intensive operations")
    elif args.full:
        log_time("Running in FULL mode - including all intensive operations")
    else:
        log_time("Running in DEFAULT mode - skipping the most intensive operations")
    
    # Ensure the input file exists
    if not os.path.exists(args.input_file):
        log_time(f"Error: Input file {args.input_file} does not exist")
        return 1
    
    # Test each component
    log_time(f"Testing components with {args.input_file}")
    
    # Test reader
    data, pvp_data, metadata = test_reader(args.input_file, skip_large_operations)
    
    if data is None and not skip_large_operations:
        log_time("Error: Failed to read data")
        return 1
    
    # For CPHD data, create a focused image
    focused_image = None
    if pvp_data is not None and data is not None:  # This is CPHD data
        log_time("\nCreating focused image from signal data...")
        if skip_large_operations:
            log_time("SKIPPING 2D FFT focusing operation (use --full to include)")
            # Create a simple placeholder focused image
            focused_image = np.abs(data[:min(1000, data.shape[0]), :min(1000, data.shape[1])])
        else:
            start_time = time.time()
            # Simple focusing through 2D FFT
            focused_image = np.fft.fftshift(np.fft.fft2(data))
            fft_time = time.time() - start_time
            log_time(f"2D FFT focusing completed in {fft_time:.2f} seconds")
        
        # Display focused image
        log_time("Creating focused image visualization...")
        plt.figure(figsize=(10, 8))
        # Use a downsampled version if needed
        display_image = focused_image
        if display_image.shape[0] > 1000 or display_image.shape[1] > 1000:
            downsample_factor = max(1, display_image.shape[0] // 1000, display_image.shape[1] // 1000)
            display_image = display_image[::downsample_factor, ::downsample_factor]
        
        plt.imshow(np.log10(np.abs(display_image) + 1), cmap='viridis')
        plt.colorbar(label='Log Magnitude')
        plt.title('Focused Image (from CPHD)')
        plt.savefig('focused_image.png')
        plt.close()
        log_time("Saved focused image to focused_image.png")
        
        # Use focused image for ship detection
        image_data_for_detection = focused_image
    else:
        # Use the image data directly
        image_data_for_detection = data
    
    # Test ship detector
    detection_results = test_ship_detector(image_data_for_detection, skip_large_operations)
    
    if detection_results is None and not skip_large_operations:
        log_time("Error: Ship detection failed")
        return 1
    
    # If skipping large operations and no detection results, create dummy results
    if detection_results is None or 'filtered_ships' not in detection_results:
        log_time("Creating dummy ship detection results for testing...")
        detection_results = {
            'filtered_ships': [{
                'region': np.random.rand(30, 20),
                'mask': np.ones((30, 20), dtype=bool),
                'bbox': (0, 0, 30, 20),
                'centroid': (15, 10),
                'area': 600,
                'perimeter': 100,
                'major_axis_length': 30,
                'minor_axis_length': 20,
                'orientation': 0.0,
            }]
        }
    
    # Test Doppler subaperture processing (only for CPHD data)
    vibration_results = None
    if pvp_data is not None and data is not None:
        vibration_results = test_doppler_subaperture(data, pvp_data, skip_large_operations)
    else:
        log_time("Skipping Doppler subaperture processing (requires CPHD data)")
        
    # If no vibration results, create dummy data for visualization testing
    if vibration_results is None:
        log_time("Creating dummy vibration results for testing visualization...")
        vibration_results = {
            'vibration_params': {
                'times': np.linspace(0, 10, 100),
                'range_shifts': np.sin(np.linspace(0, 20*np.pi, 100)),
                'azimuth_shifts': np.cos(np.linspace(0, 20*np.pi, 100)),
                'filtered_range_shifts': np.sin(np.linspace(0, 20*np.pi, 100)),
                'filtered_azimuth_shifts': np.cos(np.linspace(0, 20*np.pi, 100)),
                'frequencies': np.linspace(0, 25, 50),
                'range_spectrum': np.zeros(50),
                'azimuth_spectrum': np.zeros(50),
                'dominant_freq_range': 15.0,
                'dominant_freq_azimuth': 12.5
            }
        }
    
    # Test visualization
    test_visualization(image_data_for_detection, 
                      detection_results['filtered_ships'], 
                      vibration_results)
    
    log_time("\nAll tests completed.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 