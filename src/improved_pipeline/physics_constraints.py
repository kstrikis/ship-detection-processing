#!/usr/bin/env python3
"""
Physics-based constraints module for the improved ship micro-motion analysis pipeline.

This module applies physical constraints to detected vibrations based on expected
frequency ranges for different ship components, structural connectivity, and
physical plausibility.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from utils import (
    setup_logging, save_results, load_step_output, check_gpu_availability,
    scale_for_display
)


class PhysicsConstrainedAnalyzer:
    """
    Applies physical constraints to vibration analysis.
    """
    
    def __init__(
        self,
        use_gpu: bool = False,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the physics-based constraint analyzer.
        
        Parameters
        ----------
        use_gpu : bool, optional
            Whether to use GPU acceleration if available, by default False
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        """
        self.use_gpu = use_gpu and check_gpu_availability()
        self.logger = logger or logging.getLogger(__name__)
        
        # Define expected frequency ranges for different components
        # These are based on typical ship vibration characteristics
        self.component_freq_ranges = {
            1: (5, 15),    # Hull: 5-15 Hz
            2: (8, 20),    # Deck: 8-20 Hz
            3: (10, 30),   # Superstructure: 10-30 Hz
            4: (12, 25),   # Bow: 12-25 Hz
            5: (8, 18)     # Stern (engine area): 8-18 Hz
        }
        
        # Define typical vibration amplitudes relative to the hull
        # These are based on typical ship vibration characteristics
        self.component_amp_multipliers = {
            1: 1.0,       # Hull (reference)
            2: 1.2,       # Deck (20% stronger than hull)
            3: 1.5,       # Superstructure (50% stronger than hull)
            4: 0.8,       # Bow (80% of hull)
            5: 1.8        # Stern (80% stronger due to engines)
        }
        
        if self.use_gpu:
            self.logger.info("Using GPU acceleration for physics constraint application")
    
    def apply_component_constraints(
        self, 
        component_map: np.ndarray, 
        vibration_frequencies: np.ndarray,
        vibration_amplitudes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply component-specific frequency constraints.
        
        Parameters
        ----------
        component_map : np.ndarray
            Component classification map
        vibration_frequencies : np.ndarray
            Detected vibration frequencies
        vibration_amplitudes : np.ndarray
            Detected vibration amplitudes
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Constrained frequencies and amplitudes
        """
        self.logger.info("Applying component-specific frequency constraints")
        
        # Make copies to avoid modifying originals
        constrained_freqs = vibration_frequencies.copy()
        constrained_amps = vibration_amplitudes.copy()
        
        # Process each component
        for component_id, freq_range in self.component_freq_ranges.items():
            # Create mask for this component
            component_mask = component_map == component_id
            
            # Skip if component doesn't exist in this ship
            if not np.any(component_mask):
                continue
                
            min_freq, max_freq = freq_range
            self.logger.debug(f"Processing component {component_id} with freq range {min_freq}-{max_freq} Hz")
            
            # Find frequencies outside expected range
            invalid_mask = (
                (vibration_frequencies < min_freq) | 
                (vibration_frequencies > max_freq)
            ) & component_mask
            
            # Zero out invalid frequencies and amplitudes
            constrained_freqs[invalid_mask] = 0
            constrained_amps[invalid_mask] = 0
            
            # For pixels with no valid frequencies, find nearest valid frequency in component
            zero_mask = (constrained_amps == 0) & component_mask
            if np.any(zero_mask):
                # Calculate mean of valid frequencies for this component
                valid_freqs = constrained_freqs[component_mask & ~zero_mask]
                
                if len(valid_freqs) > 0:
                    # Use mean frequency for invalid pixels
                    mean_freq = np.mean(valid_freqs[valid_freqs > 0])
                    constrained_freqs[zero_mask] = mean_freq
                    
                    # Use small amplitude
                    mean_amp = np.mean(constrained_amps[constrained_amps > 0]) if np.any(constrained_amps > 0) else 0.1
                    constrained_amps[zero_mask] = 0.1 * mean_amp
                else:
                    # If no valid frequencies in this component, use middle of expected range
                    mid_freq = (min_freq + max_freq) / 2
                    constrained_freqs[zero_mask] = mid_freq
                    
                    # Use small amplitude
                    constrained_amps[zero_mask] = 0.1
        
        return constrained_freqs, constrained_amps
    
    def apply_structural_continuity(
        self, 
        component_map: np.ndarray, 
        vibration_frequencies: np.ndarray,
        vibration_amplitudes: np.ndarray,
        smoothing_radius: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply structural continuity constraints (smoothing within components).
        
        Parameters
        ----------
        component_map : np.ndarray
            Component classification map
        vibration_frequencies : np.ndarray
            Vibration frequencies
        vibration_amplitudes : np.ndarray
            Vibration amplitudes
        smoothing_radius : int, optional
            Radius for smoothing, by default 2
            
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Smoothed frequencies and amplitudes
        """
        self.logger.info(f"Applying structural continuity constraints with radius {smoothing_radius}")
        
        # Make copies to avoid modifying originals
        smoothed_freqs = vibration_frequencies.copy()
        smoothed_amps = vibration_amplitudes.copy()
        
        # Process each component separately
        for component_id in range(1, 6):  # Skip background (0)
            # Create mask for this component
            component_mask = component_map == component_id
            
            # Skip if component doesn't exist
            if not np.any(component_mask):
                continue
            
            # Get all pixels in this component with non-zero frequencies
            valid_pixels = component_mask & (vibration_frequencies > 0)
            
            if not np.any(valid_pixels):
                continue
            
            # Create binary mask for this component
            binary_mask = component_mask.astype(np.float32)
            
            # Apply smoothing within the component
            # Use gaussian filter for smoother results
            from scipy import ndimage
            
            # Create temporary arrays for smoothing
            temp_freqs = np.zeros_like(smoothed_freqs)
            temp_amps = np.zeros_like(smoothed_amps)
            temp_weights = np.zeros_like(smoothed_freqs)
            
            # Set values for this component
            temp_freqs[component_mask] = smoothed_freqs[component_mask]
            temp_amps[component_mask] = smoothed_amps[component_mask]
            temp_weights[component_mask] = 1.0
            
            # Apply gaussian smoothing
            smooth_freqs = ndimage.gaussian_filter(
                temp_freqs, sigma=smoothing_radius, mode='constant', cval=0)
            smooth_amps = ndimage.gaussian_filter(
                temp_amps, sigma=smoothing_radius, mode='constant', cval=0)
            smooth_weights = ndimage.gaussian_filter(
                temp_weights, sigma=smoothing_radius, mode='constant', cval=0)
            
            # Normalize by weights to prevent edge effects
            mask = smooth_weights > 0
            smooth_freqs[mask] /= smooth_weights[mask]
            smooth_amps[mask] /= smooth_weights[mask]
            
            # Update values only for this component
            smoothed_freqs[component_mask] = smooth_freqs[component_mask]
            smoothed_amps[component_mask] = smooth_amps[component_mask]
        
        return smoothed_freqs, smoothed_amps
    
    def apply_amplitude_normalization(
        self, 
        component_map: np.ndarray, 
        vibration_frequencies: np.ndarray,
        vibration_amplitudes: np.ndarray
    ) -> np.ndarray:
        """
        Apply component-specific amplitude normalization.
        
        Parameters
        ----------
        component_map : np.ndarray
            Component classification map
        vibration_frequencies : np.ndarray
            Vibration frequencies
        vibration_amplitudes : np.ndarray
            Vibration amplitudes
            
        Returns
        -------
        np.ndarray
            Normalized amplitudes
        """
        self.logger.info("Applying component-specific amplitude normalization")
        
        # Make a copy to avoid modifying original
        normalized_amps = vibration_amplitudes.copy()
        
        # First, calculate the mean amplitude of the hull (reference)
        hull_mask = (component_map == 1) & (vibration_frequencies > 0)
        if np.any(hull_mask):
            hull_mean_amp = np.mean(vibration_amplitudes[hull_mask])
        else:
            # No hull component, use overall mean as reference
            valid_mask = vibration_frequencies > 0
            hull_mean_amp = np.mean(vibration_amplitudes[valid_mask]) if np.any(valid_mask) else 1.0
        
        # Process each component
        for component_id, multiplier in self.component_amp_multipliers.items():
            # Create mask for this component
            component_mask = component_map == component_id
            
            # Skip if component doesn't exist
            if not np.any(component_mask):
                continue
            
            # Skip hull (reference component)
            if component_id == 1:
                continue
            
            # Calculate expected amplitudes relative to hull
            expected_mean_amp = hull_mean_amp * multiplier
            
            # Get current mean amplitude for component
            valid_mask = component_mask & (vibration_frequencies > 0)
            if np.any(valid_mask):
                current_mean_amp = np.mean(vibration_amplitudes[valid_mask])
                
                # Scale factor to adjust to expected amplitude
                if current_mean_amp > 0:
                    scale_factor = expected_mean_amp / current_mean_amp
                    
                    # Apply scaling to all pixels in this component
                    normalized_amps[component_mask] *= scale_factor
        
        return normalized_amps
    
    def process_ship(
        self, 
        component_map: np.ndarray, 
        vibration_frequencies: np.ndarray,
        vibration_amplitudes: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Apply all physical constraints to a ship.
        
        Parameters
        ----------
        component_map : np.ndarray
            Component classification map
        vibration_frequencies : np.ndarray
            Original vibration frequencies
        vibration_amplitudes : np.ndarray
            Original vibration amplitudes
            
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary with constrained results
        """
        self.logger.info("Applying all physical constraints to ship")
        
        # Step 1: Apply component-specific frequency constraints
        freq_constrained, amp_constrained = self.apply_component_constraints(
            component_map, vibration_frequencies, vibration_amplitudes)
        
        # Step 2: Apply structural continuity
        freq_smoothed, amp_smoothed = self.apply_structural_continuity(
            component_map, freq_constrained, amp_constrained)
        
        # Step 3: Apply amplitude normalization
        amp_normalized = self.apply_amplitude_normalization(
            component_map, freq_smoothed, amp_smoothed)
        
        # Return all results for comparison
        return {
            'original_frequencies': vibration_frequencies,
            'original_amplitudes': vibration_amplitudes,
            'freq_constrained': freq_constrained,
            'amp_constrained': amp_constrained,
            'freq_smoothed': freq_smoothed,
            'amp_smoothed': amp_smoothed,
            'amp_normalized': amp_normalized,
            'final_frequencies': freq_smoothed,
            'final_amplitudes': amp_normalized
        }


def apply_physics_constraints(
    input_file: str,
    component_file: str,
    output_file: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    Apply physical constraints to vibration analysis results.
    
    Parameters
    ----------
    input_file : str
        Path to input file (time-frequency analysis results)
    component_file : str
        Path to component classification results
    output_file : str
        Path to output file
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing constrained vibration results
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info(f"Applying physics constraints to {input_file} using components from {component_file}")
    
    # Load time-frequency analysis results
    tf_data = load_step_output(input_file)
    logger.info(f"Loaded time-frequency data with keys: {tf_data.keys()}")
    
    # Load component classification results
    comp_data = load_step_output(component_file)
    logger.info(f"Loaded component data with keys: {comp_data.keys()}")
    
    # Initialize physics constraint analyzer
    analyzer = PhysicsConstrainedAnalyzer(logger=logger)
    
    # Process each ship
    ship_results = []
    
    for i, tf_ship in enumerate(tf_data.get('ship_results', [])):
        logger.info(f"Processing ship {i+1}/{len(tf_data.get('ship_results', []))}")
        
        # Find corresponding component data
        comp_ship = None
        for comp_result in comp_data.get('ship_results', []):
            if comp_result.get('ship_index', -1) == tf_ship.get('ship_index', -1):
                comp_ship = comp_result
                break
        
        if comp_ship is None:
            logger.warning(f"No component data found for ship {i+1}, skipping")
            continue
        
        # Get vibration data
        if 'dominant_frequencies' not in tf_ship or 'dominant_amplitudes' not in tf_ship:
            logger.warning(f"No vibration data found for ship {i+1}, skipping")
            continue
        
        # Get component map
        if 'component_map' not in comp_ship:
            logger.warning(f"No component map found for ship {i+1}, skipping")
            continue
        
        # Extract data
        vibration_freqs = tf_ship['dominant_frequencies']
        vibration_amps = tf_ship['dominant_amplitudes']
        component_map = comp_ship['component_map']
        
        # Check dimensions match
        if vibration_freqs.shape != component_map.shape:
            logger.warning(
                f"Shape mismatch: vibration_freqs {vibration_freqs.shape} vs "
                f"component_map {component_map.shape}, skipping ship {i+1}"
            )
            continue
        
        # Apply physics constraints
        constrained_results = analyzer.process_ship(
            component_map, vibration_freqs, vibration_amps)
        
        # Add metadata
        result = {
            'ship_index': tf_ship.get('ship_index', i),
            'bbox': tf_ship.get('ship_bbox'),
            'dimensions': tf_ship.get('dimensions'),
            'component_map': component_map
        }
        
        # Add constrained results
        result.update(constrained_results)
        
        # Store results
        ship_results.append(result)
    
    # Create complete results dictionary
    results = {
        'ship_results': ship_results,
        'num_ships': len(ship_results),
        'component_freq_ranges': analyzer.component_freq_ranges,
        'component_amp_multipliers': analyzer.component_amp_multipliers,
        'timestamp': tf_data.get('timestamp', None),
        'input_file': input_file,
        'component_file': component_file
    }
    
    # Save results
    save_results(output_file, results)
    logger.info(f"Saved physics-constrained results to {output_file}")
    
    # Create visualizations
    create_constrained_visualizations(
        ship_results, 
        os.path.splitext(output_file)[0]
    )
    
    return results


def create_constrained_visualizations(
    ship_results: List[Dict[str, Any]],
    output_prefix: str
) -> None:
    """
    Create visualizations of physics-constrained vibration results.
    
    Parameters
    ----------
    ship_results : List[Dict[str, Any]]
        List of constrained vibration results per ship
    output_prefix : str
        Prefix for output file paths
    """
    for i, ship in enumerate(ship_results):
        # Skip if no data available
        if 'final_frequencies' not in ship or 'final_amplitudes' not in ship:
            continue
        
        # Extract data
        freq_orig = ship['original_frequencies']
        amp_orig = ship['original_amplitudes']
        freq_final = ship['final_frequencies']
        amp_final = ship['final_amplitudes']
        component_map = ship['component_map']
        
        # Create comparison figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Create valid masks
        valid_mask_orig = amp_orig > 0
        valid_mask_final = amp_final > 0
        
        # Create frequency colormaps
        vmin_freq = min(
            np.min(freq_orig[valid_mask_orig]) if np.any(valid_mask_orig) else 0,
            np.min(freq_final[valid_mask_final]) if np.any(valid_mask_final) else 0
        )
        vmax_freq = max(
            np.max(freq_orig[valid_mask_orig]) if np.any(valid_mask_orig) else 1,
            np.max(freq_final[valid_mask_final]) if np.any(valid_mask_final) else 1
        )
        
        # Plot original frequency
        im1 = axes[0, 0].imshow(freq_orig, cmap='jet', vmin=vmin_freq, vmax=vmax_freq)
        axes[0, 0].set_title("Original Frequencies")
        plt.colorbar(im1, ax=axes[0, 0], label="Frequency (Hz)")
        
        # Plot constrained frequency
        im2 = axes[0, 1].imshow(freq_final, cmap='jet', vmin=vmin_freq, vmax=vmax_freq)
        axes[0, 1].set_title("Constrained Frequencies")
        plt.colorbar(im2, ax=axes[0, 1], label="Frequency (Hz)")
        
        # Create amplitude colormaps
        vmin_amp = 0
        vmax_amp = max(
            np.max(amp_orig) if np.any(amp_orig > 0) else 1,
            np.max(amp_final) if np.any(amp_final > 0) else 1
        )
        
        # Plot original amplitude
        im3 = axes[1, 0].imshow(amp_orig, cmap='inferno', vmin=vmin_amp, vmax=vmax_amp)
        axes[1, 0].set_title("Original Amplitudes")
        plt.colorbar(im3, ax=axes[1, 0], label="Amplitude")
        
        # Plot constrained amplitude
        im4 = axes[1, 1].imshow(amp_final, cmap='inferno', vmin=vmin_amp, vmax=vmax_amp)
        axes[1, 1].set_title("Constrained Amplitudes")
        plt.colorbar(im4, ax=axes[1, 1], label="Amplitude")
        
        # Set overall title
        fig.suptitle(f"Ship {i+1} - Physics-Constrained Vibration Results", fontsize=16)
        
        # Save figure
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
        vibration_file = f"{output_prefix}_ship{i+1}_constrained.png"
        plt.savefig(vibration_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create component-colored vibration map
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create component-colored vibration map
        # This visualization shows vibration amplitude with component-based coloring
        
        # Define component colors (RGB)
        component_colors = {
            0: [0, 0, 0],         # Background (black)
            1: [0, 0, 1],         # Hull (blue)
            2: [0, 0.7, 0],       # Deck (green)
            3: [1, 0, 0],         # Superstructure (red)
            4: [1, 0.7, 0],       # Bow (orange)
            5: [0.7, 0, 1]        # Stern (purple)
        }
        
        # Create RGB image
        height, width = component_map.shape
        rgb_image = np.zeros((height, width, 3))
        
        # Normalize amplitude for brightness
        norm_amp = amp_final / np.max(amp_final) if np.max(amp_final) > 0 else np.zeros_like(amp_final)
        
        # Combine component colors with amplitude
        for comp_id, color in component_colors.items():
            if comp_id == 0:  # Skip background
                continue
                
            # Create mask for this component
            comp_mask = component_map == comp_id
            
            # Set RGB values based on component color and amplitude
            for c in range(3):  # RGB channels
                rgb_image[comp_mask, c] = color[c] * (0.3 + 0.7 * norm_amp[comp_mask])
        
        # Display image
        ax.imshow(rgb_image)
        
        # Create legend
        legend_elements = []
        for comp_id, color in component_colors.items():
            if comp_id > 0:  # Skip background
                legend_elements.append(
                    plt.Rectangle((0, 0), 1, 1, color=color, label=f"Component {comp_id}")
                )
        
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Add title
        ax.set_title(f"Ship {i+1} - Component-Based Vibration Map")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save figure
        comp_vib_file = f"{output_prefix}_ship{i+1}_comp_vibration.png"
        plt.tight_layout()
        plt.savefig(comp_vib_file, dpi=300, bbox_inches='tight')
        plt.close(fig)


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Physics-Based Constraint Analyzer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input', type=str, required=True,
                      help='Input file (time-frequency analysis results)')
    parser.add_argument('--component-file', type=str, required=True,
                      help='Component classification results file')
    parser.add_argument('--output', type=str, required=True,
                      help='Output file path')
    parser.add_argument('--log-file', type=str,
                      help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    try:
        # Apply physics constraints
        result = apply_physics_constraints(
            args.input,
            args.component_file,
            args.output,
            logger
        )
        logger.info("Physics constraint application completed successfully")
    except Exception as e:
        logger.error(f"Error during physics constraint application: {str(e)}")
        sys.exit(1) 