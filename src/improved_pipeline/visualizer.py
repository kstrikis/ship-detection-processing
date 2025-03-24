#!/usr/bin/env python3
"""
Visualization module for the improved ship micro-motion analysis pipeline.

This module creates unified visualizations that combine ship detection,
component classification, and micro-motion analysis results.
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Any, Optional, Union, Tuple

# Import utility functions
from .utils import (
    setup_logging, save_results, load_step_output, scale_for_display
)


class MicroMotionVisualizer:
    """
    Creates integrated visualizations of ship micro-motion analysis.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        logger : Optional[logging.Logger], optional
            Logger object, by default None
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Define component color maps
        self.component_colors = {
            0: [0, 0, 0, 0],          # Background (transparent)
            1: [0, 0, 1, 0.7],        # Hull (blue)
            2: [0, 0.7, 0, 0.7],      # Deck (green)
            3: [1, 0, 0, 0.7],        # Superstructure (red)
            4: [1, 0.7, 0, 0.7],      # Bow (orange)
            5: [0.7, 0, 1, 0.7]       # Stern (purple)
        }
        
        # Define component names
        self.component_names = {
            0: 'Background',
            1: 'Hull',
            2: 'Deck',
            3: 'Superstructure',
            4: 'Bow',
            5: 'Stern'
        }
    
    def create_ship_overview(
        self, 
        image_data: np.ndarray, 
        ship_data: Dict[str, Any]
    ) -> plt.Figure:
        """
        Create an overview visualization of detected ships.
        
        Parameters
        ----------
        image_data : np.ndarray
            SAR image data
        ship_data : Dict[str, Any]
            Ship detection results
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Creating ship detection overview visualization")
        
        # Scale image for display
        display_data = scale_for_display(image_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Display the image
        ax.imshow(display_data, cmap='gray')
        
        # Overlay ships
        filtered_ships = ship_data.get('filtered_ships', [])
        for i, ship in enumerate(filtered_ships):
            bbox = ship.get('bbox')
            if bbox is None:
                continue
                
            x1, y1, x2, y2 = bbox
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1-5, f"Ship {i+1}", 
                   color='red', fontsize=10, backgroundcolor='white')
        
        ax.set_title(f"Detected Ships ({len(filtered_ships)})")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def create_component_visualization(
        self, 
        ship_region: np.ndarray, 
        component_map: np.ndarray
    ) -> plt.Figure:
        """
        Create visualization of ship component classification.
        
        Parameters
        ----------
        ship_region : np.ndarray
            Ship region image data
        component_map : np.ndarray
            Component classification map
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Creating component classification visualization")
        
        # Scale ship image for display
        if np.iscomplexobj(ship_region):
            display_data = scale_for_display(ship_region)
        else:
            display_data = ship_region / np.max(ship_region) if np.max(ship_region) > 0 else ship_region
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Display ship image in grayscale
        ax.imshow(display_data, cmap='gray', alpha=0.7)
        
        # Create RGBA image for component overlay
        component_rgba = np.zeros((*component_map.shape, 4))
        
        # Fill in colors for each component
        for comp_id, color in self.component_colors.items():
            if comp_id == 0:  # Skip background
                continue
                
            # Create mask for this component
            comp_mask = component_map == comp_id
            
            # Skip if component doesn't exist
            if not np.any(comp_mask):
                continue
                
            # Set RGBA values
            for c in range(4):  # RGBA channels
                component_rgba[comp_mask, c] = color[c]
        
        # Overlay component map
        ax.imshow(component_rgba)
        
        # Create legend
        legend_elements = []
        for comp_id, name in self.component_names.items():
            if comp_id > 0:  # Skip background
                # Check if this component exists
                if np.any(component_map == comp_id):
                    color = self.component_colors[comp_id]
                    legend_elements.append(
                        plt.Rectangle((0, 0), 1, 1, color=color, label=name)
                    )
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right')
        
        ax.set_title("Ship Component Classification")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        return fig
    
    def create_vibration_heatmap(
        self, 
        ship_region: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        component_map: Optional[np.ndarray] = None
    ) -> plt.Figure:
        """
        Create heatmap visualization of vibration characteristics.
        
        Parameters
        ----------
        ship_region : np.ndarray
            Ship region image data
        frequencies : np.ndarray
            Vibration frequencies
        amplitudes : np.ndarray
            Vibration amplitudes
        component_map : Optional[np.ndarray], optional
            Component classification map, by default None
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Creating vibration heatmap visualization")
        
        # Scale ship image for display
        if np.iscomplexobj(ship_region):
            display_data = scale_for_display(ship_region)
        else:
            display_data = ship_region / np.max(ship_region) if np.max(ship_region) > 0 else ship_region
        
        # Create figure with subplots
        n_cols = 3 if component_map is not None else 2
        fig, axes = plt.subplots(1, n_cols, figsize=(15 if n_cols == 3 else 10, 5))
        
        # Display ship image
        axes[0].imshow(display_data, cmap='gray')
        axes[0].set_title("Ship Image")
        
        # Create mask for valid vibrations
        valid_mask = amplitudes > 0
        
        # Display frequency heatmap
        freq_display = np.zeros_like(frequencies)
        freq_display[valid_mask] = frequencies[valid_mask]
        
        # Set colormap limits
        if np.any(valid_mask):
            vmin = np.min(freq_display[valid_mask])
            vmax = np.max(freq_display[valid_mask])
        else:
            vmin, vmax = 0, 1
            
        freq_im = axes[1].imshow(freq_display, cmap='jet', vmin=vmin, vmax=vmax)
        axes[1].set_title("Vibration Frequency (Hz)")
        plt.colorbar(freq_im, ax=axes[1])
        
        # Display component map if provided
        if component_map is not None:
            # Create RGBA image for component map
            component_rgba = np.zeros((*component_map.shape, 4))
            
            # Fill in colors for each component
            for comp_id, color in self.component_colors.items():
                if comp_id == 0:  # Skip background
                    continue
                    
                # Create mask for this component
                comp_mask = component_map == comp_id
                
                # Skip if component doesn't exist
                if not np.any(comp_mask):
                    continue
                    
                # Set RGBA values
                for c in range(4):  # RGBA channels
                    component_rgba[comp_mask, c] = color[c]
            
            # Display component map
            axes[2].imshow(component_rgba)
            axes[2].set_title("Ship Components")
            
            # Create legend
            legend_elements = []
            for comp_id, name in self.component_names.items():
                if comp_id > 0:  # Skip background
                    # Check if this component exists
                    if np.any(component_map == comp_id):
                        color = self.component_colors[comp_id]
                        legend_elements.append(
                            plt.Rectangle((0, 0), 1, 1, color=color, label=name)
                        )
            
            if legend_elements:
                axes[2].legend(handles=legend_elements, loc='upper right')
        
        # Remove axis ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        return fig
    
    def create_component_frequency_summary(
        self, 
        component_map: np.ndarray,
        frequencies: np.ndarray,
        amplitudes: np.ndarray,
        component_ranges: Dict[int, Tuple[float, float]] = None
    ) -> plt.Figure:
        """
        Create summary of frequency distributions by component.
        
        Parameters
        ----------
        component_map : np.ndarray
            Component classification map
        frequencies : np.ndarray
            Vibration frequencies
        amplitudes : np.ndarray
            Vibration amplitudes
        component_ranges : Dict[int, Tuple[float, float]], optional
            Expected frequency ranges per component, by default None
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Creating component frequency summary visualization")
        
        # Get unique component IDs (excluding background)
        component_ids = sorted(list(set(np.unique(component_map)) - {0}))
        
        if not component_ids:
            self.logger.warning("No components found in component map")
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No components found", ha='center', va='center')
            return fig
        
        # Create figure with subplots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram data
        hist_data = []
        labels = []
        colors = []
        
        max_freq = 0
        
        for comp_id in component_ids:
            # Skip if component name not defined
            if comp_id not in self.component_names:
                continue
                
            # Create mask for this component
            comp_mask = component_map == comp_id
            
            # Get frequencies for this component
            comp_freqs = frequencies[comp_mask]
            comp_amps = amplitudes[comp_mask]
            
            # Skip if no valid data
            valid_mask = comp_amps > 0
            if not np.any(valid_mask):
                continue
                
            # Get valid frequencies
            valid_freqs = comp_freqs[valid_mask]
            
            # Update max frequency
            if len(valid_freqs) > 0:
                max_freq = max(max_freq, np.max(valid_freqs))
            
            # Add to histogram data
            hist_data.append(valid_freqs)
            labels.append(self.component_names[comp_id])
            colors.append(self.component_colors[comp_id][:3])  # RGB only
        
        if not hist_data:
            self.logger.warning("No valid frequency data found for any component")
            ax.text(0.5, 0.5, "No valid frequency data", ha='center', va='center')
            return fig
        
        # Create histogram
        n_bins = 20
        ax.hist(hist_data, bins=n_bins, range=(0, max_freq*1.1), 
               label=labels, color=colors, alpha=0.7, stacked=False)
        
        # Add expected frequency ranges if provided
        if component_ranges is not None:
            y_max = ax.get_ylim()[1]
            for comp_id, (min_freq, max_freq) in component_ranges.items():
                if comp_id in component_ids and comp_id in self.component_names:
                    color = self.component_colors[comp_id][:3]
                    ax.axvspan(min_freq, max_freq, alpha=0.2, color=color)
                    ax.axvline(min_freq, color=color, linestyle='--', alpha=0.7)
                    ax.axvline(max_freq, color=color, linestyle='--', alpha=0.7)
        
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Pixel Count")
        ax.set_title("Vibration Frequency Distribution by Component")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        return fig
    
    def create_combined_visualization(
        self, 
        ship_data: Dict[str, Any],
        preprocessed_data: Dict[str, Any] = None
    ) -> plt.Figure:
        """
        Create comprehensive visualization combining all analysis stages.
        
        Parameters
        ----------
        ship_data : Dict[str, Any]
            Ship analysis data
        preprocessed_data : Dict[str, Any], optional
            Preprocessed SAR data, by default None
            
        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        self.logger.info("Creating combined visualization")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 12))
        
        # Define grid for subplots
        gs = fig.add_gridspec(3, 3)
        
        # Extract ship data
        ship_region = ship_data.get('region')
        component_map = ship_data.get('component_map')
        frequencies = ship_data.get('final_frequencies', ship_data.get('freq_smoothed', ship_data.get('dominant_frequencies')))
        amplitudes = ship_data.get('final_amplitudes', ship_data.get('amp_normalized', ship_data.get('dominant_amplitudes')))
        
        if ship_region is None and 'bbox' in ship_data and preprocessed_data is not None:
            # Try to extract ship region from preprocessed data
            if 'focused_image' in preprocessed_data:
                x1, y1, x2, y2 = ship_data['bbox']
                ship_region = preprocessed_data['focused_image'][y1:y2+1, x1:x2+1]
        
        # Skip if missing required data
        if ship_region is None or frequencies is None or amplitudes is None:
            self.logger.warning("Missing required data for combined visualization")
            ax = fig.add_subplot(gs[:, :])
            ax.text(0.5, 0.5, "Insufficient data for visualization", ha='center', va='center')
            return fig
        
        # Scale ship image for display
        if np.iscomplexobj(ship_region):
            display_data = scale_for_display(ship_region)
        else:
            display_data = ship_region / np.max(ship_region) if np.max(ship_region) > 0 else ship_region
        
        # Ship image (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(display_data, cmap='gray')
        ax1.set_title("Ship Image")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Component map (top-center)
        ax2 = fig.add_subplot(gs[0, 1])
        
        if component_map is not None:
            # Create RGBA image for component map
            component_rgba = np.zeros((*component_map.shape, 4))
            
            # Fill in colors for each component
            for comp_id, color in self.component_colors.items():
                if comp_id == 0:  # Skip background
                    continue
                    
                # Create mask for this component
                comp_mask = component_map == comp_id
                
                # Skip if component doesn't exist
                if not np.any(comp_mask):
                    continue
                    
                # Set RGBA values
                for c in range(4):  # RGBA channels
                    component_rgba[comp_mask, c] = color[c]
            
            # Display component map
            ax2.imshow(component_rgba)
            
            # Create legend
            legend_elements = []
            for comp_id, name in self.component_names.items():
                if comp_id > 0:  # Skip background
                    # Check if this component exists
                    if np.any(component_map == comp_id):
                        color = self.component_colors[comp_id]
                        legend_elements.append(
                            plt.Rectangle((0, 0), 1, 1, color=color, label=name)
                        )
            
            if legend_elements:
                ax2.legend(handles=legend_elements, loc='upper right', fontsize='small')
        else:
            ax2.text(0.5, 0.5, "No component data", ha='center', va='center')
            
        ax2.set_title("Component Classification")
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        # Frequency map (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Create mask for valid vibrations
        valid_mask = amplitudes > 0
        
        # Display frequency heatmap
        freq_display = np.zeros_like(frequencies)
        freq_display[valid_mask] = frequencies[valid_mask]
        
        # Set colormap limits
        if np.any(valid_mask):
            vmin = np.min(freq_display[valid_mask])
            vmax = np.max(freq_display[valid_mask])
        else:
            vmin, vmax = 0, 1
            
        freq_im = ax3.imshow(freq_display, cmap='jet', vmin=vmin, vmax=vmax)
        fig.colorbar(freq_im, ax=ax3, label="Frequency (Hz)")
        ax3.set_title("Vibration Frequencies")
        ax3.set_xticks([])
        ax3.set_yticks([])
        
        # Amplitude map (middle-left)
        ax4 = fig.add_subplot(gs[1, 0])
        
        # Display amplitude heatmap
        if np.any(valid_mask):
            amp_display = amplitudes / np.max(amplitudes)
        else:
            amp_display = amplitudes
            
        amp_im = ax4.imshow(amp_display, cmap='inferno')
        fig.colorbar(amp_im, ax=ax4, label="Normalized Amplitude")
        ax4.set_title("Vibration Amplitudes")
        ax4.set_xticks([])
        ax4.set_yticks([])
        
        # Component-colored amplitude (middle-center)
        ax5 = fig.add_subplot(gs[1, 1])
        
        if component_map is not None:
            # Create component-colored amplitude map
            height, width = component_map.shape
            rgb_image = np.zeros((height, width, 3))
            
            # Normalize amplitude for brightness
            norm_amp = amplitudes / np.max(amplitudes) if np.max(amplitudes) > 0 else np.zeros_like(amplitudes)
            
            # Combine component colors with amplitude
            for comp_id, color in self.component_colors.items():
                if comp_id == 0:  # Skip background
                    continue
                    
                # Create mask for this component
                comp_mask = component_map == comp_id
                
                # Skip if component doesn't exist
                if not np.any(comp_mask):
                    continue
                    
                # Set RGB values based on component color and amplitude
                for c in range(3):  # RGB channels
                    rgb_image[comp_mask, c] = color[c] * (0.3 + 0.7 * norm_amp[comp_mask])
            
            ax5.imshow(rgb_image)
        else:
            ax5.text(0.5, 0.5, "No component data", ha='center', va='center')
            
        ax5.set_title("Component-Colored Amplitude")
        ax5.set_xticks([])
        ax5.set_yticks([])
        
        # Frequency histogram (middle-right + bottom-right)
        ax6 = fig.add_subplot(gs[1:, 2])
        
        # Get frequencies for histogram
        if np.any(valid_mask):
            valid_freqs = frequencies[valid_mask]
            
            # Create histogram
            ax6.hist(valid_freqs, bins=20, color='blue', alpha=0.7)
            ax6.set_xlabel("Frequency (Hz)")
            ax6.set_ylabel("Pixel Count")
            
            # Add component-specific expected ranges if available
            if 'component_freq_ranges' in ship_data and component_map is not None:
                component_ranges = ship_data['component_freq_ranges']
                y_max = ax6.get_ylim()[1]
                
                for comp_id, (min_freq, max_freq) in component_ranges.items():
                    if comp_id in self.component_colors and np.any(component_map == comp_id):
                        color = self.component_colors[comp_id][:3]
                        ax6.axvspan(min_freq, max_freq, alpha=0.2, color=color)
                        ax6.axvline(min_freq, color=color, linestyle='--', alpha=0.7)
                        ax6.axvline(max_freq, color=color, linestyle='--', alpha=0.7)
                        
                        # Add text label
                        mid_freq = (min_freq + max_freq) / 2
                        ax6.text(mid_freq, y_max * 0.9, self.component_names.get(comp_id, f"Comp {comp_id}"),
                               rotation=90, color=color, ha='center', va='top', fontsize='small')
        else:
            ax6.text(0.5, 0.5, "No valid frequency data", ha='center', va='center')
            
        ax6.set_title("Frequency Distribution")
        ax6.grid(True, linestyle='--', alpha=0.7)
        
        # Component frequency summary (bottom-left + bottom-center)
        ax7 = fig.add_subplot(gs[2, :2])
        
        if component_map is not None and np.any(valid_mask):
            # Create component-specific histograms
            component_ids = sorted(list(set(np.unique(component_map)) - {0}))
            
            if component_ids:
                # Create histogram data
                hist_data = []
                labels = []
                colors = []
                
                for comp_id in component_ids:
                    # Skip if component name not defined
                    if comp_id not in self.component_names:
                        continue
                        
                    # Create mask for this component
                    comp_mask = (component_map == comp_id) & valid_mask
                    
                    # Skip if no valid data
                    if not np.any(comp_mask):
                        continue
                        
                    # Get valid frequencies
                    valid_freqs = frequencies[comp_mask]
                    
                    # Add to histogram data
                    hist_data.append(valid_freqs)
                    labels.append(self.component_names[comp_id])
                    colors.append(self.component_colors[comp_id][:3])  # RGB only
                
                if hist_data:
                    # Create histogram
                    ax7.hist(hist_data, bins=20, label=labels, color=colors, alpha=0.7, stacked=False)
                    ax7.legend(fontsize='small')
                else:
                    ax7.text(0.5, 0.5, "No component-specific data", ha='center', va='center')
            else:
                ax7.text(0.5, 0.5, "No components identified", ha='center', va='center')
        else:
            ax7.text(0.5, 0.5, "No component or frequency data", ha='center', va='center')
            
        ax7.set_xlabel("Frequency (Hz)")
        ax7.set_ylabel("Pixel Count")
        ax7.set_title("Frequency Distribution by Component")
        ax7.grid(True, linestyle='--', alpha=0.7)
        
        # Set overall title
        ship_index = ship_data.get('ship_index', 0)
        fig.suptitle(f"Ship {ship_index+1} - Comprehensive Micro-Motion Analysis", fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for suptitle
        return fig


def create_visualizations(
    preprocessed_file: str,
    ship_file: str,
    component_file: str,
    vibration_file: str,
    output_dir: str,
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    Create unified visualizations from all analysis stages.
    
    Parameters
    ----------
    preprocessed_file : str
        Path to preprocessed data file
    ship_file : str
        Path to ship detection results file
    component_file : str
        Path to component classification results file
    vibration_file : str
        Path to vibration analysis results file
    output_dir : str
        Directory to save visualizations
    logger : Optional[logging.Logger], optional
        Logger object, by default None
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping visualization names to file paths
    """
    if logger is None:
        logger = setup_logging()
    
    logger.info("Creating unified visualizations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data from all stages
    logger.info(f"Loading preprocessed data from {preprocessed_file}")
    preprocessed = load_step_output(preprocessed_file)
    
    logger.info(f"Loading ship detection data from {ship_file}")
    ships = load_step_output(ship_file)
    
    logger.info(f"Loading component classification data from {component_file}")
    components = load_step_output(component_file)
    
    logger.info(f"Loading vibration analysis data from {vibration_file}")
    vibrations = load_step_output(vibration_file)
    
    # Initialize visualizer
    visualizer = MicroMotionVisualizer(logger=logger)
    
    # Output file paths
    output_files = {}
    
    # Create ship detection overview
    try:
        if 'filtered_image' in preprocessed:
            image_data = preprocessed['filtered_image']
        elif 'focused_image' in preprocessed:
            image_data = preprocessed['focused_image']
        elif 'preview_image' in preprocessed:
            image_data = preprocessed['preview_image']
        else:
            logger.warning("No suitable image found in preprocessed data")
            image_data = None
            
        if image_data is not None:
            logger.info("Creating ship detection overview")
            fig = visualizer.create_ship_overview(image_data, ships)
            overview_path = os.path.join(output_dir, "ship_detection_overview.png")
            fig.savefig(overview_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_files['overview'] = overview_path
            logger.info(f"Saved ship detection overview to {overview_path}")
    except Exception as e:
        logger.error(f"Error creating ship detection overview: {str(e)}")
    
    # Create combined visualizations for each ship
    vibration_ships = vibrations.get('ship_results', [])
    component_ships = components.get('ship_results', [])
    ship_count = 0
    
    # Process each ship from vibration results
    for vib_ship in vibration_ships:
        ship_index = vib_ship.get('ship_index', ship_count)
        
        try:
            # Find matching component data
            comp_ship = None
            for cs in component_ships:
                if cs.get('ship_index', -1) == ship_index:
                    comp_ship = cs
                    break
            
            if comp_ship is None:
                logger.warning(f"No component data found for ship {ship_index}, using vibration data only")
            
            # Combine data
            combined_data = {**vib_ship}
            if comp_ship is not None:
                combined_data.update({
                    'component_map': comp_ship.get('component_map'),
                    'component_stats': comp_ship.get('component_stats')
                })
            
            # Create combined visualization
            logger.info(f"Creating combined visualization for ship {ship_index}")
            fig = visualizer.create_combined_visualization(combined_data, preprocessed)
            combined_path = os.path.join(output_dir, f"ship{ship_index+1}_combined.png")
            fig.savefig(combined_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            output_files[f'ship{ship_index+1}_combined'] = combined_path
            logger.info(f"Saved combined visualization to {combined_path}")
            
            # Create component frequency summary if component data available
            if comp_ship is not None and 'component_map' in comp_ship:
                logger.info(f"Creating component frequency summary for ship {ship_index}")
                
                # Get component frequency ranges if available
                component_ranges = None
                if 'component_freq_ranges' in vibrations:
                    component_ranges = vibrations['component_freq_ranges']
                
                # Create visualization
                fig = visualizer.create_component_frequency_summary(
                    comp_ship['component_map'],
                    vib_ship.get('final_frequencies', vib_ship.get('dominant_frequencies')),
                    vib_ship.get('final_amplitudes', vib_ship.get('dominant_amplitudes')),
                    component_ranges
                )
                
                freq_summary_path = os.path.join(output_dir, f"ship{ship_index+1}_freq_summary.png")
                fig.savefig(freq_summary_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                output_files[f'ship{ship_index+1}_freq_summary'] = freq_summary_path
                logger.info(f"Saved frequency summary to {freq_summary_path}")
            
            ship_count += 1
            
        except Exception as e:
            logger.error(f"Error creating visualizations for ship {ship_index}: {str(e)}")
    
    logger.info(f"Created {len(output_files)} visualization files")
    return output_files


if __name__ == "__main__":
    """Run as standalone script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Visualization Creator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--preprocessed-file', type=str, required=True,
                      help='Preprocessed data file')
    parser.add_argument('--ship-file', type=str, required=True,
                      help='Ship detection results file')
    parser.add_argument('--component-file', type=str, required=True,
                      help='Component classification results file')
    parser.add_argument('--vibration-file', type=str, required=True,
                      help='Vibration analysis results file')
    parser.add_argument('--output-dir', type=str, required=True,
                      help='Output directory for visualizations')
    parser.add_argument('--log-file', type=str,
                      help='Log file path')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_file, args.log_level)
    
    try:
        # Create visualizations
        result = create_visualizations(
            args.preprocessed_file,
            args.ship_file,
            args.component_file,
            args.vibration_file,
            args.output_dir,
            logger
        )
        logger.info("Visualization creation completed successfully")
    except Exception as e:
        logger.error(f"Error during visualization creation: {str(e)}")
        sys.exit(1) 