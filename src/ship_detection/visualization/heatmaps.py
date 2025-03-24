"""
Module for visualizing ship micro-motion vibration frequencies as heatmaps.
"""

import logging
from typing import Dict, Tuple, List, Optional, Union, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import ndimage, signal
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

class VibrationHeatmapVisualizer:
    """
    Class for visualizing ship micro-motion vibration frequencies as heatmaps.
    """
    
    def __init__(self, image_data: np.ndarray, ship_regions: List[Dict[str, Any]],
                vibration_data: Dict[str, np.ndarray]):
        """
        Initialize the visualizer.
        
        Parameters
        ----------
        image_data : np.ndarray
            The SAR image data.
        ship_regions : List[Dict[str, Any]]
            List of detected ship regions.
        vibration_data : Dict[str, np.ndarray]
            Dictionary containing vibration data.
        """
        self.image_data = image_data
        self.ship_regions = ship_regions
        self.vibration_data = vibration_data
        
    def plot_ship_detection_results(self, figsize: Tuple[int, int] = (12, 10),
                                   cmap: str = 'viridis') -> plt.Figure:
        """
        Plot ship detection results.
        
        Parameters
        ----------
        figsize : Tuple[int, int], optional
            Figure size, by default (12, 10)
        cmap : str, optional
            Colormap for the image, by default 'viridis'
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object.
        """
        # Convert to magnitude if complex
        if np.iscomplexobj(self.image_data):
            disp_image = np.log10(np.abs(self.image_data) + 1)
        else:
            disp_image = self.image_data
            
        # Normalize for display
        disp_image = (disp_image - np.min(disp_image)) / (np.max(disp_image) - np.min(disp_image))
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=figsize)
        
        # Display the image
        ax.imshow(disp_image, cmap=cmap)
        
        # Overlay bounding boxes for each ship
        for ship in self.ship_regions:
            min_row, min_col, max_row, max_col = ship['bbox']
            width = max_col - min_col
            height = max_row - min_row
            
            # Create a rectangle patch
            rect = patches.Rectangle((min_col, min_row), width, height, 
                                   linewidth=2, edgecolor='r', facecolor='none')
            
            # Add centroid marker
            centroid = ship['centroid']
            ax.plot(centroid[1], centroid[0], 'ro', markersize=5)
            
            # Add the patch to the axes
            ax.add_patch(rect)
            
        ax.set_title('Ship Detection Results')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        
        plt.tight_layout()
        return fig
        
    def create_vibration_heatmap(self, ship_index: int, 
                               freq_range: Tuple[float, float] = (10.0, 30.0),
                               window_size: int = 16,
                               freq_step: float = 0.5,
                               cmap: str = 'hot') -> Tuple[plt.Figure, np.ndarray]:
        """
        Create a heatmap of vibration frequencies for a single ship.
        
        Parameters
        ----------
        ship_index : int
            Index of the ship region to analyze.
        freq_range : Tuple[float, float], optional
            Frequency range to analyze in Hz, by default (10.0, 30.0)
        window_size : int, optional
            Size of the analysis window, by default 16
        freq_step : float, optional
            Step size for frequency analysis in Hz, by default 0.5
        cmap : str, optional
            Colormap for the heatmap, by default 'hot'
            
        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            The matplotlib figure object and the frequency heatmap.
        """
        if ship_index >= len(self.ship_regions):
            raise ValueError(f"Ship index {ship_index} out of range. Only {len(self.ship_regions)} ships detected.")
            
        # Get the ship region
        ship = self.ship_regions[ship_index]
        region = ship['region']
        mask = ship['mask']
        
        # Get region dimensions
        rows, cols = region.shape
        
        # Create frequency range
        min_freq, max_freq = freq_range
        freq_bins = int((max_freq - min_freq) / freq_step) + 1
        frequencies = np.linspace(min_freq, max_freq, freq_bins)
        
        # Initialize heatmap with NaN values (for non-ship pixels)
        frequency_heatmap = np.full((rows, cols), np.nan)
        power_heatmap = np.full((rows, cols), np.nan)
        
        # Extract time series data from vibration data
        times = self.vibration_data['vibration_params']['times']
        range_shifts = self.vibration_data['vibration_params']['filtered_range_shifts']
        azimuth_shifts = self.vibration_data['vibration_params']['filtered_azimuth_shifts']
        
        # Calculate sampling frequency
        sampling_freq = 1 / (times[1] - times[0])
        
        # Find ship pixel locations
        ship_pixels = np.where(mask)
        
        # If there are no ship pixels, return empty heatmap
        if len(ship_pixels[0]) == 0:
            logger.warning(f"No ship pixels found in region {ship_index}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            ax1.imshow(region, cmap='gray')
            ax1.set_title(f'Ship {ship_index}')
            ax2.set_title(f'No ship pixels found')
            plt.tight_layout()
            return fig, frequency_heatmap
            
        # Apply sliding window FFT only to windows containing ship pixels
        for i in range(0, rows - window_size + 1, window_size // 2):
            for j in range(0, cols - window_size + 1, window_size // 2):
                # Get window mask and check if it contains ship pixels
                window_mask = mask[i:i+window_size, j:j+window_size]
                ship_pixel_count = np.sum(window_mask)
                
                # Only process windows with ship pixels
                if ship_pixel_count > 0:
                    # Create time series (simplified - using global shifts)
                    local_signal = range_shifts + azimuth_shifts
                    
                    # Apply Hanning window
                    windowed_signal = local_signal * np.hanning(len(local_signal))
                    
                    # Compute FFT
                    n_samples = len(windowed_signal)
                    fft_result = np.abs(np.fft.fft(windowed_signal))[:n_samples//2]
                    fft_freqs = np.fft.fftfreq(n_samples, 1/sampling_freq)[:n_samples//2]
                    
                    # Find indices within our frequency range
                    freq_mask = (fft_freqs >= min_freq) & (fft_freqs <= max_freq)
                    valid_freqs = fft_freqs[freq_mask]
                    valid_fft = fft_result[freq_mask]
                    
                    if len(valid_fft) > 0:
                        # Find dominant frequency
                        max_idx = np.argmax(valid_fft)
                        dominant_freq = valid_freqs[max_idx]
                        power = valid_fft[max_idx]
                        
                        # ONLY assign to ship pixels in the window
                        for wi in range(window_size):
                            for wj in range(window_size):
                                if i+wi < rows and j+wj < cols and window_mask[wi, wj]:
                                    frequency_heatmap[i+wi, j+wj] = dominant_freq
                                    power_heatmap[i+wi, j+wj] = power
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot original image
        if np.iscomplexobj(region):
            disp_region = np.log10(np.abs(region) + 1)
        else:
            disp_region = region
        disp_region = (disp_region - np.min(disp_region)) / (np.max(disp_region) - np.min(disp_region))
        
        ax1.imshow(disp_region, cmap='gray')
        ax1.set_title(f'Ship {ship_index}')
        ax1.set_xlabel('Column')
        ax1.set_ylabel('Row')
        
        # Plot frequency heatmap with a custom masked array for clearer visualization
        masked_heatmap = np.ma.array(frequency_heatmap, mask=np.isnan(frequency_heatmap))
        im = ax2.imshow(masked_heatmap, cmap=cmap, vmin=min_freq, vmax=max_freq)
        ax2.set_title(f'Vibration Frequency Heatmap ({min_freq}-{max_freq} Hz)')
        ax2.set_xlabel('Column')
        ax2.set_ylabel('Row')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Frequency (Hz)')
        
        plt.tight_layout()
        return fig, frequency_heatmap
    
    def plot_vibration_spectra(self, ship_index: int = 0, 
                              max_freq: float = 50.0,
                              num_points: int = 5) -> plt.Figure:
        """
        Plot vibration spectra for multiple points on a ship.
        
        Parameters
        ----------
        ship_index : int, optional
            Index of the ship to analyze, by default 0
        max_freq : float, optional
            Maximum frequency to plot in Hz, by default 50.0
        num_points : int, optional
            Number of points to sample on the ship, by default 5
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object.
        """
        if ship_index >= len(self.ship_regions):
            raise ValueError(f"Ship index {ship_index} out of range. Only {len(self.ship_regions)} ships detected.")
            
        # Get the ship region
        ship = self.ship_regions[ship_index]
        region = ship['region']
        mask = ship['mask']
        
        # Find intensity values in the ship region
        if np.iscomplexobj(region):
            intensity = np.abs(region)
        else:
            intensity = region
            
        # Create a display image for visualization
        disp_region = np.log10(intensity + 1)
        disp_region = (disp_region - np.min(disp_region)) / (np.max(disp_region) - np.min(disp_region))
        
        # Identify high-intensity pixels (likely the actual ship)
        # Use a threshold at 70% of the maximum intensity after applying the mask
        masked_intensity = intensity.copy()
        masked_intensity[~mask] = 0
        if np.max(masked_intensity) > 0:
            threshold = 0.7 * np.max(masked_intensity)
            high_intensity_mask = masked_intensity > threshold
        else:
            # Fallback if no clear high intensity pixels
            high_intensity_mask = mask
        
        # Find ship pixels using the high intensity mask for better ship detection
        ship_pixels = np.where(high_intensity_mask)
        
        if len(ship_pixels[0]) == 0:
            # Fallback to original mask if no high intensity pixels found
            ship_pixels = np.where(mask)
            if len(ship_pixels[0]) == 0:
                raise ValueError("No ship pixels found in the mask.")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot ship region
        ax1.imshow(disp_region, cmap='gray')
        ax1.set_title(f'Ship {ship_index} with Measurement Points')
        
        # Extract vibration data
        frequencies = self.vibration_data['vibration_params']['frequencies']
        range_spectrum = self.vibration_data['vibration_params']['range_spectrum']
        azimuth_spectrum = self.vibration_data['vibration_params']['azimuth_spectrum']
        
        # Limit frequency range for display
        valid_idx = frequencies <= max_freq
        plot_freqs = frequencies[valid_idx]
        plot_range = range_spectrum[valid_idx]
        plot_azimuth = azimuth_spectrum[valid_idx]
        
        # Generate colors for points
        colors = plt.cm.tab10(np.linspace(0, 1, num_points))
        
        # If we have enough pixels, select points in a more structured way
        if len(ship_pixels[0]) >= num_points:
            # Find the centroid of the high intensity region
            centroid_y = np.mean(ship_pixels[0])
            centroid_x = np.mean(ship_pixels[1])
            
            # Calculate distances from centroid
            distances = np.sqrt((ship_pixels[0] - centroid_y)**2 + (ship_pixels[1] - centroid_x)**2)
            
            # Select points: 1 at center, others distributed from center to edge
            sorted_indices = np.argsort(distances)
            # Take the center point, then evenly spaced points outward
            selected_indices = [sorted_indices[0]]  # Center point
            if num_points > 1:
                # Add points from different parts of the ship
                edge_indices = sorted_indices[1:]
                step = len(edge_indices) // (num_points - 1)
                if step == 0:
                    step = 1
                selected_indices.extend(edge_indices[::step][:num_points-1])
            
            # Ensure we have the right number of points
            selected_indices = selected_indices[:num_points]
        else:
            # If we don't have enough pixels, use all available
            selected_indices = list(range(len(ship_pixels[0])))
            num_points = len(selected_indices)
        
        # Plot frequency spectra for selected points
        for i, idx in enumerate(selected_indices):
            row, col = ship_pixels[0][idx], ship_pixels[1][idx]
            
            # Plot the point on the image
            ax1.plot(col, row, 'o', color=colors[i], markersize=8, label=f'Point {i+1}')
            
            # Plot the spectrum for this point
            # In a real implementation, you would use point-specific spectra
            # Here we're using the global spectrum as a demonstration
            ax2.plot(plot_freqs, plot_range, '-', color=colors[i], label=f'Point {i+1} (Range)')
            ax2.plot(plot_freqs, plot_azimuth, '--', color=colors[i], alpha=0.5, label=f'Point {i+1} (Azimuth)')
        
        # Highlight the high intensity area for visualization
        ax1.contour(high_intensity_mask, levels=[0.5], colors='red', alpha=0.7)
        
        ax2.set_title('Vibration Frequency Spectra')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True)
        ax2.set_xlim(0, max_freq)
        
        # Create a separate legend for range and azimuth
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def create_combined_visualization(self, ship_index: int = 0,
                                    freq_range: Tuple[float, float] = (10.0, 30.0)) -> plt.Figure:
        """
        Create a combined visualization of ship detection and vibration analysis.
        
        Parameters
        ----------
        ship_index : int, optional
            Index of the ship to analyze, by default 0
        freq_range : Tuple[float, float], optional
            Frequency range for the heatmap in Hz, by default (10.0, 30.0)
            
        Returns
        -------
        plt.Figure
            The matplotlib figure object.
        """
        # Create figure
        fig = plt.figure(figsize=(18, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 3)
        
        # Full SAR image with detection overlays
        ax_full = fig.add_subplot(gs[0, :])
        
        # Convert to magnitude if complex
        if np.iscomplexobj(self.image_data):
            disp_image = np.log10(np.abs(self.image_data) + 1)
        else:
            disp_image = self.image_data
            
        # Normalize for display
        disp_image = (disp_image - np.min(disp_image)) / (np.max(disp_image) - np.min(disp_image))
        
        # Display the image
        ax_full.imshow(disp_image, cmap='gray')
        
        # Overlay bounding boxes for each ship
        for i, ship in enumerate(self.ship_regions):
            min_row, min_col, max_row, max_col = ship['bbox']
            width = max_col - min_col
            height = max_row - min_row
            
            # Create a rectangle patch with different color for the selected ship
            if i == ship_index:
                rect = patches.Rectangle((min_col, min_row), width, height, 
                                       linewidth=3, edgecolor='r', facecolor='none')
            else:
                rect = patches.Rectangle((min_col, min_row), width, height, 
                                       linewidth=1, edgecolor='yellow', facecolor='none')
            
            # Add the patch to the axes
            ax_full.add_patch(rect)
            
            # Add ship index label
            ax_full.text(min_col, min_row-5, f"Ship {i}", color='white', 
                       fontsize=10, backgroundcolor='black')
        
        ax_full.set_title('SAR Image with Ship Detections')
        
        # Get the selected ship
        ship = self.ship_regions[ship_index]
        region = ship['region']
        mask = ship['mask']
        
        # Ship zoom view
        ax_zoom = fig.add_subplot(gs[1, 0])
        
        if np.iscomplexobj(region):
            disp_region = np.log10(np.abs(region) + 1)
        else:
            disp_region = region
        disp_region = (disp_region - np.min(disp_region)) / (np.max(disp_region) - np.min(disp_region))
        
        ax_zoom.imshow(disp_region, cmap='gray')
        ax_zoom.set_title(f'Ship {ship_index} Zoom')
        
        # Vibration time series
        ax_time = fig.add_subplot(gs[1, 1])
        
        times = self.vibration_data['vibration_params']['times']
        range_shifts = self.vibration_data['vibration_params']['range_shifts']
        azimuth_shifts = self.vibration_data['vibration_params']['azimuth_shifts']
        filtered_range = self.vibration_data['vibration_params']['filtered_range_shifts']
        filtered_azimuth = self.vibration_data['vibration_params']['filtered_azimuth_shifts']
        
        ax_time.plot(times, range_shifts, 'b-', alpha=0.5, label='Range Shifts')
        ax_time.plot(times, azimuth_shifts, 'g-', alpha=0.5, label='Azimuth Shifts')
        ax_time.plot(times, filtered_range, 'b-', linewidth=2, label='Filtered Range')
        ax_time.plot(times, filtered_azimuth, 'g-', linewidth=2, label='Filtered Azimuth')
        
        ax_time.set_title('Vibration Time Series')
        ax_time.set_xlabel('Time (s)')
        ax_time.set_ylabel('Displacement')
        ax_time.grid(True)
        ax_time.legend()
        
        # Vibration frequency spectra
        ax_freq = fig.add_subplot(gs[1, 2])
        
        frequencies = self.vibration_data['vibration_params']['frequencies']
        range_spectrum = self.vibration_data['vibration_params']['range_spectrum']
        azimuth_spectrum = self.vibration_data['vibration_params']['azimuth_spectrum']
        
        # Limit frequency range for display
        valid_idx = frequencies <= 50.0  # Display up to 50 Hz
        plot_freqs = frequencies[valid_idx]
        plot_range = range_spectrum[valid_idx]
        plot_azimuth = azimuth_spectrum[valid_idx]
        
        ax_freq.plot(plot_freqs, plot_range, 'b-', label='Range Spectrum')
        ax_freq.plot(plot_freqs, plot_azimuth, 'g-', label='Azimuth Spectrum')
        
        # Highlight the frequency range of interest
        min_freq, max_freq = freq_range
        ax_freq.axvspan(min_freq, max_freq, alpha=0.2, color='red')
        
        # Mark dominant frequencies
        dominant_freq_range = self.vibration_data['vibration_params']['dominant_freq_range']
        dominant_freq_azimuth = self.vibration_data['vibration_params']['dominant_freq_azimuth']
        
        if dominant_freq_range <= 50.0:
            idx_range = np.abs(plot_freqs - dominant_freq_range).argmin()
            ax_freq.plot(dominant_freq_range, plot_range[idx_range], 'bo', markersize=8)
            ax_freq.text(dominant_freq_range, plot_range[idx_range], 
                       f' {dominant_freq_range:.1f} Hz', verticalalignment='bottom')
            
        if dominant_freq_azimuth <= 50.0:
            idx_azimuth = np.abs(plot_freqs - dominant_freq_azimuth).argmin()
            ax_freq.plot(dominant_freq_azimuth, plot_azimuth[idx_azimuth], 'go', markersize=8)
            ax_freq.text(dominant_freq_azimuth, plot_azimuth[idx_azimuth], 
                       f' {dominant_freq_azimuth:.1f} Hz', verticalalignment='bottom')
        
        ax_freq.set_title('Vibration Frequency Spectra')
        ax_freq.set_xlabel('Frequency (Hz)')
        ax_freq.set_ylabel('Amplitude')
        ax_freq.grid(True)
        ax_freq.set_xlim(0, 50.0)
        ax_freq.legend()
        
        plt.tight_layout()
        return fig 