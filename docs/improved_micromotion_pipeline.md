# Improved Ship Micro-Motion Analysis Pipeline

## 1. Introduction and Background

Ship micro-motion analysis in Synthetic Aperture Radar (SAR) data involves detecting and characterizing small-scale vibrations and movements of vessels. These vibrations can reveal crucial information about:

- Engine characteristics and operational status
- Vessel type identification
- Activity patterns (idle, loading/unloading, active operations)
- Structural properties of the vessel

The current implementation uses a simplified Doppler sub-aperture approach where:
1. The SAR data is split into temporal sub-apertures
2. Global shifts are calculated across the entire scene
3. The same vibration spectrum is applied to all points on a ship
4. A single bandpass filter is applied to all data

This approach has several limitations that reduce accuracy and physical realism.

## 2. Limitations of Current Approach

### 2.1 Signal Processing Limitations

- **Global shift analysis**: The current approach calculates global shifts between sub-apertures rather than pixel-specific shifts
- **Identical frequency analysis**: Every point on a ship is assigned the same vibration spectrum
- **Limited frequency resolution**: Current time-series length constrains frequency resolution
- **Generic bandpass filtering**: Uses same filter parameters for all ships regardless of size or type
- **No compensation for surrounding clutter**: Water motion can contaminate vibration measurements

### 2.2 Visualization and Measurement Limitations

- **Imprecise ship detection**: Mask may include water/clutter pixels
- **Uniform vibration assumption**: The whole ship is treated as vibrating uniformly
- **Limited frequency range**: Fixed analysis range may miss important frequencies
- **No connection to physical ship structure**: Doesn't account for expected vibration patterns

## 3. Proposed Improved Pipeline

The improved pipeline incorporates advanced signal processing techniques and physical models to generate more accurate, point-specific vibration measurements.

### 3.1 Overview of Processing Steps

1. **Enhanced SAR Data Preprocessing**
   - Improved radiometric calibration
   - Speckle reduction while preserving micro-motion information
   - Range and azimuth oversampling for finer motion detection

2. **Advanced Ship Detection and Segmentation**
   - Multi-scale convolutional neural network for precise ship detection
   - Semantic segmentation to identify different ship components (bow, stern, hull, superstructure)
   - Structural feature extraction to inform vibration analysis
   
3. **Pixel-Level Micro-Motion Analysis**
   - Time-series extraction for each individual pixel
   - Local phase history analysis instead of global shifts
   - Coherent and non-coherent integration techniques
   
4. **Adaptive Time-Frequency Analysis**
   - Wavelet decomposition for multi-resolution analysis
   - Short-time Fourier transform with adaptive window sizing
   - Empirical mode decomposition for non-stationary signals
   
5. **Physics-Based Constraints and Modeling**
   - Ship structure models to constrain vibration patterns
   - Coupled vibration analysis between connected ship components
   - Removal of sea-state-induced apparent motion
   
6. **Unified Visualization Framework**
   - Component-specific vibration mapping
   - Confidence metrics for detected vibrations
   - Comparison with reference databases of known vessel types

### 3.2 Detailed Signal Processing Chain

```
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  SAR Raw Data    │────▶│  Focused Complex  │────▶│  Ship Detection  │
│  Preprocessing   │     │  Image Formation  │     │  & Segmentation  │
└──────────────────┘     └───────────────────┘     └────────┬─────────┘
                                                            │
                                                            ▼
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Visualization   │◀────│  Ship Component   │◀────│  Sub-Aperture    │
│  & Reporting     │     │  Classification   │     │  Generation      │
└──────────────────┘     └───────────────────┘     └────────┬─────────┘
        ▲                                                   │
        │                                                   ▼
┌──────────────────┐     ┌───────────────────┐     ┌──────────────────┐
│  Physical Model  │◀────│  Vibration Mode   │◀────│  Pixel-Specific  │
│  Integration     │     │  Analysis         │     │  Phase History   │
└──────────────────┘     └───────────────────┘     └──────────────────┘
```

## 4. Key Technical Improvements

### 4.1 Pixel-Specific Phase History Analysis

The core improvement is shifting from global shift analysis to pixel-specific phase history:

```python
def extract_pixel_phase_history(pixel_row, pixel_col, subapertures):
    """Extract phase history for a specific pixel across all subapertures."""
    phase_history = []
    for subaperture in subapertures:
        # Extract complex value for this pixel from each subaperture
        complex_value = subaperture[pixel_row, pixel_col]
        # Track phase changes
        phase = np.angle(complex_value)
        phase_history.append(phase)
    return np.array(phase_history)
```

### 4.2 Advanced Time-Frequency Analysis

Multiple time-frequency analysis techniques can be applied and compared:

```python
def multi_method_time_frequency_analysis(signal, sampling_freq):
    """Apply multiple time-frequency analysis methods and compare results."""
    results = {}
    
    # Standard FFT with windowing
    windowed_signal = signal * np.hanning(len(signal))
    fft_result = np.abs(np.fft.fft(windowed_signal))
    fft_freqs = np.fft.fftfreq(len(signal), 1/sampling_freq)
    results['fft'] = {'freqs': fft_freqs, 'spectrum': fft_result}
    
    # Continuous Wavelet Transform
    widths = np.arange(1, 31)
    cwt_result = signal.cwt(signal, signal.ricker, widths)
    results['cwt'] = {'scales': widths, 'coefficients': cwt_result}
    
    # Short-Time Fourier Transform with adaptive window
    window_size = min(len(signal) // 4, 128)
    f, t, stft_result = signal.stft(signal, sampling_freq, nperseg=window_size)
    results['stft'] = {'freqs': f, 'times': t, 'coefficients': stft_result}
    
    return results
```

### 4.3 Ship Component Classification

Different ship components have characteristic vibration patterns:

```python
def classify_ship_components(ship_region, intensity_mask):
    """Segment a ship into components based on image features."""
    # Create empty component map
    component_map = np.zeros_like(ship_region, dtype=int)
    
    # Extract features for classification
    gradient_x = ndimage.sobel(ship_region, axis=1)
    gradient_y = ndimage.sobel(ship_region, axis=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    
    # Use thresholding and morphological operations to identify:
    # Hull (1), Deck (2), Superstructure (3), Bow (4), Stern (5)
    
    # Example: Identify bow (usually at one end with high gradients)
    # (Simplified example - actual implementation would be more sophisticated)
    rows, cols = ship_region.shape
    left_quarter = int(cols * 0.25)
    right_quarter = int(cols * 0.75)
    
    # Check gradient patterns to determine which end is bow
    left_gradients = np.sum(gradient_magnitude[:, :left_quarter])
    right_gradients = np.sum(gradient_magnitude[:, right_quarter:])
    
    if left_gradients > right_gradients:
        # Bow is on the left
        component_map[:, :left_quarter][intensity_mask[:, :left_quarter]] = 4
        component_map[:, right_quarter:][intensity_mask[:, right_quarter:]] = 5
    else:
        # Bow is on the right
        component_map[:, right_quarter:][intensity_mask[:, right_quarter:]] = 4
        component_map[:, :left_quarter][intensity_mask[:, :left_quarter]] = 5
    
    # Additional component identification would follow
    # ...
    
    return component_map
```

### 4.4 Physics-Based Vibration Constraints

Applying physical constraints to ensure realistic vibration patterns:

```python
def apply_physical_constraints(component_map, vibration_frequencies):
    """Apply physical constraints to detected vibrations based on ship components."""
    constrained_vibrations = vibration_frequencies.copy()
    
    # Define expected frequency ranges for different components
    component_freq_ranges = {
        1: (5, 15),    # Hull: 5-15 Hz
        2: (8, 20),    # Deck: 8-20 Hz
        3: (10, 30),   # Superstructure: 10-30 Hz
        4: (12, 25),   # Bow: 12-25 Hz
        5: (8, 18)     # Stern (engine area): 8-18 Hz
    }
    
    # Apply component-specific constraints
    for component_id, freq_range in component_freq_ranges.items():
        component_mask = component_map == component_id
        if not np.any(component_mask):
            continue
            
        min_freq, max_freq = freq_range
        
        # For each pixel in this component
        for idx in zip(*np.where(component_mask)):
            pixel_freqs = vibration_frequencies[idx]
            
            # Filter out frequencies outside the expected range
            valid_mask = (pixel_freqs >= min_freq) & (pixel_freqs <= max_freq)
            if not np.any(valid_mask):
                # If no frequencies in expected range, use closest valid frequency
                closest_idx = np.argmin(np.abs(pixel_freqs - np.mean(freq_range)))
                valid_mask[closest_idx] = True
                
            # Zero out invalid frequencies
            constrained_freqs = pixel_freqs.copy()
            constrained_freqs[~valid_mask] = 0
            constrained_vibrations[idx] = constrained_freqs
    
    return constrained_vibrations
```

## 5. Implementation Details

### 5.1 Software Architecture

The new pipeline should be implemented as a set of modular components:

1. **AdvancedSARPreprocessor**: Enhanced calibration and preprocessing
2. **ShipSegmentationEngine**: Neural network-based ship detection and segmentation
3. **PixelPhaseHistoryExtractor**: Pixel-level phase history analysis
4. **MultiModalTimeFrequencyAnalyzer**: Advanced frequency analysis
5. **ShipComponentClassifier**: Ship structure analysis
6. **PhysicalConstraintSolver**: Applying physical models
7. **MicroMotionVisualizer**: Enhanced visualization tools

These components should communicate through well-defined interfaces to allow for future improvements and extensions.

### 5.2 Computational Considerations

The pixel-level analysis will significantly increase computational requirements. To address this:

- Implement parallel processing for pixel-level operations
- Use GPU acceleration for neural network components
- Apply initial filtering to focus processing on high-probability ship pixels
- Employ incremental processing to handle large datasets

### 5.3 Validation Methodology

The improved pipeline should be validated using:

1. Simulated SAR data with known vibration patterns
2. Real SAR data with ground truth measurements from ships
3. Comparison with existing techniques from literature
4. Expert assessment of physical plausibility of results

## 6. Expected Improvements

The proposed pipeline is expected to deliver:

1. **Higher accuracy**: More precise vibration frequency measurements
2. **Physical realism**: Vibration patterns consistent with ship structures
3. **Component-specific analysis**: Different frequencies for different ship parts
4. **Improved visualization**: Clearer presentation of results with confidence metrics
5. **Better classification**: Enhanced ability to distinguish vessel types
6. **Noise rejection**: Reduced false detections from water motion and clutter

## 7. Future Extensions

The modular architecture allows for future extensions:

1. Integration of multiple SAR data sources (multi-sensor fusion)
2. Temporal analysis across multiple acquisitions
3. Machine learning for automatic ship type classification based on vibration signatures
4. Integration with AIS data for validation
5. Real-time monitoring capabilities for maritime surveillance

## 8. References

1. Martorella, M., Giusti, E., Bacci, A., Cataldo, D., & Pastina, D. (2019). "Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment"
2. Yamaguchi, Y. (2020). "Disaster and Environmental Monitoring Using ALOS/PALSAR"
3. Pelich, R., Chini, M., Hostache, R., Matgen, P., & Lopez-Martinez, C. (2021). "Detection of Vessels in SAR Imagery Using Deep Learning"
4. Gao, G., & Shi, G. (2022). "Ship Detection in Polarimetric SAR Images via Spatial Pyramid Attention Network"
5. Marbouti, M., Praks, J., Antropov, O., Rinne, E., & Leppäranta, M. (2018). "Detection of Sea Ice Surface" 