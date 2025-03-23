# Ship Detection and Micro-Motion Processing

This project processes SAR (Synthetic Aperture Radar) data to detect ships and analyze their micro-motion vibration frequencies. The analysis is based on the approach described in the paper *"Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment"*.

## Overview

The tool performs the following key functions:

1. **Ship Detection**: Identifies ships in SAR imagery using adaptive thresholding and CFAR detection methods.
2. **Doppler Sub-aperture Processing**: Analyzes micro-motions by splitting raw CPHD data into time-domain Doppler sub-apertures.
3. **Vibration Analysis**: Computes vibration frequency spectra in the 10-30Hz range typical for ship engines.
4. **Visualization**: Generates heatmaps showing the distribution of vibration frequencies across ship structures.

## Requirements

- Python 3.12+
- SarPy (1.3.50+) 
- NumPy
- SciPy
- Matplotlib
- scikit-image
- scikit-learn

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/ship-detection-processing.git
   cd ship-detection-processing
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   pip install -e .
   ```

## Data Requirements

The tool can process:

- CPHD files (preferred for full vibration analysis)
- SICD files (ship detection only)
- NITF files containing SICD data

Example data files can be found in the `sample_data` directory.

## Usage

### Basic Usage

```
python main.py /path/to/your/sar_file.cphd
```

### Options

```
python main.py /path/to/your/sar_file.cphd --output-dir results --verbose
```

- `--output-dir` or `-o`: Directory to save results (default: results)
- `--log-file` or `-l`: Path to log file (optional)
- `--verbose` or `-v`: Enable verbose output

### Output

The program generates:

1. Ship detection masks and bounding boxes
2. Vibration analysis spectrograms
3. Frequency heatmaps for each detected ship
4. Combined visualization with spectral information
5. Numerical data saved as .npz files

All results are saved to the specified output directory.

## Project Structure

- `src/ship_detection/`: Main package
  - `io/`: Data input/output modules
  - `processing/`: Ship detection and vibration analysis
  - `visualization/`: Heatmap and result visualization
  - `utils/`: Utility functions
  - `processor.py`: Main processing pipeline

## Technical Approach

### Ship Detection

The ship detection module employs two primary methods:
1. **Adaptive Thresholding**: Uses Otsu's method to find optimal threshold values
2. **CFAR (Constant False Alarm Rate)**: Detects targets by comparing them to surrounding background

### Micro-Motion Analysis

The approach follows the paper methodology:
1. Split raw CPHD data into ~200 temporal sub-apertures
2. Focus each sub-aperture into an image
3. Apply sub-pixel coregistration to track small movements between consecutive frames
4. Analyze the vibration frequency spectra using FFT

### Limitations

- Full vibration analysis requires CPHD data with raw signal information
- SICD files only support ship detection, not vibration analysis
- The current focusing algorithm is simplified compared to production-quality SAR processors

## Acknowledgements

This project uses SarPy, a Python library developed by NGA to work with SICD and CPHD formatted data.

## License

[MIT License](LICENSE)

## References

- Bianco, P., Cappuccio, P., Orlando, D., & Schirinzi, G. (2019). Micro-Motion Estimation of Maritime Targets Using Pixel Tracking in Cosmo-Skymed Synthetic Aperture Radar Data—An Operative Assessment.
