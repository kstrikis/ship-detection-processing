# Improved Ship Micro-Motion Analysis Pipeline

This directory contains a modular implementation of the improved ship micro-motion analysis pipeline as described in `docs/improved_micromotion_pipeline.md`.

## Key Improvements

1. **Pixel-Level Phase History**: Analysis of individual pixel vibrations instead of global shifts
2. **Advanced Time-Frequency Analysis**: Multiple analysis methods including STFT, CWT, and adaptive windowing
3. **Component Classification**: Identification of different ship components (hull, deck, superstructure, etc.)
4. **Physics-Based Constraints**: Application of physical constraints to vibration patterns
5. **Modular Architecture**: Each step can be run in isolation with intermediate results saved to disk

## Directory Structure

```
improved_pipeline/
├── __init__.py               - Package initialization
├── main.py                   - Entrypoint script
├── preprocessor.py           - SAR data preprocessing
├── ship_detector.py          - Ship detection and segmentation
├── manual_selection.py       - Manual ship selection utility
├── phase_extractor.py        - Pixel-level phase history extraction
├── time_frequency_analyzer.py - Advanced time-frequency analysis
├── component_classifier.py   - Ship component classification
├── physics_constraints.py    - Physics-based constraints
├── visualizer.py             - Result visualization
└── utils.py                  - Common utilities
```

## Usage

### Full Pipeline

Run the complete pipeline:

```bash
python -m src.improved_pipeline.main --input /path/to/sar_data.cphd --output-dir results
```

### Individual Steps

Each step can be run independently:

1. **Preprocess SAR Data**:
   ```bash
   python -m src.improved_pipeline.preprocessor --input /path/to/sar_data.cphd --output results/01_preprocessed.npz
   ```

2. **Detect Ships**:
   ```bash
   python -m src.improved_pipeline.ship_detector --input results/01_preprocessed.npz --output results/02_detected_ships.npz
   ```

   or manually select ships:
   ```bash
   python -m src.improved_pipeline.manual_selection --input results/01_preprocessed.npz --output results/02_manual_ships.npz
   ```

3. **Extract Phase History**:
   ```bash
   python -m src.improved_pipeline.phase_extractor --input results/01_preprocessed.npz --ship-file results/02_detected_ships.npz --output results/03_phase_history.npz
   ```

4. **Analyze Time-Frequency**:
   ```bash
   python -m src.improved_pipeline.time_frequency_analyzer --input results/03_phase_history.npz --output results/04_time_frequency.npz
   ```

5. **Classify Components**:
   ```bash
   python -m src.improved_pipeline.component_classifier --input results/03_phase_history.npz --ship-file results/02_detected_ships.npz --output results/05_components.npz
   ```

6. **Apply Physics Constraints**:
   ```bash
   python -m src.improved_pipeline.physics_constraints --input results/04_time_frequency.npz --component-file results/05_components.npz --output results/06_physics_constrained.npz
   ```

7. **Create Visualizations**:
   ```bash
   python -m src.improved_pipeline.visualizer --preprocessed-file results/01_preprocessed.npz --ship-file results/02_detected_ships.npz --component-file results/05_components.npz --vibration-file results/06_physics_constrained.npz --output-dir results/visualizations
   ```

### Using the Pipeline in Code

```python
from src.improved_pipeline import run_pipeline

# Run the full pipeline
result = run_pipeline(
    input_file="/path/to/sar_data.cphd",
    output_dir="results",
    use_gpu=True
)

# Or run individual steps
from src.improved_pipeline import preprocess_sar_data, detect_ships

preprocessed = preprocess_sar_data(
    input_file="/path/to/sar_data.cphd",
    output_file="results/preprocessed.npz"
)

ships = detect_ships(
    input_file="results/preprocessed.npz",
    output_file="results/ships.npz"
)
```

## Notes for Developers

- Each module can be run standalone or imported for use in other scripts
- Intermediate results are saved in NumPy's `.npz` format for efficient storage
- GPU acceleration is available for computationally intensive steps if CuPy is installed
- Manual intervention is supported at any stage of the pipeline

## Dependencies

- NumPy
- SciPy
- Matplotlib
- sarpy (for reading SAR data formats)
- CuPy (optional, for GPU acceleration)

## Troubleshooting

- If you encounter memory issues with large datasets, try using the crop functionality:
  ```bash
  python -m src.improved_pipeline.preprocessor --input /path/to/sar_data.cphd --output results/cropped.npz --crop-only
  ```

- For issues with automated ship detection, use the manual selection tool:
  ```bash
  python -m src.improved_pipeline.manual_selection --input results/01_preprocessed.npz --output results/02_manual_ships.npz
  ``` 