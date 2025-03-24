"""
Improved ship micro-motion analysis pipeline package.

This package implements the enhanced pipeline for ship detection and 
micro-motion analysis described in the improved_micromotion_pipeline.md document.

Key features:
- Pixel-level phase history extraction
- Component-specific vibration analysis
- Advanced time-frequency analysis
- Physics-based constraints
- Modular, step-by-step execution
"""

__version__ = '0.1.0'

# Import public interfaces
from .main import run_pipeline
from .preprocessor import preprocess_sar_data
from .ship_detector import detect_ships
from .manual_selection import manually_select_ships
from .phase_extractor import extract_phase_history
from .time_frequency_analyzer import analyze_time_frequency
from .component_classifier import classify_components
from .physics_constraints import apply_physics_constraints
from .visualizer import create_visualizations
from .utils import setup_logging, save_results, load_step_output 