
# Root __init__.py (multimodal_stimulus_fmri_predict/__init__.py)
"""
Multimodal fMRI Stimulus Prediction Framework

A flexible, extensible framework for experimenting with different classifiers
on multimodal fMRI data following SOLID principles and factory design patterns.
"""

from . import config
from . import data
from . import models
from . import utils
from . import experiments

from .models import ClassifierFactory
from .experiments import ExperimentRunner
from .data import FMRIDataLoader
from .utils import ModelEvaluator

__version__ = '1.0.0'
__author__ = 'fMRI Analysis Team'
__email__ = 'fmri-team@research.org'

__all__ = [
    # Main modules
    'config',
    'data', 
    'models',
    'utils',
    'experiments',
    
    # Key classes for direct import
    'ClassifierFactory',
    'ExperimentRunner',
    'FMRIDataLoader',
    'ModelEvaluator'
]


def get_version():
    """Return the current version of the package."""
    return __version__


def list_available_classifiers():
    """List all available classifier types."""
    return ClassifierFactory.get_available_classifiers()


def create_quick_experiment(config_path: str, experiment_name: str, subjects: list):
    """Quick experiment setup utility.
    
    Args:
        config_path: Path to experiment configuration file
        experiment_name: Name of experiment to run
        subjects: List of subject IDs
        
    Returns:
        ExperimentRunner instance ready to execute
    """
    runner = ExperimentRunner(config_path)
    return runner


# Package-level configuration
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('fmri_analysis.log')
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Multimodal fMRI Stimulus Prediction Framework v{__version__} initialized")