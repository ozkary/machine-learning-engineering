# your_package/__init__.py
from .__version__ import __version__
from .data_train import DDIProcessData, DDIModelFactory
from .data_train_mlp import DDIMLPFactory, DDIProcessor
from .data_predict import DDIPredictor, DDIModelLoader, predict, load_test_cases

print(f"Initializing drug-drug interaction ddi_lib {__version__}")