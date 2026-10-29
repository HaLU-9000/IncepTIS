"""Public inference API for the IncepTIS exp038 ensemble."""

from .predictor import CONTEXT_LENGTH, InceptisPredictor, Prediction

__all__ = ["CONTEXT_LENGTH", "InceptisPredictor", "Prediction"]
__version__ = "0.1.0"
