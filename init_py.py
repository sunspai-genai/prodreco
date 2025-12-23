"""
Hybrid Product Recommendation System
Wide & Deep Learning for Business Banking
"""

__version__ = "1.0.0"
__author__ = "Banking AI Team"

from .pipeline import HybridRecommendationPipeline
from .data_generator import BankingDataGenerator
from .preprocessor import DataPreprocessor
from .model_architecture import WideDeepModel
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .hyperparameter_tuner import HyperparameterTuner
from .recommender import ProductRecommender
from .explainer import ModelExplainer

__all__ = [
    'HybridRecommendationPipeline',
    'BankingDataGenerator',
    'DataPreprocessor',
    'WideDeepModel',
    'ModelTrainer',
    'ModelEvaluator',
    'HyperparameterTuner',
    'ProductRecommender',
    'ModelExplainer'
]
