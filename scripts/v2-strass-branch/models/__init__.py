# models/__init__.py
from typing import Dict, Any
from models.base_classifier import BaseClassifier
from models.classical.svm import SVMClassifier
from models.classical.random_forest import RandomForestClassifier
from models.classical.logistic_regression import LogisticRegressionClassifier
from models.neural.mlp import MLPClassifier
from models.neural.cnn import CNNClassifier
from models.neural.lstm import LSTMClassifier
from models.neural.transformer import TransformerClassifier
from attention import SelfAttentionClassifier  # ← Added
#(Need MLP?)

class ClassifierFactory:
    """Factory for creating different classifier instances"""
    
    _classifiers = {
        'svm': SVMClassifier,
        'random_forest': RandomForestClassifier,
        'logistic_regression': LogisticRegressionClassifier,
        'mlp': MLPClassifier,
        'cnn': CNNClassifier,
        'lstm': LSTMClassifier,
        'transformer': TransformerClassifier,
        'self_attention': SelfAttentionClassifier,  # ← Added 
        #(Need MLP?)
    }
    
    @classmethod
    def create_classifier(cls, classifier_type: str, config: Dict[str, Any]) -> BaseClassifier:
        """Create a classifier instance"""
        if classifier_type not in cls._classifiers:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
        
        return cls._classifiers[classifier_type](config)
    
    @classmethod
    def get_available_classifiers(cls) -> list:
        """Get list of available classifier types"""
        return list(cls._classifiers.keys())