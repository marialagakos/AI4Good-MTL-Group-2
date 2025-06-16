# config/model_configs.py
"""
Model-specific configuration settings for different classifier types.

This module provides optimized hyperparameters for various models used in
multimodal fMRI stimulus prediction, designed to potentially outperform
the Algonauts baseline.
"""

from typing import Dict, Any, List
import numpy as np


class ModelConfigs:
    """
    Model configuration repository following the Open/Closed Principle.
    
    New model configurations can be added without modifying existing ones.
    """
    
    @staticmethod
    def get_svm_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get SVM configuration optimized for fMRI data.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            SVM configuration dictionary
        """
        configs = {
            'fast': {
                'C': 1.0,
                'kernel': 'linear',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42,
                'max_iter': 1000,
                'cache_size': 200
            },
            'balanced': {
                'C': 10.0,
                'kernel': 'rbf',
                'gamma': 'scale',
                'probability': True,
                'random_state': 42,
                'max_iter': 3000,
                'cache_size': 500
            },
            'accurate': {
                'C': [0.1, 1.0, 10.0, 100.0],  # For grid search
                'kernel': ['rbf', 'linear'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'probability': True,
                'random_state': 42,
                'max_iter': 5000,
                'cache_size': 1000
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_random_forest_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get Random Forest configuration optimized for high-dimensional fMRI data.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Random Forest configuration dictionary
        """
        configs = {
            'fast': {
                'n_estimators': 50,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True
            },
            'balanced': {
                'n_estimators': 200,
                'max_depth': 20,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True
            },
            'accurate': {
                'n_estimators': 500,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'oob_score': True
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_logistic_regression_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get Logistic Regression configuration for multimodal features.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Logistic Regression configuration dictionary
        """
        configs = {
            'fast': {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 1000,
                'multi_class': 'auto',
                'random_state': 42
            },
            'balanced': {
                'C': 10.0,
                'penalty': 'l2',
                'solver': 'lbfgs',
                'max_iter': 2000,
                'multi_class': 'auto',
                'random_state': 42,
                'class_weight': 'balanced'
            },
            'accurate': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': 'saga',  # Supports all penalties
                'max_iter': 5000,
                'multi_class': 'auto',
                'random_state': 42,
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # For elasticnet
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_mlp_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get MLP configuration optimized for multimodal fMRI prediction.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            MLP configuration dictionary
        """
        configs = {
            'fast': {
                'hidden_dims': [256, 128],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'n_epochs': 50,
                'batch_size': 64,
                'activation': 'relu',
                'use_batch_norm': True,
                'optimizer': 'adam',
                'weight_decay': 1e-4,
                'early_stopping_patience': 10
            },
            'balanced': {
                'hidden_dims': [512, 256, 128],
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'n_epochs': 100,
                'batch_size': 32,
                'activation': 'relu',
                'use_batch_norm': True,
                'use_residual': False,
                'optimizer': 'adam',
                'weight_decay': 1e-4,
                'scheduler_type': 'plateau',
                'early_stopping_patience': 15
            },
            'accurate': {
                'hidden_dims': [1024, 512, 256, 128],
                'dropout_rate': 0.4,
                'learning_rate': 0.0005,
                'n_epochs': 200,
                'batch_size': 16,
                'activation': 'gelu',
                'use_batch_norm': True,
                'use_residual': True,
                'optimizer': 'adamw',
                'weight_decay': 1e-3,
                'scheduler_type': 'cosine',
                'early_stopping_patience': 20
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_cnn_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get CNN configuration for spatial patterns in fMRI data.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            CNN configuration dictionary
        """
        configs = {
            'fast': {
                'n_filters': [32, 64],
                'kernel_size': 3,
                'dropout_rate': 0.4,
                'learning_rate': 0.001,
                'n_epochs': 50,
                'batch_size': 32,
                'weight_decay': 1e-4
            },
            'balanced': {
                'n_filters': [64, 128, 256],
                'kernel_size': 3,
                'dropout_rate': 0.5,
                'learning_rate': 0.001,
                'n_epochs': 100,
                'batch_size': 16,
                'weight_decay': 1e-4,
                'early_stopping_patience': 15
            },
            'accurate': {
                'n_filters': [64, 128, 256, 512],
                'kernel_size': [3, 5],
                'dropout_rate': 0.5,
                'learning_rate': 0.0005,
                'n_epochs': 200,
                'batch_size': 8,
                'weight_decay': 1e-3,
                'early_stopping_patience': 20
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_lstm_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get LSTM configuration for temporal patterns in fMRI data.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            LSTM configuration dictionary
        """
        configs = {
            'fast': {
                'hidden_size': 64,
                'num_layers': 1,
                'dropout_rate': 0.3,
                'bidirectional': False,
                'sequence_length': 20,
                'learning_rate': 0.001,
                'n_epochs': 50,
                'batch_size': 32
            },
            'balanced': {
                'hidden_size': 128,
                'num_layers': 2,
                'dropout_rate': 0.3,
                'bidirectional': True,
                'sequence_length': 50,
                'learning_rate': 0.001,
                'n_epochs': 100,
                'batch_size': 16,
                'early_stopping_patience': 15
            },
            'accurate': {
                'hidden_size': 256,
                'num_layers': 3,
                'dropout_rate': 0.4,
                'bidirectional': True,
                'sequence_length': 100,
                'learning_rate': 0.0005,
                'n_epochs': 200,
                'batch_size': 8,
                'early_stopping_patience': 20
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_transformer_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get Transformer configuration for attention-based modeling.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Transformer configuration dictionary
        """
        configs = {
            'fast': {
                'd_model': 128,
                'nhead': 4,
                'num_layers': 2,
                'dim_feedforward': 512,
                'dropout_rate': 0.1,
                'sequence_length': 50,
                'learning_rate': 0.0001,
                'n_epochs': 50,
                'batch_size': 16
            },
            'balanced': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout_rate': 0.1,
                'sequence_length': 100,
                'learning_rate': 0.0001,
                'n_epochs': 150,
                'batch_size': 8,
                'patience': 20
            },
            'accurate': {
                'd_model': 512,
                'nhead': 16,
                'num_layers': 8,
                'dim_feedforward': 2048,
                'dropout_rate': 0.1,
                'sequence_length': 200,
                'learning_rate': 0.00005,
                'n_epochs': 200,
                'batch_size': 4,
                'patience': 25
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_self_attention_config(optimization_level: str = 'balanced') -> Dict[str, Any]:
        """
        Get Self-Attention configuration for multimodal integration.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Self-Attention configuration dictionary
        """
        configs = {
            'fast': {
                'd_model': 128,
                'n_heads': 4,
                'n_layers': 2,
                'd_ff': 512,
                'dropout_rate': 0.1,
                'max_seq_len': 100,
                'learning_rate': 0.001,
                'n_epochs': 50,
                'batch_size': 16
            },
            'balanced': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 6,
                'd_ff': 1024,
                'dropout_rate': 0.1,
                'max_seq_len': 1000,
                'learning_rate': 0.001,
                'n_epochs': 100,
                'batch_size': 8
            },
            'accurate': {
                'd_model': 512,
                'n_heads': 16,
                'n_layers': 8,
                'd_ff': 2048,
                'dropout_rate': 0.1,
                'max_seq_len': 2000,
                'learning_rate': 0.0005,
                'n_epochs': 150,
                'batch_size': 4
            }
        }
        return configs.get(optimization_level, configs['balanced'])
    
    @staticmethod
    def get_ensemble_config(base_models: List[str] = None) -> Dict[str, Any]:
        """
        Get ensemble configuration for combining multiple models.
        
        Args:
            base_models: List of base model names to ensemble
            
        Returns:
            Ensemble configuration dictionary
        """
        if base_models is None:
            base_models = ['svm', 'random_forest', 'mlp']
        
        return {
            'base_models': base_models,
            'voting_type': 'soft',  # 'hard' or 'soft'
            'weights': None,  # Equal weights if None
            'use_stacking': True,
            'meta_learner': 'logistic_regression',
            'cv_folds': 5,
            'use_features_in_secondary': True
        }
    
    @staticmethod
    def get_all_configs(optimization_level: str = 'balanced') -> Dict[str, Dict[str, Any]]:
        """
        Get all model configurations.
        
        Args:
            optimization_level: 'fast', 'balanced', or 'accurate'
            
        Returns:
            Dictionary mapping model names to their configurations
        """
        return {
            'svm': ModelConfigs.get_svm_config(optimization_level),
            'random_forest': ModelConfigs.get_random_forest_config(optimization_level),
            'logistic_regression': ModelConfigs.get_logistic_regression_config(optimization_level),
            'mlp': ModelConfigs.get_mlp_config(optimization_level),
            'cnn': ModelConfigs.get_cnn_config(optimization_level),
            'lstm': ModelConfigs.get_lstm_config(optimization_level),
            'transformer': ModelConfigs.get_transformer_config(optimization_level),
            'self_attention': ModelConfigs.get_self_attention_config(optimization_level),
            'ensemble': ModelConfigs.get_ensemble_config()
        }


# Specialized configurations for different fMRI prediction tasks
class TaskSpecificConfigs:
    """Task-specific model configurations for different aspects of fMRI prediction."""
    
    @staticmethod
    def get_multimodal_fusion_config() -> Dict[str, Any]:
        """Configuration optimized for multimodal feature fusion."""
        return {
            'fusion_method': 'attention',  # 'concat', 'attention', 'gated'
            'modality_weights': {'visual': 0.4, 'audio': 0.3, 'language': 0.3},
            'cross_modal_attention': True,
            'shared_representation_dim': 256,
            'modality_specific_layers': 2
        }
    
    @staticmethod
    def get_temporal_modeling_config() -> Dict[str, Any]:
        """Configuration for modeling temporal dynamics in fMRI."""
        return {
            'window_size': 10,  # TRs
            'temporal_aggregation': 'attention',  # 'mean', 'max', 'attention'
            'use_temporal_convolution': True,
            'temporal_kernel_size': 3,
            'causal_modeling': False
        }
    
    @staticmethod
    def get_brain_region_specific_config() -> Dict[str, Any]:
        """Configuration for brain region-specific modeling."""
        return {
            'use_region_specific_models': True,
            'brain_networks': ['visual', 'auditory', 'language', 'default_mode'],
            'cross_region_connections': True,
            'hierarchical_modeling': True
        }