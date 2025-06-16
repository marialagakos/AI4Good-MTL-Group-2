# models/neural/mlp.py
"""
Multi-Layer Perceptron (MLP) classifier for multimodal fMRI stimulus prediction.

This module implements a flexible MLP architecture with configurable layers,
dropout, batch normalization, and advanced training features.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
from pathlib import Path

from models.base_classifier import BaseClassifier

class MLPClassifier(BaseClassifier):
    """Multi-Layer Perceptron classifier using PyTorch"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_model(self, input_dim: int, n_classes: int) -> nn.Module:
        """Create MLP model"""
        hidden_dims = self.config.get('hidden_dims', [512, 256, 128])
        dropout_rate = self.config.get('dropout_rate', 0.3)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, n_classes))
        
        return nn.Sequential(*layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier':
        """Train MLP classifier"""
        # Preprocess data
        X_scaled = self.scaler.fit_transform(X)
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Create model
        input_dim = X_scaled.shape[1]
        n_classes = len(np.unique(y_encoded))
        self.model = self._create_model(input_dim, n_classes).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 0.001),
            weight_decay=self.config.get('weight_decay', 1e-4)
        )
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=True
        )
        
        # Training loop
        n_epochs = self.config.get('n_epochs', 100)
        self.model.train()
        
        for epoch in range(n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 20 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return prediction probabilities"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
            
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            
        return probabilities
    
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance (gradient-based)"""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
            
        # Use gradient-based feature importance
        dummy_input = torch.randn(1, self.scaler.n_features_in_).to(self.device)
        dummy_input.requires_grad_(True)
        
        self.model.eval()
        output = self.model(dummy_input)
        output.backward(torch.ones_like(output))
        
        importance = torch.abs(dummy_input.grad).mean(dim=0).cpu().numpy()
        return importance

class MLPNetwork(nn.Module):
    """
    Multi-Layer Perceptron network with configurable architecture.
    
    This class implements the core MLP architecture with support for:
    - Variable number of hidden layers
    - Batch normalization
    - Dropout regularization
    - Residual connections (optional)
    - Multiple activation functions
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 n_classes: int,
                 dropout_rate: float = 0.3,
                 activation: str = 'relu',
                 use_batch_norm: bool = True,
                 use_residual: bool = False):
        """
        Initialize MLP network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            n_classes: Number of output classes
            dropout_rate: Dropout probability
            activation: Activation function ('relu', 'gelu', 'swish')
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(MLPNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_classes = n_classes
        self.use_residual = use_residual
        
        # Activation function selection
        self.activation_fn = self._get_activation_function(activation)
        
        # Build network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Batch normalization
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            # Dropout
            self.dropouts.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, n_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation_function(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU()
        }
        
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        
        return activations[activation]
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                # He initialization for ReLU-like activations
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)
        
        # Initialize output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, n_classes)
        """
        residual = None
        
        for i, layer in enumerate(self.layers):
            # Store residual connection input
            if self.use_residual and i > 0 and x.shape[1] == layer.out_features:
                residual = x
            
            # Linear transformation
            x = layer(x)
            
            # Batch normalization
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            
            # Activation
            x = self.activation_fn(x)
            
            # Residual connection
            if self.use_residual and residual is not None and x.shape == residual.shape:
                x = x + residual
                residual = None
            
            # Dropout
            x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x


class MLPClassifier(BaseClassifier):
    """
    Multi-Layer Perceptron classifier for fMRI stimulus prediction.
    
    This classifier provides a flexible MLP architecture with advanced training
    features including learning rate scheduling, early stopping, and gradient clipping.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MLP classifier.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__(config)
        
        # Data preprocessing
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model: Optional[MLPNetwork] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
        
        # Training history
        self.training_history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        # Configuration validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        required_params = ['hidden_dims']
        for param in required_params:
            if param not in self.config:
                raise ValueError(f"Missing required parameter: {param}")
        
        # Set default values
        defaults = {
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'n_epochs': 100,
            'batch_size': 32,
            'activation': 'relu',
            'use_batch_norm': True,
            'use_residual': False,
            'early_stopping_patience': 10,
            'grad_clip_norm': 1.0,
            'scheduler_type': 'plateau',
            'scheduler_patience': 5,
            'scheduler_factor': 0.5,
            'validation_split': 0.2,
            'random_state': 42
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _create_model(self, input_dim: int, n_classes: int) -> MLPNetwork:
        """
        Create MLP model architecture.
        
        Args:
            input_dim: Number of input features
            n_classes: Number of output classes
            
        Returns:
            MLPNetwork instance
        """
        return MLPNetwork(
            input_dim=input_dim,
            hidden_dims=self.config['hidden_dims'],
            n_classes=n_classes,
            dropout_rate=self.config['dropout_rate'],
            activation=self.config['activation'],
            use_batch_norm=self.config['use_batch_norm'],
            use_residual=self.config['use_residual']
        )
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        optimizer_type = self.config.get('optimizer', 'adam')
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay'],
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config['scheduler_type']
        
        if scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config['scheduler_factor'],
                patience=self.config['scheduler_patience'],
                verbose=True
            )
        elif scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['n_epochs']
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.get('step_size', 30),
                gamma=self.config['scheduler_factor']
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def _prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare data for training.
        
        Args:
            X: Input features
            y: Target labels
            
        Returns:
            Tuple of (X_tensor, y_tensor)
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        return X_tensor, y_tensor
    
    def _create_data_loaders(self, 
                           X_tensor: torch.Tensor, 
                           y_tensor: torch.Tensor) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Create training and validation data loaders.
        
        Args:
            X_tensor: Input tensor
            y_tensor: Target tensor
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create validation split if specified
        val_split = self.config['validation_split']
        if val_split > 0:
            n_samples = X_tensor.shape[0]
            n_val = int(n_samples * val_split)
            
            # Random split
            indices = torch.randperm(n_samples)
            train_indices = indices[n_val:]
            val_indices = indices[:n_val]
            
            # Create datasets
            train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
            val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['batch_size'],
                shuffle=False,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            return train_loader, val_loader
        else:
            # No validation split
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(
                dataset,
                batch_size=self.config['batch_size'],
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            return train_loader, None
    
    def _train_epoch(self, train_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['grad_clip_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip_norm'])
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MLPClassifier':
        """
        Train the MLP classifier.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            y: Target labels of shape (n_samples,)
            
        Returns:
            Self for method chaining
        """
        print(f"Training MLP classifier on {X.shape[0]} samples with {X.shape[1]} features")
        start_time = time.time()
        
        # Prepare data
        X_tensor, y_tensor = self._prepare_data(X, y)
        
        # Create model
        input_dim = X_tensor.shape[1]
        n_classes = len(torch.unique(y_tensor))
        self.model = self._create_model(input_dim, n_classes).to(self.device)
        
        print(f"Model architecture: {input_dim} -> {' -> '.join(map(str, self.config['hidden_dims']))} -> {n_classes}")
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Create data loaders
        train_loader, val_loader = self._create_data_loaders(X_tensor, y_tensor)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        n_epochs = self.config['n_epochs']
        print(f"Training for {n_epochs} epochs...")
        
        for epoch in range(n_epochs):
            # Train
            train_loss, train_acc = self._train_epoch(train_loader, criterion)
            self.training_history['loss'].append(train_loss)
            self.training_history['accuracy'].append(train_acc)
            
            # Validate
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader, criterion)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_accuracy'].append(val_acc)
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                # Print progress
                if epoch % 20 == 0 or epoch == n_epochs - 1:
                    print(f"Epoch {epoch+1}/{n_epochs}: "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Early stopping check
                if patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                # No validation, just print training progress
                if epoch % 20 == 0 or epoch == n_epochs - 1:
                    print(f"Epoch {epoch+1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
                
                # Learning rate scheduling
                if self.scheduler is not None and not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
        
        # Restore best model if early stopping was used
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Restored best model from early stopping")
        
        self.is_fitted = True
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            # Process in batches to handle large datasets
            batch_size = self.config['batch_size']
            for i in range(0, X_tensor.shape[0], batch_size):
                batch_X = X_tensor[i:i + batch_size]
                outputs = self.model(batch_X)
                batch_predictions = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        # Convert back to original labels
        predictions = np.array(predictions)
        return self.label_encoder.inverse_transform(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return prediction probabilities.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Prediction probabilities of shape (n_samples, n_classes)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before making predictions")
        
        # Prepare data
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Make predictions
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            # Process in batches
            batch_size = self.config['batch_size']
            for i in range(0, X_tensor.shape[0], batch_size):
                batch_X = X_tensor[i:i + batch_size]
                outputs = self.model(batch_X)
                batch_probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probabilities.append(batch_probs)
        
        return np.vstack(probabilities)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Return feature importance scores using gradient-based method.
        
        Returns:
            Feature importance scores of shape (n_features,)
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before getting feature importance")
        
        # Create dummy input with gradients enabled
        n_features = self.scaler.n_features_in_
        dummy_input = torch.randn(1, n_features, requires_grad=True).to(self.device)
        
        self.model.eval()
        
        # Forward pass
        output = self.model(dummy_input)
        
        # Compute gradients for each class
        importance_scores = []
        
        for class_idx in range(output.shape[1]):
            # Zero gradients
            if dummy_input.grad is not None:
                dummy_input.grad.zero_()
            
            # Backward pass for specific class
            output[0, class_idx].backward(retain_graph=True)
            
            # Collect gradient magnitudes
            if dummy_input.grad is not None:
                class_importance = torch.abs(dummy_input.grad).cpu().numpy().flatten()
                importance_scores.append(class_importance)
        
        # Average importance across all classes
        if importance_scores:
            importance = np.mean(importance_scores, axis=0)
        else:
            importance = np.zeros(n_features)
        
        return importance
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before saving")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'training_history': self.training_history
        }
        
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        self.scaler = checkpoint['scaler']
        self.label_encoder = checkpoint['label_encoder']
        self.training_history = checkpoint['training_history']
        
        # Recreate model architecture
        input_dim = self.scaler.n_features_in_
        n_classes = len(self.label_encoder.classes_)
        self.model = self._create_model(input_dim, n_classes).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.is_fitted = True
        print(f"Model loaded from {filepath}")