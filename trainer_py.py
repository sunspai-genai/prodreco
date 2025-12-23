"""
Model training module
"""
import numpy as np
from typing import Dict, Tuple, Any
from tensorflow.keras import Model
from loguru import logger
from .model_architecture import WideDeepModel
from .preprocessor import DataPreprocessor


class ModelTrainer:
    """Handle model training operations"""
    
    def __init__(self, config: Dict, category: str):
        """
        Args:
            config: Configuration dictionary
            category: 'deposit' or 'loan'
        """
        self.config = config
        self.category = category
        self.model_builder = WideDeepModel(config, category)
        self.model = None
        self.history = None
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: np.ndarray,
             y_val: np.ndarray,
             model_path: str,
             hyperparams: Dict = None) -> Tuple[Model, Dict]:
        """
        Train Wide & Deep model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_path: Path to save best model
            hyperparams: Optional hyperparameters override
        
        Returns:
            Trained model and training history
        """
        logger.info("="*80)
        logger.info(f"TRAINING {self.category.upper()} MODEL")
        logger.info("="*80)
        
        training_config = self.config['training'][self.category]
        
        # Log data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Log target distribution
        self._log_target_distribution(y_train, "Training")
        self._log_target_distribution(y_val, "Validation")
        
        # Build model
        if hyperparams:
            logger.info("Using custom hyperparameters")
            self.model = self.model_builder.build(
                input_dim=X_train.shape[1],
                **hyperparams
            )
        else:
            logger.info("Using default hyperparameters from config")
            self.model = self.model_builder.build(input_dim=X_train.shape[1])
        
        # Compile model
        use_weighted_loss = self.category == 'loan'
        pos_weight = training_config.get('positive_class_weight', 10.0)
        
        self.model = self.model_builder.compile_model(
            self.model,
            use_weighted_loss=use_weighted_loss,
            pos_weight=pos_weight
        )
        
        # Print model summary
        logger.info("\nModel Summary:")
        self.model.summary(print_fn=logger.info)
        
        # Get callbacks
        callbacks = self.model_builder.get_callbacks(model_path)
        
        # Training parameters
        epochs = training_config['epochs']
        batch_size = training_config['batch_size']
        
        logger.info(f"\nTraining parameters:")
        logger.info(f"  Epochs: {epochs}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Batches per epoch: {len(X_train) // batch_size}")
        
        # Train model
        logger.info("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history.history
        
        logger.info("\nTraining completed!")
        logger.info(f"Best val_auc: {max(self.history['val_auc']):.4f}")
        logger.info(f"Final val_loss: {self.history['val_loss'][-1]:.4f}")
        
        return self.model, self.history
    
    def _log_target_distribution(self, y: np.ndarray, dataset_name: str):
        """Log target variable distribution"""
        products = self.config['products'][f"{self.category}s"]
        
        logger.info(f"\n{dataset_name} set target distribution:")
        for idx, product in enumerate(products):
            pos_count = int(y[:, idx].sum())
            pos_ratio = y[:, idx].mean()
            logger.info(f"  {product:<25} {pos_count:>6} ({pos_ratio:>6.2%})")
