"""
Hyperparameter tuning module
"""
import numpy as np
import itertools
from typing import Dict, List, Tuple, Any
from loguru import logger
from .model_architecture import WideDeepModel
from sklearn.metrics import roc_auc_score


class HyperparameterTuner:
    """Perform hyperparameter tuning"""
    
    def __init__(self, config: Dict, category: str):
        """
        Args:
            config: Configuration dictionary
            category: 'deposit' or 'loan'
        """
        self.config = config
        self.category = category
        self.model_builder = WideDeepModel(config, category)
        self.tuning_config = config.get('hyperparameter_tuning', {})
    
    def tune(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray) -> Tuple[Dict, List[Dict]]:
        """
        Perform grid search hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Best hyperparameters and all trial results
        """
        if not self.tuning_config.get('enabled', False):
            logger.info("Hyperparameter tuning disabled in config")
            return None, []
        
        logger.info("="*80)
        logger.info(f"HYPERPARAMETER TUNING - {self.category.upper()}")
        logger.info("="*80)
        
        param_grid = self.tuning_config['param_grid']
        n_trials = self.tuning_config.get('n_trials', 12)
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_grid)
        
        # Limit number of trials
        if len(param_combinations) > n_trials:
            logger.info(f"Limiting trials from {len(param_combinations)} to {n_trials}")
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:n_trials]
        
        logger.info(f"Total parameter combinations to try: {len(param_combinations)}")
        
        # Track results
        results = []
        best_score = 0
        best_params = None
        
        # Try each combination
        for trial_idx, params in enumerate(param_combinations, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Trial {trial_idx}/{len(param_combinations)}")
            logger.info(f"{'='*80}")
            logger.info(f"Parameters: {params}")
            
            try:
                score = self._evaluate_params(
                    params, X_train, y_train, X_val, y_val
                )
                
                results.append({
                    'trial': trial_idx,
                    'params': params.copy(),
                    'val_auc': score
                })
                
                logger.info(f"Validation AUC: {score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    logger.info(f"✓ New best score: {best_score:.4f}")
            
            except Exception as e:
                logger.error(f"Trial {trial_idx} failed: {e}")
                results.append({
                    'trial': trial_idx,
                    'params': params.copy(),
                    'val_auc': 0.0,
                    'error': str(e)
                })
        
        # Log summary
        logger.info("\n" + "="*80)
        logger.info("HYPERPARAMETER TUNING SUMMARY")
        logger.info("="*80)
        logger.info(f"Best validation AUC: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        # Sort results by score
        results_sorted = sorted(results, key=lambda x: x['val_auc'], reverse=True)
        
        logger.info("\nTop 5 parameter combinations:")
        for i, result in enumerate(results_sorted[:5], 1):
            logger.info(f"{i}. AUC={result['val_auc']:.4f} | {result['params']}")
        
        return best_params, results
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """Generate all combinations of parameters"""
        
        # Get all parameter names and values
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        
        # Generate all combinations
        combinations = []
        for values in itertools.product(*param_values):
            param_dict = dict(zip(param_names, values))
            combinations.append(param_dict)
        
        return combinations
    
    def _evaluate_params(self,
                        params: Dict,
                        X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        epochs: int = 20) -> float:
        """
        Evaluate a single parameter combination
        
        Returns average validation AUC across all products
        """
        training_config = self.config['training'][self.category]
        
        # Build model with these parameters
        model = self.model_builder.build(
            input_dim=X_train.shape[1],
            wide_dim=params.get('wide_dim'),
            deep_dims=params.get('deep_dims'),
            dropout_rate=params.get('dropout_rate'),
            l2_reg=self.config['model']['l2_regularization']
        )
        
        # Compile model
        use_weighted_loss = self.category == 'loan'
        learning_rate = params.get('learning_rate')
        
        model = self.model_builder.compile_model(
            model,
            learning_rate=learning_rate,
            use_weighted_loss=use_weighted_loss,
            pos_weight=training_config.get('positive_class_weight', 10.0)
        )
        
        # Train for limited epochs
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=training_config['batch_size'],
            verbose=0
        )
        
        # Get best validation AUC
        best_val_auc = max(history.history['val_auc'])
        
        return best_val_auc
