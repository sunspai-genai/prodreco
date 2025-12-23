"""
Model explainability module using SHAP
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from tensorflow.keras import Model
import shap
from loguru import logger
from pathlib import Path


class ModelExplainer:
    """Generate model explanations using SHAP"""
    
    def __init__(self, config: Dict, category: str):
        """
        Args:
            config: Configuration dictionary
            category: 'deposit' or 'loan'
        """
        self.config = config
        self.category = category
        self.products = config['products'][f'{category}s']
        self.explainability_config = config.get('explainability', {})
        self.feature_names = None
        self.shap_values = None
        self.explainer = None
    
    def set_feature_names(self, feature_names: List[str]):
        """Set feature names for interpretability"""
        self.feature_names = feature_names
    
    def global_explainability(self,
                             model: Model,
                             X_test: np.ndarray,
                             product_idx: int = 0,
                             save_dir: str = None) -> Tuple:
        """
        Generate global feature importance using SHAP
        
        Args:
            model: Trained model
            X_test: Test features
            product_idx: Index of product to explain (default: 0 = first product)
            save_dir: Directory to save plots
        
        Returns:
            SHAP values and explainer object
        """
        product_name = self.products[product_idx]
        
        logger.info("="*80)
        logger.info(f"GLOBAL EXPLAINABILITY - {self.category.upper()} - {product_name}")
        logger.info("="*80)
        
        sample_size = self.explainability_config.get('shap_sample_size', 200)
        background_size = self.explainability_config.get('background_sample_size', 100)
        
        # Sample data for SHAP (computational efficiency)
        if len(X_test) > sample_size:
            logger.info(f"Sampling {sample_size} examples for SHAP analysis...")
            sample_indices = np.random.choice(
                len(X_test), sample_size, replace=False
            )
            X_sample = X_test[sample_indices]
        else:
            X_sample = X_test
        
        # Background data for KernelExplainer
        if len(X_test) > background_size:
            background_indices = np.random.choice(
                len(X_test), background_size, replace=False
            )
            X_background = X_test[background_indices]
        else:
            X_background = X_test[:background_size]
        
        logger.info(f"Creating SHAP explainer...")
        logger.info(f"  Sample size: {len(X_sample)}")
        logger.info(f"  Background size: {len(X_background)}")
        
        # Create prediction function for this product
        def predict_fn(X):
            preds = model.predict(X, verbose=0)
            return preds[:, product_idx]
        
        # Create SHAP explainer
        self.explainer = shap.KernelExplainer(
            predict_fn,
            X_background
        )
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values (this may take a few minutes)...")
        self.shap_values = self.explainer.shap_values(
            X_sample,
            nsamples=100
        )
        
        logger.info("SHAP values calculated successfully")
        
        # Generate visualizations
        if save_dir:
            self._plot_global_importance(
                X_sample, product_name, save_dir
            )
        
        # Log top features
        self._log_top_features(product_name)
        
        return self.shap_values, self.explainer
    
    def local_explainability(self,
                            model: Model,
                            customer_data: np.ndarray,
                            product_idx: int = 0,
                            customer_id: int = 0,
                            save_dir: str = None) -> Tuple:
        """
        Generate local explanation for a specific customer
        
        Args:
            model: Trained model
            customer_data: Customer features
            product_idx: Index of product to explain
            customer_id: Customer identifier for logging
            save_dir: Directory to save plots
        
        Returns:
            SHAP values and feature contributions
        """
        product_name = self.products[product_idx]
        
        logger.info("="*80)
        logger.info(f"LOCAL EXPLAINABILITY - Customer {customer_id} - {product_name}")
        logger.info("="*80)
        
        # Get prediction
        prediction = model.predict(customer_data, verbose=0)[0, product_idx]
        logger.info(f"Prediction probability: {prediction:.4f}")
        
        # If explainer doesn't exist, create it
        if self.explainer is None:
            logger.info("Creating SHAP explainer for local explanation...")
            
            def predict_fn(X):
                preds = model.predict(X, verbose=0)
                return preds[:, product_idx]
            
            # Use small background
            self.explainer = shap.KernelExplainer(
                predict_fn,
                customer_data[:min(50, len(customer_data))]
            )
        
        # Calculate SHAP values for this customer
        logger.info("Calculating SHAP values for customer...")
        shap_values = self.explainer.shap_values(customer_data[:1], nsamples=100)
        
        # Feature contributions
        feature_contributions = list(zip(
            self.feature_names,
            shap_values[0]
        ))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Log top contributing features
        logger.info(f"\nTop 10 Feature Contributions:")
        for feature, contribution in feature_contributions[:10]:
            direction = "increases" if contribution > 0 else "decreases"
            logger.info(f"  {feature:<30} {contribution:>8.4f} ({direction} probability)")
        
        # Generate visualization
        if save_dir:
            self._plot_local_explanation(
                customer_data, shap_values, product_name,
                customer_id, save_dir
            )
        
        return shap_values, feature_contributions
    
    def _plot_global_importance(self,
                               X_sample: np.ndarray,
                               product_name: str,
                               save_dir: str):
        """Plot global feature importance"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Shorten feature names for visualization
        feature_names_short = [
            f'F{i+1}' if len(name) > 15 else name
            for i, name in enumerate(self.feature_names)
        ]
        
        top_features = self.explainability_config.get('top_features', 20)
        
        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X_sample,
            feature_names=feature_names_short,
            max_display=top_features,
            show=False
        )
        plt.title(f'Global Feature Importance - {self.category.capitalize()} - {product_name}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.category}_{product_name}_global_importance.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Global importance plot saved: {filename}")
        
        # Bar plot of mean absolute SHAP values
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_features:][::-1]
        
        plt.figure(figsize=(10, 8))
        plt.barh(
            range(len(top_indices)),
            mean_abs_shap[top_indices],
            color='steelblue'
        )
        plt.yticks(
            range(len(top_indices)),
            [feature_names_short[i] for i in top_indices]
        )
        plt.xlabel('Mean |SHAP value|')
        plt.title(f'Top {top_features} Important Features - {product_name}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.category}_{product_name}_feature_importance_bar.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Bar plot saved: {filename}")
    
    def _plot_local_explanation(self,
                               customer_data: np.ndarray,
                               shap_values: np.ndarray,
                               product_name: str,
                               customer_id: int,
                               save_dir: str):
        """Plot local explanation for specific customer"""
        
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        feature_names_short = [
            f'F{i+1}' if len(name) > 15 else name
            for i, name in enumerate(self.feature_names)
        ]
        
        # Waterfall plot
        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=self.explainer.expected_value,
                data=customer_data[0],
                feature_names=feature_names_short
            ),
            max_display=15,
            show=False
        )
        plt.title(f'Local Explanation - Customer {customer_id} - {product_name}',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filename = f'{self.category}_{product_name}_customer_{customer_id}_local.png'
        plt.savefig(save_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Local explanation plot saved: {filename}")
    
    def _log_top_features(self, product_name: str):
        """Log top features by importance"""
        
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        top_indices = np.argsort(mean_abs_shap)[-10:][::-1]
        
        logger.info(f"\nTop 10 Most Important Features for {product_name}:")
        for idx in top_indices:
            feature_name = self.feature_names[idx]
            importance = mean_abs_shap[idx]
            logger.info(f"  {feature_name:<30} {importance:.4f}")
