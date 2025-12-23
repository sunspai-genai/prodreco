"""
Model evaluation module
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from tensorflow.keras import Model
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_fscore_support,
    average_precision_score,
    hamming_loss,
    jaccard_score,
    classification_report
)
from loguru import logger


class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, config: Dict, category: str):
        """
        Args:
            config: Configuration dictionary
            category: 'deposit' or 'loan'
        """
        self.config = config
        self.category = category
        self.products = config['products'][f'{category}s']
        self.threshold = config['evaluation']['threshold']
    
    def evaluate(self,
                model: Model,
                X_test: np.ndarray,
                y_test: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary containing all metrics
        """
        logger.info("="*80)
        logger.info(f"{self.category.upper()} MODEL EVALUATION")
        logger.info("="*80)
        
        # Get predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Overall multi-label metrics
        overall_metrics = self._calculate_overall_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Per-product metrics
        per_product_metrics = self._calculate_per_product_metrics(
            y_test, y_pred, y_pred_proba
        )
        
        # Log results
        self._log_evaluation_results(overall_metrics, per_product_metrics)
        
        return {
            'overall': overall_metrics,
            'per_product': per_product_metrics,
            'predictions': {
                'y_test': y_test,
                'y_pred_proba': y_pred_proba,
                'y_pred': y_pred
            }
        }
    
    def _calculate_overall_metrics(self,
                                   y_test: np.ndarray,
                                   y_pred: np.ndarray,
                                   y_pred_proba: np.ndarray) -> Dict:
        """Calculate overall multi-label metrics"""
        
        metrics = {
            'hamming_loss': hamming_loss(y_test, y_pred),
            'jaccard_score_samples': jaccard_score(y_test, y_pred, average='samples'),
            'jaccard_score_macro': jaccard_score(y_test, y_pred, average='macro'),
            'subset_accuracy': np.mean(np.all(y_test == y_pred, axis=1)),
        }
        
        # Average metrics across all products
        try:
            # Micro average (all samples and products together)
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                y_test.ravel(), y_pred.ravel(), average='binary', zero_division=0
            )
            
            metrics.update({
                'precision_micro': precision_micro,
                'recall_micro': recall_micro,
                'f1_micro': f1_micro
            })
        except:
            logger.warning("Could not calculate micro-averaged metrics")
        
        return metrics
    
    def _calculate_per_product_metrics(self,
                                      y_test: np.ndarray,
                                      y_pred: np.ndarray,
                                      y_pred_proba: np.ndarray) -> pd.DataFrame:
        """Calculate metrics for each product"""
        
        metrics_list = []
        
        for idx, product in enumerate(self.products):
            y_true_product = y_test[:, idx]
            y_pred_product = y_pred[:, idx]
            y_proba_product = y_pred_proba[:, idx]
            
            # Skip if no positive samples in test set
            if y_true_product.sum() == 0:
                logger.warning(f"{product}: No positive samples in test set")
                continue
            
            # Calculate metrics
            try:
                auc = roc_auc_score(y_true_product, y_proba_product)
            except:
                auc = 0.0
                logger.warning(f"{product}: Could not calculate AUC")
            
            try:
                avg_precision = average_precision_score(y_true_product, y_proba_product)
            except:
                avg_precision = 0.0
            
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_product, y_pred_product,
                average='binary',
                zero_division=0
            )
            
            # Confusion matrix elements
            tp = np.sum((y_true_product == 1) & (y_pred_product == 1))
            fp = np.sum((y_true_product == 0) & (y_pred_product == 1))
            tn = np.sum((y_true_product == 0) & (y_pred_product == 0))
            fn = np.sum((y_true_product == 1) & (y_pred_product == 0))
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            metrics_list.append({
                'product': product,
                'auc_roc': auc,
                'avg_precision': avg_precision,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'specificity': specificity,
                'support': int(y_true_product.sum()),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            })
        
        return pd.DataFrame(metrics_list)
    
    def _log_evaluation_results(self,
                               overall_metrics: Dict,
                               per_product_df: pd.DataFrame):
        """Log evaluation results"""
        
        logger.info("\n" + "-"*80)
        logger.info("OVERALL METRICS")
        logger.info("-"*80)
        
        for metric, value in overall_metrics.items():
            logger.info(f"{metric:<30} {value:.4f}")
        
        logger.info("\n" + "-"*80)
        logger.info("PER-PRODUCT METRICS")
        logger.info("-"*80)
        
        # Print table header
        logger.info(f"{'Product':<25} {'AUC':>8} {'Precision':>10} {'Recall':>8} "
                   f"{'F1':>8} {'Support':>8}")
        logger.info("-"*80)
        
        # Print each product's metrics
        for _, row in per_product_df.iterrows():
            logger.info(
                f"{row['product']:<25} "
                f"{row['auc_roc']:>8.4f} "
                f"{row['precision']:>10.4f} "
                f"{row['recall']:>8.4f} "
                f"{row['f1_score']:>8.4f} "
                f"{row['support']:>8}"
            )
        
        # Summary statistics
        logger.info("\n" + "-"*80)
        logger.info("SUMMARY STATISTICS")
        logger.info("-"*80)
        logger.info(f"Average AUC:       {per_product_df['auc_roc'].mean():.4f}")
        logger.info(f"Average Precision: {per_product_df['precision'].mean():.4f}")
        logger.info(f"Average Recall:    {per_product_df['recall'].mean():.4f}")
        logger.info(f"Average F1:        {per_product_df['f1_score'].mean():.4f}")
    
    def find_optimal_thresholds(self,
                               model: Model,
                               X_val: np.ndarray,
                               y_val: np.ndarray) -> Dict[str, float]:
        """
        Find optimal decision threshold for each product
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation targets
        
        Returns:
            Dictionary mapping product names to optimal thresholds
        """
        logger.info("Finding optimal thresholds...")
        
        y_pred_proba = model.predict(X_val, verbose=0)
        optimal_thresholds = {}
        
        for idx, product in enumerate(self.products):
            y_true = y_val[:, idx]
            y_proba = y_pred_proba[:, idx]
            
            if y_true.sum() == 0:
                optimal_thresholds[product] = 0.5
                continue
            
            # Try different thresholds
            thresholds = np.arange(0.1, 0.9, 0.05)
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in thresholds:
                y_pred = (y_proba >= threshold).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(
                    y_true, y_pred,
                    average='binary',
                    zero_division=0
                )
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds[product] = best_threshold
            logger.info(f"{product:<25} Optimal threshold: {best_threshold:.2f} "
                       f"(F1={best_f1:.4f})")
        
        return optimal_thresholds
