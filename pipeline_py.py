"""
Main pipeline orchestrator
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from loguru import logger

from .utils import (
    load_config, setup_directories, save_model,
    save_preprocessor, save_json, MetricsTracker
)
from .data_generator import BankingDataGenerator
from .preprocessor import DataPreprocessor
from .trainer import ModelTrainer
from .evaluator import ModelEvaluator
from .hyperparameter_tuner import HyperparameterTuner
from .recommender import ProductRecommender
from .explainer import ModelExplainer


class HybridRecommendationPipeline:
    """End-to-end pipeline for hybrid product recommendation"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = load_config(config_path)
        setup_directories(self.config)
        
        self.deposit_model = None
        self.loan_model = None
        self.deposit_preprocessor = None
        self.loan_preprocessor = None
        self.metrics_tracker = MetricsTracker()
        
        logger.info("="*80)
        logger.info("HYBRID PRODUCT RECOMMENDATION SYSTEM")
        logger.info("="*80)
    
    def run(self, data_path: str = None, generate_data: bool = True):
        """
        Run complete pipeline
        
        Args:
            data_path: Path to existing data CSV (optional)
            generate_data: Whether to generate synthetic data
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING PIPELINE EXECUTION")
        logger.info("="*80)
        
        # Step 1: Load or generate data
        if data_path:
            df = self._load_data(data_path)
        elif generate_data:
            df = self._generate_data()
        else:
            raise ValueError("Must provide data_path or set generate_data=True")
        
        # Step 2: Train deposit model
        logger.info("\n" + "#"*80)
        logger.info("DEPOSIT MODEL PIPELINE")
        logger.info("#"*80)
        
        deposit_results = self._train_category_model(df, 'deposit')
        
        # Step 3: Train loan model
        logger.info("\n" + "#"*80)
        logger.info("LOAN MODEL PIPELINE")
        logger.info("#"*80)
        
        loan_results = self._train_category_model(df, 'loan')
        
        # Step 4: Generate recommendations
        logger.info("\n" + "#"*80)
        logger.info("GENERATING RECOMMENDATIONS")
        logger.info("#"*80)
        
        recommendations = self._generate_recommendations(df)
        
        # Step 5: Model explainability
        logger.info("\n" + "#"*80)
        logger.info("MODEL EXPLAINABILITY")
        logger.info("#"*80)
        
        self._generate_explanations(
            df,
            deposit_results,
            loan_results
        )
        
        # Step 6: Save results
        self._save_pipeline_results(recommendations)
        
        logger.info("\n" + "="*80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        
        return {
            'deposit_results': deposit_results,
            'loan_results': loan_results,
            'recommendations': recommendations
        }
    
    def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from CSV"""
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded: {df.shape}")
        return df
    
    def _generate_data(self) -> pd.DataFrame:
        """Generate synthetic banking data"""
        logger.info("\n" + "="*80)
        logger.info("STEP 1: DATA GENERATION")
        logger.info("="*80)
        
        generator = BankingDataGenerator(self.config)
        df = generator.generate(n_samples=self.config['data']['total_customers'])
        
        # Save generated data
        data_path = Path(self.config['paths']['data']) / 'banking_data.csv'
        df.to_csv(data_path, index=False)
        logger.info(f"Data saved to {data_path}")
        
        return df
    
    def _train_category_model(self, df: pd.DataFrame, category: str) -> Dict:
        """
        Train model for a specific category (deposit or loan)
        
        Returns dictionary with all results
        """
        logger.info(f"\nTraining {category} model...")
        
        # Initialize components
        preprocessor = DataPreprocessor(self.config, category)
        trainer = ModelTrainer(self.config, category)
        evaluator = ModelEvaluator(self.config, category)
        tuner = HyperparameterTuner(self.config, category)
        
        # Step 1: Prepare data
        df_prepared = preprocessor.identify_features(df)
        
        if category == 'deposit':
            strategy = self.config['training']['deposit']['strategy']
            df_category = preprocessor.prepare_deposit_data(df_prepared, strategy)
        else:
            strategy = self.config['training']['loan']['strategy']
            df_category, resampling_strategy = preprocessor.prepare_loan_data(
                df_prepared, strategy
            )
        
        # Step 2: Preprocess
        X = preprocessor.encode_and_scale(df_category, is_training=True)
        y = preprocessor.extract_targets(df_category)
        
        # Step 3: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            X, y,
            test_size=self.config['data']['test_size'],
            val_size=self.config['data']['val_size'],
            random_state=self.config['data']['random_seed']
        )
        
        # Step 4: Apply SMOTE if needed (loan model only)
        if category == 'loan' and resampling_strategy == 'smote':
            sampling_ratio = self.config['training']['loan']['smote_sampling_ratio']
            X_train, y_train = preprocessor.apply_smote(
                X_train, y_train, sampling_ratio
            )
        
        # Step 5: Hyperparameter tuning
        best_params = None
        tuning_results = []
        
        if self.config['hyperparameter_tuning']['enabled']:
            best_params, tuning_results = tuner.tune(
                X_train, y_train, X_val, y_val
            )
            
            save_json(
                tuning_results,
                Path(self.config['paths']['results']) / f'{category}_tuning_results.json'
            )
        
        # Step 6: Train final model
        model_path = str(Path(self.config['paths']['models']) / f'best_{category}_model.keras')
        
        model, history = trainer.train(
            X_train, y_train,
            X_val, y_val,
            model_path,
            hyperparams=best_params
        )
        
        # Step 7: Evaluate
        eval_results = evaluator.evaluate(model, X_test, y_test)
        
        # Step 8: Find optimal thresholds
        optimal_thresholds = evaluator.find_optimal_thresholds(
            model, X_val, y_val
        )
        
        # Save model and preprocessor
        save_model(model, model_path)
        save_preprocessor(
            preprocessor,
            Path(self.config['paths']['models']) / f'{category}_preprocessor.pkl'
        )
        
        # Save evaluation metrics
        eval_results['per_product'].to_csv(
            Path(self.config['paths']['results']) / f'{category}_metrics.csv',
            index=False
        )
        
        # Save optimal thresholds
        save_json(
            optimal_thresholds,
            Path(self.config['paths']['results']) / f'{category}_optimal_thresholds.json'
        )
        
        # Store in class attributes
        if category == 'deposit':
            self.deposit_model = model
            self.deposit_preprocessor = preprocessor
        else:
            self.loan_model = model
            self.loan_preprocessor = preprocessor
        
        return {
            'model': model,
            'preprocessor': preprocessor,
            'history': history,
            'eval_results': eval_results,
            'optimal_thresholds': optimal_thresholds,
            'test_data': (X_test, y_test),
            'best_params': best_params
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Generate product recommendations"""
        
        recommender = ProductRecommender(self.config)
        
        # Sample customers for recommendations
        test_customers = df.sample(
            n=min(1000, len(df)),
            random_state=self.config['data']['random_seed']
        )
        
        logger.info(f"Generating recommendations for {len(test_customers):,} customers...")
        
        # All products ranking
        all_products_recs = recommender.rank_all_products(
            test_customers,
            self.deposit_model,
            self.loan_model,
            self.deposit_preprocessor,
            self.loan_preprocessor,
            top_n=self.config['recommendations']['top_n']
        )
        
        # Deposit-specific rankings
        deposit_recs = recommender.rank_by_category(
            test_customers,
            self.deposit_model,
            self.deposit_preprocessor,
            'deposit',
            top_n=3
        )
        
        # Loan-specific rankings
        loan_recs = recommender.rank_by_category(
            test_customers,
            self.loan_model,
            self.loan_preprocessor,
            'loan',
            top_n=3
        )
        
        # Probability matrix
        prob_matrix = recommender.get_product_probabilities(
            test_customers,
            self.deposit_model,
            self.loan_model,
            self.deposit_preprocessor,
            self.loan_preprocessor
        )
        
        # Save recommendations
        results_path = Path(self.config['paths']['results'])
        
        all_products_recs.to_csv(
            results_path / 'recommendations_all_products.csv',
            index=False
        )
        deposit_recs.to_csv(
            results_path / 'recommendations_deposits.csv',
            index=False
        )
        loan_recs.to_csv(
            results_path / 'recommendations_loans.csv',
            index=False
        )
        prob_matrix.to_csv(
            results_path / 'probability_matrix.csv',
            index=False
        )
        
        logger.info("Recommendations saved successfully")
        
        return {
            'all_products': all_products_recs,
            'deposits': deposit_recs,
            'loans': loan_recs,
            'probabilities': prob_matrix
        }
    
    def _generate_explanations(self,
                              df: pd.DataFrame,
                              deposit_results: Dict,
                              loan_results: Dict):
        """Generate model explanations"""
        
        plots_path = Path(self.config['paths']['plots'])
        
        # Deposit model explanations
        logger.info("\nGenerating deposit model explanations...")
        deposit_explainer = ModelExplainer(self.config, 'deposit')
        deposit_explainer.set_feature_names(
            self.deposit_preprocessor.feature_names
        )
        
        X_test_deposit, y_test_deposit = deposit_results['test_data']
        
        # Global explanation for first deposit product
        deposit_explainer.global_explainability(
            self.deposit_model,
            X_test_deposit,
            product_idx=0,
            save_dir=str(plots_path)
        )
        
        # Local explanation for a sample customer
        sample_customer = df.sample(n=1, random_state=42)
        X_customer = self.deposit_preprocessor.encode_and_scale(
            sample_customer, is_training=False
        )
        
        deposit_explainer.local_explainability(
            self.deposit_model,
            X_customer,
            product_idx=0,
            customer_id=int(sample_customer['customer_id'].values[0]),
            save_dir=str(plots_path)
        )
        
        # Loan model explanations
        logger.info("\nGenerating loan model explanations...")
        loan_explainer = ModelExplainer(self.config, 'loan')
        loan_explainer.set_feature_names(
            self.loan_preprocessor.feature_names
        )
        
        X_test_loan, y_test_loan = loan_results['test_data']
        
        # Global explanation for first loan product
        loan_explainer.global_explainability(
            self.loan_model,
            X_test_loan,
            product_idx=0,
            save_dir=str(plots_path)
        )
        
        # Local explanation
        X_customer_loan = self.loan_preprocessor.encode_and_scale(
            sample_customer, is_training=False
        )
        
        loan_explainer.local_explainability(
            self.loan_model,
            X_customer_loan,
            product_idx=0,
            customer_id=int(sample_customer['customer_id'].values[0]),
            save_dir=str(plots_path)
        )
        
        logger.info("Explainability analysis complete")
    
    def _save_pipeline_results(self, recommendations: Dict):
        """Save final pipeline results summary"""
        
        summary = {
            'config': self.config,
            'deposit_model_path': str(Path(self.config['paths']['models']) / 'best_deposit_model.keras'),
            'loan_model_path': str(Path(self.config['paths']['models']) / 'best_loan_model.keras'),
            'total_recommendations': len(recommendations['all_products']),
            'unique_customers': recommendations['all_products']['customer_id'].nunique()
        }
        
        save_json(
            summary,
            Path(self.config['paths']['results']) / 'pipeline_summary.json'
        )
        
        logger.info("\nPipeline summary saved")
