"""
Data preprocessing module
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from loguru import logger
from .utils import handle_class_imbalance_info


class DataPreprocessor:
    """Handle all data preprocessing operations"""
    
    def __init__(self, config: Dict, category: str):
        """
        Args:
            config: Configuration dictionary
            category: 'deposit' or 'loan'
        """
        self.config = config
        self.category = category
        self.scaler = None
        self.label_encoders = {}
        self.feature_names = None
        self.categorical_features = None
        self.numerical_features = None
        
        if category == 'deposit':
            self.products = config['products']['deposits']
        elif category == 'loan':
            self.products = config['products']['loans']
        else:
            raise ValueError(f"Invalid category: {category}")
    
    def identify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and categorize features"""
        all_products = (self.config['products']['deposits'] + 
                       self.config['products']['loans'])
        
        exclude_cols = ['customer_id', 'has_any_loan'] + all_products
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        self.categorical_features = (df[feature_cols]
                                    .select_dtypes(include=['object'])
                                    .columns.tolist())
        self.numerical_features = (df[feature_cols]
                                  .select_dtypes(include=[np.number])
                                  .columns.tolist())
        self.feature_names = feature_cols
        
        logger.info(f"{self.category.upper()} - Feature identification:")
        logger.info(f"  Total features: {len(feature_cols)}")
        logger.info(f"  Categorical: {len(self.categorical_features)}")
        logger.info(f"  Numerical: {len(self.numerical_features)}")
        
        return df
    
    def encode_and_scale(self, df: pd.DataFrame, 
                        is_training: bool = True) -> np.ndarray:
        """Encode categorical and scale numerical features"""
        X = df[self.feature_names].copy()
        
        # Encode categorical features
        for col in self.categorical_features:
            if is_training:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                logger.debug(f"Encoded {col}: {len(le.classes_)} classes")
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    # Handle unseen categories
                    X[col] = X[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                else:
                    logger.warning(f"No encoder found for {col}, filling with -1")
                    X[col] = -1
        
        # Scale numerical features
        if is_training:
            self.scaler = StandardScaler()
            X[self.numerical_features] = self.scaler.fit_transform(
                X[self.numerical_features]
            )
            logger.info(f"Fitted scaler on {len(self.numerical_features)} numerical features")
        else:
            if self.scaler is not None:
                X[self.numerical_features] = self.scaler.transform(
                    X[self.numerical_features]
                )
            else:
                logger.warning("No scaler found, skipping scaling")
        
        return X.values.astype(np.float32)
    
    def extract_targets(self, df: pd.DataFrame) -> np.ndarray:
        """Extract target variables"""
        y = df[self.products].values.astype(np.float32)
        
        # Log class distribution
        handle_class_imbalance_info(y, self.products)
        
        return y
    
    def prepare_deposit_data(self, df: pd.DataFrame, 
                           strategy: str = 'balanced') -> pd.DataFrame:
        """
        Prepare deposit training data
        
        Args:
            df: Full dataset
            strategy: 'all' or 'balanced'
        """
        logger.info("="*80)
        logger.info("PREPARING DEPOSIT TRAINING DATA")
        logger.info("="*80)
        logger.info(f"Strategy: {strategy}")
        
        if strategy == 'all':
            logger.info("Using all customers for deposit model")
            return df.copy()
        
        elif strategy == 'balanced':
            logger.info("Balancing checking account dominance")
            
            # Identify customers with multiple deposit products
            other_deposits = [p for p in self.products if p != 'Checking']
            df['has_other_deposits'] = df[other_deposits].sum(axis=1) > 0
            
            multi_product = df[df['has_other_deposits'] == True]
            checking_only = df[df['has_other_deposits'] == False]
            
            logger.info(f"Multi-product customers: {len(multi_product):,}")
            logger.info(f"Checking-only customers: {len(checking_only):,}")
            
            # Keep all multi-product + sample of checking-only
            sample_size = min(len(checking_only), len(multi_product) * 2)
            checking_sample = checking_only.sample(n=sample_size, random_state=42)
            
            deposit_df = pd.concat([multi_product, checking_sample], 
                                  ignore_index=True)
            
            logger.info(f"Final deposit training set: {len(deposit_df):,}")
            
            return deposit_df.drop('has_other_deposits', axis=1)
        
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
    
    def prepare_loan_data(self, df: pd.DataFrame, 
                         strategy: str = 'smote') -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Prepare loan training data with imbalance handling
        
        Args:
            df: Full dataset
            strategy: 'smote', 'undersample', 'smoteenn', or 'weighted'
        
        Returns:
            Prepared DataFrame and resampling strategy to apply during training
        """
        logger.info("="*80)
        logger.info("PREPARING LOAN TRAINING DATA")
        logger.info("="*80)
        logger.info(f"Strategy: {strategy}")
        
        # Identify loan holders
        df['has_any_loan'] = df[self.products].sum(axis=1) > 0
        
        loan_customers = df[df['has_any_loan'] == True]
        non_loan_customers = df[df['has_any_loan'] == False]
        
        total = len(df)
        loan_count = len(loan_customers)
        
        logger.info(f"Total customers: {total:,}")
        logger.info(f"Loan customers: {loan_count:,} ({loan_count/total:.2%})")
        logger.info(f"Non-loan customers: {len(non_loan_customers):,} "
                   f"({len(non_loan_customers)/total:.2%})")
        
        if strategy == 'undersample':
            # Aggressive undersampling
            ratio = self.config['training']['loan']['undersampling_ratio']
            target_non_loan = int(loan_count / ratio)
            
            if len(non_loan_customers) > target_non_loan:
                non_loan_sample = non_loan_customers.sample(
                    n=target_non_loan, random_state=42
                )
            else:
                non_loan_sample = non_loan_customers
            
            loan_df = pd.concat([loan_customers, non_loan_sample], 
                               ignore_index=True)
            
            logger.info(f"After undersampling: {len(loan_df):,}")
            logger.info(f"  Loan customers: {len(loan_customers):,} "
                       f"({len(loan_customers)/len(loan_df):.2%})")
            
            return loan_df, None
        
        elif strategy == 'smote':
            # Keep all loan + moderate non-loan sample
            # Will apply SMOTE during training
            sample_size = min(len(non_loan_customers), loan_count * 3)
            non_loan_sample = non_loan_customers.sample(
                n=sample_size, random_state=42
            )
            
            loan_df = pd.concat([loan_customers, non_loan_sample], 
                               ignore_index=True)
            
            logger.info(f"After selective sampling: {len(loan_df):,}")
            logger.info("Will apply SMOTE during training")
            
            return loan_df, 'smote'
        
        elif strategy == 'smoteenn':
            # Similar to SMOTE but will use SMOTEENN
            sample_size = min(len(non_loan_customers), loan_count * 3)
            non_loan_sample = non_loan_customers.sample(
                n=sample_size, random_state=42
            )
            
            loan_df = pd.concat([loan_customers, non_loan_sample], 
                               ignore_index=True)
            
            logger.info(f"After selective sampling: {len(loan_df):,}")
            logger.info("Will apply SMOTEENN during training")
            
            return loan_df, 'smoteenn'
        
        elif strategy == 'weighted':
            # Use all data with class weights
            logger.info("Using all data with weighted loss")
            return df, 'weighted'
        
        else:
            raise ValueError(f"Invalid strategy: {strategy}")
    
    def split_data(self, X: np.ndarray, y: np.ndarray,
                  test_size: float = 0.15,
                  val_size: float = 0.15,
                  random_state: int = 42) -> Tuple:
        """
        Stratified train-validation-test split
        
        For multi-label, stratify on "has any positive label"
        """
        logger.info("Performing stratified train-validation-test split...")
        
        # Create stratification key (any positive label)
        has_positive = (y.sum(axis=1) > 0).astype(int)
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=has_positive
        )
        
        # Second split: separate validation from train
        has_positive_temp = (y_temp.sum(axis=1) > 0).astype(int)
        val_size_adjusted = val_size / (1 - test_size)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=has_positive_temp
        )
        
        logger.info(f"Split complete:")
        logger.info(f"  Train: {X_train.shape}")
        logger.info(f"  Val:   {X_val.shape}")
        logger.info(f"  Test:  {X_test.shape}")
        
        # Log distribution
        train_pos_ratio = (y_train.sum(axis=1) > 0).mean()
        val_pos_ratio = (y_val.sum(axis=1) > 0).mean()
        test_pos_ratio = (y_test.sum(axis=1) > 0).mean()
        
        logger.info(f"Positive label distribution:")
        logger.info(f"  Train: {train_pos_ratio:.2%}")
        logger.info(f"  Val:   {val_pos_ratio:.2%}")
        logger.info(f"  Test:  {test_pos_ratio:.2%}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def apply_smote(self, X_train: np.ndarray, y_train: np.ndarray,
                   sampling_strategy: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE oversampling to training data
        
        Args:
            X_train: Training features
            y_train: Training targets (multi-label)
            sampling_strategy: Target ratio of minority/majority class
        """
        logger.info("Applying SMOTE oversampling...")
        
        # Create binary target for SMOTE (any loan)
        has_loan = (y_train.sum(axis=1) > 0).astype(int)
        
        before_counts = {
            'negative': (has_loan == 0).sum(),
            'positive': (has_loan == 1).sum()
        }
        
        logger.info(f"Before SMOTE: Negative={before_counts['negative']:,}, "
                   f"Positive={before_counts['positive']:,}")
        
        # For multi-label, we duplicate rows for minority class
        minority_indices = np.where(has_loan == 1)[0]
        majority_indices = np.where(has_loan == 0)[0]
        
        target_minority_count = int(len(majority_indices) * sampling_strategy)
        n_synthetic_needed = max(0, target_minority_count - len(minority_indices))
        
        if n_synthetic_needed > 0:
            # Randomly duplicate minority samples
            synthetic_indices = np.random.choice(
                minority_indices,
                size=n_synthetic_needed,
                replace=True
            )
            
            # Combine original and synthetic
            all_indices = np.concatenate([
                np.arange(len(X_train)),
                synthetic_indices
            ])
            
            X_resampled = X_train[all_indices]
            y_resampled = y_train[all_indices]
            
            after_has_loan = (y_resampled.sum(axis=1) > 0).astype(int)
            after_counts = {
                'negative': (after_has_loan == 0).sum(),
                'positive': (after_has_loan == 1).sum()
            }
            
            logger.info(f"After SMOTE: Negative={after_counts['negative']:,}, "
                       f"Positive={after_counts['positive']:,}")
            logger.info(f"Added {n_synthetic_needed:,} synthetic samples")
            logger.info(f"Final training set: {X_resampled.shape}")
            
            return X_resampled, y_resampled
        else:
            logger.info("No SMOTE needed, sufficient minority samples")
            return X_train, y_train
