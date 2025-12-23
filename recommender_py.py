"""
Product recommendation generation module
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from tensorflow.keras import Model
from loguru import logger


class ProductRecommender:
    """Generate product recommendations for customers"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.deposit_products = config['products']['deposits']
        self.loan_products = config['products']['loans']
        self.all_products = self.deposit_products + self.loan_products
        self.top_n = config['recommendations']['top_n']
        self.min_score = config['recommendations']['min_score']
    
    def rank_all_products(self,
                         customer_df: pd.DataFrame,
                         deposit_model: Model,
                         loan_model: Model,
                         deposit_preprocessor: Any,
                         loan_preprocessor: Any,
                         top_n: int = None) -> pd.DataFrame:
        """
        Rank all products for each customer
        
        Args:
            customer_df: DataFrame with customer data
            deposit_model: Trained deposit model
            loan_model: Trained loan model
            deposit_preprocessor: Deposit data preprocessor
            loan_preprocessor: Loan data preprocessor
            top_n: Number of top products to return
        
        Returns:
            DataFrame with rankings
        """
        top_n = top_n or self.top_n
        
        logger.info(f"Generating top {top_n} product recommendations...")
        logger.info(f"Customers: {len(customer_df):,}")
        
        # Get deposit predictions
        X_deposit = deposit_preprocessor.encode_and_scale(
            customer_df, is_training=False
        )
        deposit_probs = deposit_model.predict(X_deposit, verbose=0)
        
        # Get loan predictions
        X_loan = loan_preprocessor.encode_and_scale(
            customer_df, is_training=False
        )
        loan_probs = loan_model.predict(X_loan, verbose=0)
        
        # Generate recommendations
        recommendations = []
        
        for idx, customer_id in enumerate(customer_df['customer_id'].values):
            # Combine all product probabilities
            all_probs = {}
            
            for prod_idx, product in enumerate(self.deposit_products):
                all_probs[product] = float(deposit_probs[idx, prod_idx])
            
            for prod_idx, product in enumerate(self.loan_products):
                all_probs[product] = float(loan_probs[idx, prod_idx])
            
            # Sort by probability
            sorted_products = sorted(
                all_probs.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Filter by minimum score
            sorted_products = [
                (prod, score) for prod, score in sorted_products
                if score >= self.min_score
            ]
            
            # Top N products
            for rank, (product, score) in enumerate(sorted_products[:top_n], 1):
                category = ('Deposit' if product in self.deposit_products 
                          else 'Loan')
                
                recommendations.append({
                    'customer_id': customer_id,
                    'rank': rank,
                    'product': product,
                    'category': category,
                    'score': score,
                    'recommendation': 'High' if score > 0.7 else 
                                    'Medium' if score > 0.5 else 'Low'
                })
        
        recommendations_df = pd.DataFrame(recommendations)
        
        logger.info(f"Generated {len(recommendations_df):,} recommendations")
        
        return recommendations_df
    
    def rank_by_category(self,
                        customer_df: pd.DataFrame,
                        model: Model,
                        preprocessor: Any,
                        category: str,
                        top_n: int = 3) -> pd.DataFrame:
        """
        Rank products within a specific category
        
        Args:
            customer_df: DataFrame with customer data
            model: Trained model for this category
            preprocessor: Data preprocessor for this category
            category: 'deposit' or 'loan'
            top_n: Number of top products
        
        Returns:
            DataFrame with category-specific rankings
        """
        logger.info(f"Generating top {top_n} {category} recommendations...")
        
        products = (self.deposit_products if category == 'deposit' 
                   else self.loan_products)
        
        # Get predictions
        X = preprocessor.encode_and_scale(customer_df, is_training=False)
        probs = model.predict(X, verbose=0)
        
        recommendations = []
        
        for idx, customer_id in enumerate(customer_df['customer_id'].values):
            # Get probabilities for this customer
            customer_probs = [(products[i], float(probs[idx, i])) 
                            for i in range(len(products))]
            
            # Sort by probability
            customer_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Top N
            for rank, (product, score) in enumerate(customer_probs[:top_n], 1):
                recommendations.append({
                    'customer_id': customer_id,
                    'category': category.capitalize(),
                    'rank': rank,
                    'product': product,
                    'score': score
                })
        
        return pd.DataFrame(recommendations)
    
    def get_product_probabilities(self,
                                 customer_df: pd.DataFrame,
                                 deposit_model: Model,
                                 loan_model: Model,
                                 deposit_preprocessor: Any,
                                 loan_preprocessor: Any) -> pd.DataFrame:
        """
        Get probability matrix for all products and customers
        
        Returns:
            DataFrame with customer_id and probability for each product
        """
        logger.info("Generating probability matrix...")
        
        # Get deposit predictions
        X_deposit = deposit_preprocessor.encode_and_scale(
            customer_df, is_training=False
        )
        deposit_probs = deposit_model.predict(X_deposit, verbose=0)
        
        # Get loan predictions
        X_loan = loan_preprocessor.encode_and_scale(
            customer_df, is_training=False
        )
        loan_probs = loan_model.predict(X_loan, verbose=0)
        
        # Build DataFrame
        result = {'customer_id': customer_df['customer_id'].values}
        
        for idx, product in enumerate(self.deposit_products):
            result[f'{product}_prob'] = deposit_probs[:, idx]
        
        for idx, product in enumerate(self.loan_products):
            result[f'{product}_prob'] = loan_probs[:, idx]
        
        prob_df = pd.DataFrame(result)
        
        logger.info(f"Probability matrix generated: {prob_df.shape}")
        
        return prob_df
    
    def generate_campaign_lists(self,
                               customer_df: pd.DataFrame,
                               product: str,
                               model: Model,
                               preprocessor: Any,
                               min_probability: float = 0.7) -> pd.DataFrame:
        """
        Generate targeted campaign list for a specific product
        
        Args:
            customer_df: DataFrame with customer data
            product: Product name
            model: Trained model (deposit or loan)
            preprocessor: Data preprocessor
            min_probability: Minimum probability threshold
        
        Returns:
            DataFrame with target customers
        """
        logger.info(f"Generating campaign list for {product}...")
        logger.info(f"Minimum probability: {min_probability}")
        
        # Determine product index
        if product in self.deposit_products:
            product_idx = self.deposit_products.index(product)
        elif product in self.loan_products:
            product_idx = self.loan_products.index(product)
        else:
            raise ValueError(f"Unknown product: {product}")
        
        # Get predictions
        X = preprocessor.encode_and_scale(customer_df, is_training=False)
        probs = model.predict(X, verbose=0)
        
        # Extract probabilities for this product
        product_probs = probs[:, product_idx]
        
        # Filter customers
        target_mask = product_probs >= min_probability
        
        campaign_df = pd.DataFrame({
            'customer_id': customer_df['customer_id'].values[target_mask],
            'product': product,
            'probability': product_probs[target_mask]
        })
        
        # Sort by probability
        campaign_df = campaign_df.sort_values('probability', ascending=False)
        campaign_df['rank'] = range(1, len(campaign_df) + 1)
        
        logger.info(f"Campaign list generated: {len(campaign_df):,} customers")
        
        return campaign_df
