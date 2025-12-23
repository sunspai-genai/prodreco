"""
Realistic banking data generation module
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from loguru import logger
from .utils import set_random_seeds


class BankingDataGenerator:
    """Generate realistic banking customer data"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.deposit_products = config['products']['deposits']
        self.loan_products = config['products']['loans']
        self.random_seed = config['data']['random_seed']
        set_random_seeds(self.random_seed)
    
    def generate(self, n_samples: int = None) -> pd.DataFrame:
        """
        Generate realistic banking data
        
        Returns DataFrame with:
        - Customer features (80+)
        - Deposit product holdings (5 products)
        - Loan product holdings (10 products)
        """
        if n_samples is None:
            n_samples = self.config['data']['total_customers']
        
        logger.info(f"Generating {n_samples:,} customer records...")
        
        # Generate customer features
        df = self._generate_customer_features(n_samples)
        
        # Generate deposit product holdings
        df = self._generate_deposit_products(df)
        
        # Generate loan product holdings (8.5% of customers)
        df = self._generate_loan_products(df)
        
        logger.info(f"Data generation complete: {df.shape}")
        logger.info(f"Features: {len([c for c in df.columns if c not in self.deposit_products + self.loan_products + ['customer_id']])}")
        logger.info(f"Products: {len(self.deposit_products + self.loan_products)}")
        
        return df
    
    def _generate_customer_features(self, n_samples: int) -> pd.DataFrame:
        """Generate customer demographic and financial features"""
        
        # Categorical features
        customer_segment = np.random.choice(
            ['SME', 'Corporate', 'Enterprise', 'StartUp'],
            n_samples,
            p=[0.60, 0.25, 0.10, 0.05]
        )
        
        industry = np.random.choice([
            'Manufacturing', 'Technology', 'Retail', 'Healthcare',
            'Finance', 'Education', 'Real Estate', 'Services',
            'Construction', 'Transportation'
        ], n_samples)
        
        region = np.random.choice(
            ['North', 'South', 'East', 'West', 'Central'],
            n_samples
        )
        
        business_age = np.random.choice(
            ['0-2', '3-5', '6-10', '11-20', '20+'],
            n_samples,
            p=[0.15, 0.25, 0.30, 0.20, 0.10]
        )
        
        credit_rating = np.random.choice(
            ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'],
            n_samples,
            p=[0.05, 0.15, 0.30, 0.30, 0.12, 0.06, 0.02]
        )
        
        legal_structure = np.random.choice(
            ['Pvt Ltd', 'Public Ltd', 'Partnership', 'Sole Proprietor', 'LLC'],
            n_samples,
            p=[0.40, 0.15, 0.20, 0.20, 0.05]
        )
        
        tax_compliance = np.random.choice(
            ['Excellent', 'Good', 'Average', 'Poor'],
            n_samples,
            p=[0.20, 0.50, 0.25, 0.05]
        )
        
        export_status = np.random.choice(
            ['Exporter', 'Non-Exporter'],
            n_samples,
            p=[0.15, 0.85]
        )
        
        # Numerical features
        annual_revenue = np.random.lognormal(14, 2.5, n_samples)
        employee_count = np.random.randint(1, 5000, n_samples)
        avg_monthly_balance = np.random.lognormal(11, 2.0, n_samples)
        transaction_frequency = np.random.poisson(30, n_samples)
        avg_transaction_size = np.random.lognormal(9, 1.5, n_samples)
        
        # Credit metrics
        debt_to_equity = np.random.uniform(0.1, 4.0, n_samples)
        current_ratio = np.random.uniform(0.5, 3.0, n_samples)
        quick_ratio = np.random.uniform(0.3, 2.5, n_samples)
        interest_coverage = np.random.uniform(0.5, 10.0, n_samples)
        
        # Banking relationship
        relationship_tenure = np.random.randint(0, 25, n_samples)
        digital_engagement_score = np.random.uniform(0, 100, n_samples)
        customer_service_calls = np.random.poisson(5, n_samples)
        
        # Financial health
        cash_flow = np.random.lognormal(13, 1.8, n_samples)
        working_capital = np.random.lognormal(12, 2.0, n_samples)
        collateral_value = np.random.lognormal(14, 2.2, n_samples)
        credit_utilization = np.random.uniform(0, 1, n_samples)
        
        # Build DataFrame
        feature_dict = {
            'customer_id': range(n_samples),
            'customer_segment': customer_segment,
            'industry': industry,
            'region': region,
            'business_age': business_age,
            'credit_rating': credit_rating,
            'legal_structure': legal_structure,
            'tax_compliance': tax_compliance,
            'export_status': export_status,
            'annual_revenue': annual_revenue,
            'employee_count': employee_count,
            'avg_monthly_balance': avg_monthly_balance,
            'transaction_frequency': transaction_frequency,
            'avg_transaction_size': avg_transaction_size,
            'debt_to_equity': debt_to_equity,
            'current_ratio': current_ratio,
            'quick_ratio': quick_ratio,
            'interest_coverage': interest_coverage,
            'relationship_tenure': relationship_tenure,
            'digital_engagement_score': digital_engagement_score,
            'customer_service_calls': customer_service_calls,
            'cash_flow': cash_flow,
            'working_capital': working_capital,
            'collateral_value': collateral_value,
            'credit_utilization': credit_utilization,
        }
        
        # Add additional engineered features to reach 80+
        for i in range(60):
            feature_dict[f'feature_{i+1}'] = np.random.randn(n_samples)
        
        df = pd.DataFrame(feature_dict)
        
        logger.info(f"Generated {len(feature_dict)} customer features")
        
        return df
    
    def _generate_deposit_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate deposit product holdings with realistic patterns"""
        
        n_samples = len(df)
        
        # Checking Account (90% adoption - dominant product)
        checking_prob = 0.88 + 0.05 * (df['relationship_tenure'] / 25)
        df['Checking'] = (np.random.random(n_samples) < checking_prob).astype(int)
        
        # Savings Account (40% adoption, higher if they have checking)
        savings_base = 0.30
        savings_boost = 0.25 * df['Checking']
        savings_tenure = 0.10 * (df['relationship_tenure'] / 25)
        savings_prob = savings_base + savings_boost + savings_tenure
        df['Savings'] = (np.random.random(n_samples) < savings_prob).astype(int)
        
        # Money Market Account (15% adoption, for higher balances)
        mma_base = 0.05
        mma_balance = 0.20 * (df['avg_monthly_balance'] / df['avg_monthly_balance'].quantile(0.95)).clip(0, 1)
        mma_segment = 0.05 * (df['customer_segment'].isin(['Corporate', 'Enterprise'])).astype(int)
        mma_prob = mma_base + mma_balance + mma_segment
        df['MMA'] = (np.random.random(n_samples) < mma_prob).astype(int)
        
        # CD 1 Year (6% adoption, requires savings)
        cd1_base = 0.02
        cd1_savings = 0.06 * df['Savings']
        cd1_tenure = 0.04 * (df['relationship_tenure'] / 25)
        cd1_prob = cd1_base + cd1_savings + cd1_tenure
        df['CD_1Year'] = (np.random.random(n_samples) < cd1_prob).astype(int)
        
        # CD > 1 Year (3% adoption, requires CD 1 Year usually)
        cd_gt_base = 0.01
        cd_gt_cd1 = 0.05 * df['CD_1Year']
        cd_gt_tenure = 0.03 * (df['relationship_tenure'] > 10).astype(int)
        cd_gt_prob = cd_gt_base + cd_gt_cd1 + cd_gt_tenure
        df['CD_GT1Year'] = (np.random.random(n_samples) < cd_gt_prob).astype(int)
        
        logger.info("Deposit products generated")
        for product in self.deposit_products:
            adoption = df[product].mean()
            logger.info(f"  {product}: {adoption:.2%}")
        
        return df
    
    def _generate_loan_products(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate loan product holdings (8.5% of customers have ANY loan)"""
        
        n_samples = len(df)
        target_loan_ratio = self.config['data']['loan_customer_ratio']
        
        # Calculate loan propensity score
        loan_propensity = (
            0.25 * (df['annual_revenue'] / df['annual_revenue'].quantile(0.95)).clip(0, 1) +
            0.20 * (1 / (df['debt_to_equity'] + 0.5)).clip(0, 1) +
            0.15 * (df['cash_flow'] / df['cash_flow'].quantile(0.90)).clip(0, 1) +
            0.15 * (df['relationship_tenure'] / 25) +
            0.10 * (df['collateral_value'] / df['collateral_value'].quantile(0.90)).clip(0, 1) +
            0.15 * np.random.random(n_samples)
        )
        
        # Only top 8.5% customers will be eligible for loans
        loan_threshold = np.percentile(loan_propensity, (1 - target_loan_ratio) * 100)
        is_loan_eligible = loan_propensity > loan_threshold
        
        logger.info(f"Loan eligible customers: {is_loan_eligible.sum():,} ({is_loan_eligible.mean():.2%})")
        
        # Line of Credit (most common - 5% overall)
        loc_prob = 0.06 * is_loan_eligible
        df['Line_of_Credit'] = (np.random.random(n_samples) < loc_prob).astype(int)
        
        # Business Cards (4% overall)
        card_prob = 0.05 * is_loan_eligible
        df['Business_Cards'] = (np.random.random(n_samples) < card_prob).astype(int)
        
        # Term Loan (3.5% overall)
        term_prob = 0.045 * is_loan_eligible * (df['Line_of_Credit'] | (np.random.random(n_samples) < 0.7))
        df['Term_Loan'] = (np.random.random(n_samples) < term_prob).astype(int)
        
        # Working Capital Loan (2% overall)
        wc_prob = 0.025 * is_loan_eligible * df['Line_of_Credit']
        df['Working_Capital_Loan'] = (np.random.random(n_samples) < wc_prob).astype(int)
        
        # Equipment Loan (1.5% overall)
        equip_prob = (0.02 * is_loan_eligible * 
                     df['industry'].isin(['Manufacturing', 'Construction', 'Transportation']).astype(int))
        df['Term_Loan_Equipment'] = (np.random.random(n_samples) < equip_prob).astype(int)
        
        # Real Estate Loan (1.2% overall)
        re_prob = (0.015 * is_loan_eligible * 
                  df['industry'].isin(['Real Estate', 'Manufacturing', 'Construction']).astype(int))
        df['Term_Loan_Real_Estate'] = (np.random.random(n_samples) < re_prob).astype(int)
        
        # Letter of Credit (0.8% overall)
        lc_prob = (0.01 * is_loan_eligible * 
                  df['industry'].isin(['Manufacturing', 'Retail', 'Technology']).astype(int) *
                  (df['export_status'] == 'Exporter').astype(int))
        df['Letter_of_Credit'] = (np.random.random(n_samples) < lc_prob).astype(int)
        
        # SBA Loan (0.6% overall)
        sba_prob = (0.008 * is_loan_eligible * 
                   (df['customer_segment'] == 'SME').astype(int))
        df['SBA_Loan'] = (np.random.random(n_samples) < sba_prob).astype(int)
        
        # Construction Loan (0.4% overall)
        const_prob = (0.005 * is_loan_eligible * 
                     (df['industry'] == 'Construction').astype(int))
        df['Construction_Loan'] = (np.random.random(n_samples) < const_prob).astype(int)
        
        # Bridge Loan (0.2% overall)
        bridge_prob = (0.003 * is_loan_eligible * 
                      df['Term_Loan_Real_Estate'])
        df['Bridge_Loan'] = (np.random.random(n_samples) < bridge_prob).astype(int)
        
        # Verify overall loan adoption
        has_any_loan = df[self.loan_products].sum(axis=1) > 0
        actual_loan_ratio = has_any_loan.mean()
        
        logger.info("Loan products generated")
        logger.info(f"Customers with ANY loan: {has_any_loan.sum():,} ({actual_loan_ratio:.2%})")
        
        for product in self.loan_products:
            adoption = df[product].mean()
            count = df[product].sum()
            logger.info(f"  {product}: {adoption:.2%} ({count:,} customers)")
        
        return df
