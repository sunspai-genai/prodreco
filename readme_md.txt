# Hybrid Wide & Deep Product Recommendation System

Production-grade recommendation system for business banking products using hybrid Wide & Deep learning architecture.

## Overview

This system predicts customer propensity for 15 banking products (5 deposit + 10 loan products) using separate specialized models for each category. It handles severe class imbalance (8.5% loan adoption) with advanced techniques including SMOTE, weighted loss functions, and stratified sampling.

## Features

✅ **Hybrid Architecture**: Separate Wide & Deep models for deposits and loans
✅ **Class Imbalance Handling**: SMOTE, weighted loss, undersampling
✅ **Stratified Splitting**: Maintains class distribution across train/val/test
✅ **Hyperparameter Tuning**: Grid search with cross-validation
✅ **Comprehensive Evaluation**: AUC-ROC, Precision, Recall, F1, and more
✅ **Product Ranking**: Top-N recommendations per customer
✅ **Model Explainability**: SHAP-based global and local explanations
✅ **Production Ready**: Modular code, logging, error handling

## Project Structure

```
.
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── main.py                     # Main execution script
├── README.md                   # This file
├── USAGE.md                    # Detailed usage guide
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── pipeline.py             # Main pipeline orchestrator
│   ├── data_generator.py       # Synthetic data generation
│   ├── preprocessor.py         # Data preprocessing
│   ├── model_architecture.py   # Wide & Deep architecture
│   ├── trainer.py              # Model training
│   ├── evaluator.py            # Model evaluation
│   ├── hyperparameter_tuner.py # Hyperparameter tuning
│   ├── recommender.py          # Recommendation generation
│   ├── explainer.py            # SHAP explainability
│   └── utils.py                # Utility functions
│
├── data/                       # Data directory
│   └── banking_data.csv        # Generated/input data
│
├── models/                     # Trained models
│   ├── best_deposit_model.keras
│   ├── best_loan_model.keras
│   ├── deposit_preprocessor.pkl
│   └── loan_preprocessor.pkl
│
├── results/                    # Results and metrics
│   ├── deposit_metrics.csv
│   ├── loan_metrics.csv
│   ├── recommendations_*.csv
│   ├── probability_matrix.csv
│   └── *_optimal_thresholds.json
│
├── plots/                      # Explainability plots
│   ├── deposit_*_global_importance.png
│   ├── deposit_*_customer_*_local.png
│   ├── loan_*_global_importance.png
│   └── loan_*_customer_*_local.png
│
└── logs/                       # Execution logs
    └── pipeline_*.log
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or download the repository**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

## Quick Start

### Basic Usage

Run the complete pipeline with default settings:

```bash
python main.py
```

This will:
1. Generate 355,000 synthetic customer records
2. Train deposit model (5 products)
3. Train loan model (10 products) with imbalance handling
4. Perform hyperparameter tuning
5. Evaluate models on test set
6. Generate product recommendations
7. Create SHAP explainability plots

### Expected Runtime

- Data Generation: ~2 minutes
- Deposit Model Training: ~15-20 minutes
- Loan Model Training: ~25-30 minutes (includes SMOTE)
- Hyperparameter Tuning: ~60-90 minutes (if enabled)
- Total: ~2-3 hours (with tuning), ~1 hour (without tuning)

## Configuration

Edit `config.yaml` to customize:

### Data Configuration
```yaml
data:
  total_customers: 355000
  loan_customer_ratio: 0.085  # 8.5% have loans
  test_size: 0.15
  val_size: 0.15
```

### Model Architecture
```yaml
model:
  wide_dim: 64
  deep_dims: [256, 128, 64]
  dropout_rate: 0.4
  l2_regularization: 0.001
```

### Training Strategy
```yaml
training:
  deposit:
    strategy: balanced  # 'all' or 'balanced'
    epochs: 50
    batch_size: 256
  
  loan:
    strategy: smote  # 'smote', 'undersample', 'weighted'
    epochs: 60
    batch_size: 128
    positive_class_weight: 10.0
```

### Hyperparameter Tuning
```yaml
hyperparameter_tuning:
  enabled: true  # Set to false to skip tuning
  n_trials: 12
```

## Products

### Deposit Products (5)
- Checking Account
- Savings Account
- Money Market Account (MMA)
- Certificate of Deposit - 1 Year
- Certificate of Deposit - >1 Year

### Loan Products (10)
- Term Loan
- Letter of Credit
- Line of Credit
- Term Loan Equipment
- Term Loan Real Estate
- Business Cards
- Working Capital Loan
- SBA Loan
- Construction Loan
- Bridge Loan

## Key Features Explained

### 1. Class Imbalance Handling

**Problem**: Only 8.5% of customers have loans (severe imbalance)

**Solutions**:
- **SMOTE**: Synthetic oversampling of minority class (loan customers)
- **Weighted Loss**: 10x penalty for misclassifying positive class
- **Undersampling**: Reduce majority class (deposit-only customers)
- **Stratified Splitting**: Maintain class distribution in all splits

### 2. Hybrid Architecture

**Why separate models?**
- Deposit and loan customers have different profiles
- Different feature importance
- Different imbalance ratios
- Better business alignment (separate teams)

**Wide Component**: Memorizes feature interactions
**Deep Component**: Generalizes patterns

### 3. Evaluation Metrics

**Multi-label Metrics**:
- Hamming Loss
- Jaccard Score
- Subset Accuracy

**Per-Product Metrics**:
- AUC-ROC (primary metric)
- Precision, Recall, F1
- Average Precision
- Specificity

### 4. Explainability

**Global**: Which features are most important overall?
- SHAP summary plots
- Feature importance rankings

**Local**: Why did customer X get recommendation Y?
- SHAP waterfall plots
- Feature contribution analysis

## Output Files

### Models
- `best_deposit_model.keras`: Trained deposit model
- `best_loan_model.keras`: Trained loan model
- `*_preprocessor.pkl`: Preprocessing pipelines

### Metrics
- `deposit_metrics.csv`: Per-product performance metrics
- `loan_metrics.csv`: Per-product performance metrics

### Recommendations
- `recommendations_all_products.csv`: Top-5 products per customer
- `recommendations_deposits.csv`: Top-3 deposit products
- `recommendations_loans.csv`: Top-3 loan products
- `probability_matrix.csv`: Probability for all products

### Configuration
- `*_optimal_thresholds.json`: Best decision thresholds per product
- `pipeline_summary.json`: Execution summary

## Performance Expectations

### Deposit Model
- **Checking**: AUC 0.85-0.90 (dominant product)
- **Savings**: AUC 0.75-0.82
- **MMA**: AUC 0.72-0.78
- **CDs**: AUC 0.70-0.76

### Loan Model (with imbalance handling)
- **Line of Credit**: AUC 0.75-0.82
- **Business Cards**: AUC 0.72-0.78
- **Term Loans**: AUC 0.70-0.76
- **Specialty Loans**: AUC 0.65-0.72 (rare products)

## Troubleshooting

### Memory Issues
Reduce `total_customers` in config.yaml:
```yaml
data:
  total_customers: 50000  # Instead of 355000
```

### Training Too Slow
Disable hyperparameter tuning:
```yaml
hyperparameter_tuning:
  enabled: false
```

### GPU Not Detected
TensorFlow should auto-detect GPU. Verify:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Customization

### Use Your Own Data

1. Prepare CSV with required columns (see `data_generator.py`)
2. Modify `main.py`:
```python
results = pipeline.run(
    data_path="path/to/your/data.csv",
    generate_data=False
)
```

### Add New Products

1. Update `config.yaml`:
```yaml
products:
  deposits:
    - Checking
    - YourNewProduct
```

2. Ensure your data has corresponding binary columns

### Modify Architecture

Edit `config.yaml`:
```yaml
model:
  wide_dim: 128  # Increase for more capacity
  deep_dims: [512, 256, 128, 64]  # Deeper network
```

## API Usage

Use trained models in production:

```python
from src.pipeline import HybridRecommendationPipeline
from src.recommender import ProductRecommender
import pandas as pd

# Load pipeline
pipeline = HybridRecommendationPipeline("config.yaml")

# Load your customer data
customers = pd.read_csv("new_customers.csv")

# Generate recommendations
recommender = ProductRecommender(pipeline.config)
recommendations = recommender.rank_all_products(
    customers,
    pipeline.deposit_model,
    pipeline.loan_model,
    pipeline.deposit_preprocessor,
    pipeline.loan_preprocessor,
    top_n=5
)

print(recommendations)
```

## Citation

If you use this system in your research or production, please cite:

```
Hybrid Wide & Deep Product Recommendation System
Business Banking Application
Version 1.0.0
```

## License

Proprietary - Internal Use Only

## Support

For issues or questions:
1. Check USAGE.md for detailed documentation
2. Review logs in `logs/` directory
3. Contact: [your-email@bank.com]

## Changelog

### Version 1.0.0 (2024-12-22)
- Initial release
- Hybrid deposit/loan architecture
- SMOTE imbalance handling
- SHAP explainability
- Hyperparameter tuning
- Comprehensive evaluation metrics
