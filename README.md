project/
├── config.yaml                  # Central configuration
├── requirements.txt             # Python dependencies
├── main.py                      # Main execution script
├── setup_check.py              # Environment validation
├── README.md                    # Project documentation
├── USAGE.md                     # Detailed usage guide
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── pipeline.py             # Main orchestrator
│   ├── data_generator.py       # Realistic data generation
│   ├── preprocessor.py         # Data preprocessing & imbalance handling
│   ├── model_architecture.py   # Wide & Deep neural network
│   ├── trainer.py              # Model training logic
│   ├── evaluator.py            # Comprehensive evaluation
│   ├── hyperparameter_tuner.py # Grid search tuning
│   ├── recommender.py          # Product ranking & recommendations
│   ├── explainer.py            # SHAP explainability
│   └── utils.py                # Utility functions
│
└── [Auto-created directories]
    ├── data/                   # Generated/input data
    ├── models/                 # Trained models
    ├── results/                # Metrics & recommendations
    ├── plots/                  # Explainability charts
    └── logs/                   # Execution logs

    ✓ Stratified Train/Test Split: Maintains class distribution across splits
✓ Evaluation Metrics: AUC-ROC, Precision, Recall, F1, Hamming Loss, Jaccard Score
✓ Hyperparameter Tuning: Grid search with configurable trials
✓ Best Model Selection: Based on validation AUC
✓ Top-N Product Ranking: For each customer with confidence scores
✓ Global Explainability: SHAP feature importance plots
✓ Local Explainability: Customer-specific SHAP waterfall plots
✓ Hybrid Approach: Separate models for deposits (5) and loans (10)
✓ Imbalance Handling: SMOTE, weighted loss, undersampling, stratification
✓ Production Grade: Error handling, logging, modular design, type hints

🎯 Key Features
Imbalance Management:

SMOTE oversampling for loan model (8.5% positive class)
Weighted binary crossentropy (10x penalty for false negatives)
Stratified sampling throughout
Configurable strategies per category

Model Architecture:

Wide component for memorization
Deep component for generalization
Batch normalization and dropout
L2 regularization
Separate models for deposit vs loan products

Comprehensive Evaluation:

Per-product metrics (AUC, Precision, Recall, F1)
Overall multi-label metrics
Optimal threshold finding
Confusion matrix analysis

Explainability:

SHAP global importance (top features)
SHAP local explanations (per customer)
Automated visualization generation

🚀 How to Use

Setup:

bashpip install -r requirements.txt
python setup_check.py  # Validate installation

Configure (optional):
Edit config.yaml to adjust dataset size, training parameters, etc.
Run:

bashpython main.py

Review Results:


Models: models/best_deposit_model.keras, models/best_loan_model.keras
Metrics: results/deposit_metrics.csv, results/loan_metrics.csv
Recommendations: results/recommendations_*.csv
Explanations: plots/*.png

📊 Expected Performance
Deposit Model:

Checking: AUC 0.85-0.90
Other deposits: AUC 0.70-0.82

Loan Model (with imbalance handling):

Common loans (5-6% adoption): AUC 0.75-0.82
Rare loans (1-2% adoption): AUC 0.65-0.75

🔧 Customization

Dataset size: Adjust total_customers in config.yaml
Training speed: Disable hyperparameter_tuning or reduce epochs
Model capacity: Modify wide_dim and deep_dims
Imbalance strategy: Change strategy to smote, undersample, or weighted

✨ Production Features

Logging: Comprehensive logging to console and files
Error Handling: Graceful failure with informative messages
Modularity: Each component is independent and testable
Validation: Data validation and environment checking
Documentation: Extensive README and USAGE guides
Type Hints: For better IDE support and maintainability
