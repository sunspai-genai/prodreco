"""
Main execution script for Hybrid Product Recommendation System
"""
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import HybridRecommendationPipeline
from src.utils import set_random_seeds


def setup_logging():
    """Configure logging"""
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # File handler
    Path("logs").mkdir(exist_ok=True)
    logger.add(
        "logs/pipeline_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
        level="DEBUG"
    )


def main():
    """Main execution function"""
    
    # Setup logging
    setup_logging()
    
    logger.info("="*80)
    logger.info("HYBRID WIDE & DEEP PRODUCT RECOMMENDATION SYSTEM")
    logger.info("For Business Banking - Deposits and Loans")
    logger.info("="*80)
    
    try:
        # Set random seeds for reproducibility
        set_random_seeds(42)
        
        # Initialize pipeline
        pipeline = HybridRecommendationPipeline(config_path="config.yaml")
        
        # Run complete pipeline
        results = pipeline.run(generate_data=True)
        
        logger.success("\n" + "="*80)
        logger.success("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.success("="*80)
        
        logger.info("\nGenerated outputs:")
        logger.info("  Models:")
        logger.info("    - models/best_deposit_model.keras")
        logger.info("    - models/best_loan_model.keras")
        logger.info("    - models/deposit_preprocessor.pkl")
        logger.info("    - models/loan_preprocessor.pkl")
        
        logger.info("\n  Data:")
        logger.info("    - data/banking_data.csv")
        
        logger.info("\n  Results:")
        logger.info("    - results/deposit_metrics.csv")
        logger.info("    - results/loan_metrics.csv")
        logger.info("    - results/recommendations_all_products.csv")
        logger.info("    - results/recommendations_deposits.csv")
        logger.info("    - results/recommendations_loans.csv")
        logger.info("    - results/probability_matrix.csv")
        logger.info("    - results/deposit_optimal_thresholds.json")
        logger.info("    - results/loan_optimal_thresholds.json")
        logger.info("    - results/pipeline_summary.json")
        
        logger.info("\n  Explainability Plots:")
        logger.info("    - plots/deposit_*_global_importance.png")
        logger.info("    - plots/deposit_*_customer_*_local.png")
        logger.info("    - plots/loan_*_global_importance.png")
        logger.info("    - plots/loan_*_customer_*_local.png")
        
        logger.info("\n  Logs:")
        logger.info("    - logs/pipeline_*.log")
        
        return results
    
    except Exception as e:
        logger.exception(f"Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
