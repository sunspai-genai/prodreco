"""
Setup validation script
Run this to verify your environment is configured correctly
"""
import sys
from pathlib import Path


def check_python_version():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} - FAIL")
        print("  Required: Python 3.8 or higher")
        return False


def check_dependencies():
    """Check required packages"""
    print("\nChecking dependencies...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tensorflow': 'tensorflow',
        'keras': 'keras',
        'sklearn': 'scikit-learn',
        'imblearn': 'imbalanced-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'shap': 'shap',
        'yaml': 'pyyaml',
        'loguru': 'loguru',
        'joblib': 'joblib',
    }
    
    all_ok = True
    
    for module_name, package_name in required_packages.items():
        try:
            if module_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            elif module_name == 'imblearn':
                import imblearn
                version = imblearn.__version__
            elif module_name == 'yaml':
                import yaml
                version = yaml.__version__
            else:
                module = __import__(module_name)
                version = module.__version__
            
            print(f"✓ {package_name:<20} {version:<15} - OK")
        except ImportError:
            print(f"✗ {package_name:<20} {'not found':<15} - FAIL")
            all_ok = False
        except AttributeError:
            print(f"✓ {package_name:<20} {'installed':<15} - OK")
    
    return all_ok


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            print(f"✓ GPU detected: {len(gpus)} device(s)")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            print("  Note: Training will use GPU (faster)")
        else:
            print("ℹ No GPU detected")
            print("  Note: Training will use CPU (slower but functional)")
        
        return True
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_directories():
    """Check required directories"""
    print("\nChecking directory structure...")
    
    required_dirs = [
        'src',
        'data',
        'models',
        'results',
        'plots',
        'logs'
    ]
    
    all_ok = True
    
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"✓ {dir_name}/ - OK")
        else:
            print(f"ℹ {dir_name}/ - Creating...")
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ {dir_name}/ - Created")
    
    return all_ok


def check_config():
    """Check configuration file"""
    print("\nChecking configuration file...")
    
    config_path = Path('config.yaml')
    
    if not config_path.exists():
        print("✗ config.yaml not found")
        return False
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required sections
        required_sections = ['data', 'products', 'model', 'training', 'paths']
        missing = []
        
        for section in required_sections:
            if section not in config:
                missing.append(section)
        
        if missing:
            print(f"✗ config.yaml missing sections: {', '.join(missing)}")
            return False
        
        print("✓ config.yaml - OK")
        print(f"  - Total customers: {config['data']['total_customers']:,}")
        print(f"  - Deposit products: {len(config['products']['deposits'])}")
        print(f"  - Loan products: {len(config['products']['loans'])}")
        
        return True
    
    except Exception as e:
        print(f"✗ Error reading config.yaml: {e}")
        return False


def check_imports():
    """Check if src module can be imported"""
    print("\nChecking src module...")
    
    try:
        sys.path.insert(0, str(Path.cwd()))
        
        from src.pipeline import HybridRecommendationPipeline
        from src.data_generator import BankingDataGenerator
        from src.preprocessor import DataPreprocessor
        
        print("✓ All src modules imported successfully")
        return True
    
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def estimate_requirements():
    """Estimate resource requirements"""
    print("\n" + "="*60)
    print("RESOURCE REQUIREMENTS")
    print("="*60)
    
    try:
        import yaml
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        n_customers = config['data']['total_customers']
        
        # Estimate memory
        features = 80
        bytes_per_float = 4
        
        # Data: n_customers * features * bytes_per_float * 3 (train/val/test copies)
        data_mb = (n_customers * features * bytes_per_float * 3) / (1024 * 1024)
        
        # Models: ~200MB each
        model_mb = 400
        
        total_mb = data_mb + model_mb
        total_gb = total_mb / 1024
        
        print(f"\nEstimated Requirements:")
        print(f"  Customers: {n_customers:,}")
        print(f"  Memory: ~{total_gb:.1f} GB RAM")
        print(f"  Disk Space: ~{total_gb*2:.1f} GB (with outputs)")
        
        # Estimate time
        tuning_enabled = config.get('hyperparameter_tuning', {}).get('enabled', False)
        
        if tuning_enabled:
            print(f"  Training Time: ~2-3 hours (with tuning)")
        else:
            print(f"  Training Time: ~45-60 minutes (without tuning)")
        
        print(f"\nRecommendations:")
        if total_gb > 16:
            print("  ⚠ Consider reducing total_customers in config.yaml")
            print(f"    Suggested: {min(n_customers, 100000):,} for 16GB RAM")
        else:
            print("  ✓ Memory requirements look good")
        
    except Exception as e:
        print(f"Could not estimate requirements: {e}")


def main():
    """Run all checks"""
    print("="*60)
    print("SETUP VALIDATION")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("GPU", check_gpu),
        ("Directories", check_directories),
        ("Configuration", check_config),
        ("Module Imports", check_imports),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} check failed with error: {e}")
            results.append((name, False))
    
    # Resource estimation
    estimate_requirements()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name:<20} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL CHECKS PASSED")
        print("="*60)
        print("\nYou're ready to run the pipeline!")
        print("\nNext steps:")
        print("  1. Review config.yaml settings")
        print("  2. Run: python main.py")
        print("  3. Monitor progress in console and logs/")
        return 0
    else:
        print("\n" + "="*60)
        print("✗ SOME CHECKS FAILED")
        print("="*60)
        print("\nPlease fix the issues above before running the pipeline.")
        print("\nCommon solutions:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Check Python version: python --version")
        print("  - Verify config.yaml exists and is valid")
        return 1


if __name__ == "__main__":
    sys.exit(main())
