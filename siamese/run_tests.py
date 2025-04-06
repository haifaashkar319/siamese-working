from model_testing import SiameseModelTester
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from siamese import (
    SiameseNetwork, 
    create_base_network, 
    create_head_model,
    CosineSimilarity,  # Import from our siamese module instead
    contrastive_loss  # Updated loss function
)

def load_test_data_and_model():
    """Load the saved model and test data."""
    try:
        # Load the model with custom objects
        custom_objects = {
            'CosineSimilarity': CosineSimilarity,
            'contrastive_loss': contrastive_loss,  # Updated loss function
            'SiameseNetwork': SiameseNetwork
        }
        model = tf.keras.models.load_model(
            "models/siamese_model.keras",
            custom_objects=custom_objects
        )
        print("Model loaded successfully")
        
        # Load test pairs from CSV
        test_pairs_df = pd.read_csv('test_pairs.csv')
        print(f"Loaded {len(test_pairs_df)} test pairs")
        
        # Load feature vectors
        features_df = pd.read_csv('features.csv', index_col='session')
        
        # Prepare test data
        X1_test = []
        X2_test = []
        Y_test = []
        test_pairs = []
        
        for _, row in test_pairs_df.iterrows():
            vec1 = features_df.loc[row['session1']].values
            vec2 = features_df.loc[row['session2']].values
            X1_test.append(vec1)
            X2_test.append(vec2)
            Y_test.append(row['label'])
            test_pairs.append((row['session1'], row['session2']))
        
        X1_test = np.array(X1_test)
        X2_test = np.array(X2_test)
        Y_test = np.array(Y_test)
        
        return model, (X1_test, X2_test, Y_test, test_pairs)
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def convert_to_serializable(obj):
    """Convert NumPy arrays and other non-serializable types to Python native types"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(x) for x in obj]
    return obj

def validate_test_data(X1_test, X2_test, Y_test, test_pairs):
    """Validate test data before running tests."""
    try:
        # Shape validation
        assert X1_test.shape == X2_test.shape, "Input shapes don't match"
        assert len(Y_test) == len(test_pairs), "Labels and pairs count mismatch"
        
        # Data quality checks
        assert np.all(np.isfinite(X1_test)), "Invalid values in X1_test"
        assert np.all(np.isfinite(X2_test)), "Invalid values in X2_test"
        assert set(Y_test).issubset({0, 1}), "Invalid labels found"
        
        # Class balance check
        pos_ratio = np.mean(Y_test)
        if not (0.2 <= pos_ratio <= 0.4):
            print(f"⚠️ Warning: Unusual class balance. Positive ratio: {pos_ratio:.2f}")
            
        return True
    except AssertionError as e:
        print(f"❌ Validation Error: {str(e)}")
        return False

def visualize_results(results, output_dir='test_results'):
    """Generate visualizations for test results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(results['roc_analysis']['fpr'], results['roc_analysis']['tpr'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve (AUC = {results['roc_analysis']['auc']:.3f})")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    # 2. Score Distribution
    plt.figure(figsize=(10, 6))
    failure_scores = [f['predicted'] for f in results['failure_analysis']]
    sns.histplot(failure_scores, bins=20)
    plt.axvline(results['threshold_info']['threshold'], color='r', linestyle='--')
    plt.title('Distribution of Failure Case Scores')
    plt.savefig(f"{output_dir}/failure_distribution.png")
    plt.close()

def run_model_tests(model, test_data, output_dir='test_results'):
    """
    Run comprehensive model testing.
    
    Args:
        model: Trained Siamese model.
        test_data: Tuple of (X1_test, X2_test, Y_test, test_pairs).
        output_dir: Directory to save results.
    """
    X1_test, X2_test, Y_test, test_pairs = test_data
    
    # Add validation step
    if not validate_test_data(X1_test, X2_test, Y_test, test_pairs):
        raise ValueError("Test data validation failed")
    
    tester = SiameseModelTester(model, output_dir)
    
    # Run all tests
    results = tester.evaluate_comprehensive(
        X1_test, X2_test, Y_test, test_pairs
    )
    
    # Convert results to JSON-serializable format
    serializable_results = convert_to_serializable(results)
    
    # Save results
    with open(f'{output_dir}/test_results.json', 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # Visualize results
    visualize_results(results, output_dir)
    
    return results

if __name__ == "__main__":
    print("Loading model and test data...")
    model, test_data = load_test_data_and_model()
    
    print("\nRunning tests...")
    results = run_model_tests(model, test_data)
    print("Testing complete. Results saved in test_results/")
    
    # Print summary of results
    print("\nResults Summary:")
    print(f"Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"AUC: {results['roc_analysis']['auc']:.4f}")
    print(f"Number of failures analyzed: {len(results['failure_analysis'])}")
