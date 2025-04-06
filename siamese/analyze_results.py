import json
import numpy as np
import pandas as pd
from utils.threshold_optimizer import find_optimal_threshold_from_roc

def analyze_test_results(results_file="test_results/test_results.json"):
    """Analyze test results using optimized threshold."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n=== Performance Analysis ===")
    
    # Print metrics using threshold info from results
    print("\nThreshold Information:")
    threshold_info = results.get('threshold_info', {})
    print(f"Optimal Threshold: {threshold_info.get('threshold', 'N/A')}")
    print(f"True Positive Rate: {threshold_info.get('tpr', 'N/A')}")
    print(f"False Positive Rate: {threshold_info.get('fpr', 'N/A')}")
    
    # Basic Metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"AUC Score: {results['roc_analysis']['auc']:.4f}")
    
    # Class-wise Performance
    print("\nClass-wise Performance:")
    for label in ['0', '1']:
        metrics = results['basic_metrics'][label]
        label_name = "Different Users" if label == '0' else "Same User"
        print(f"\n{label_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-score: {metrics['f1-score']:.4f}")
        print(f"  Support: {int(metrics['support'])} samples")
    
    # Analyze Failures
    failures = results['failure_analysis']
    print(f"\nFailure Analysis ({len(failures)} failures):")
    
    false_positives = [f for f in failures if f['true_label'] == 0]
    false_negatives = [f for f in failures if f['true_label'] == 1]
    
    print(f"\nFalse Positives (Different users classified as same): {len(false_positives)}")
    for fp in false_positives[:3]:  # Show first 3
        print(f"  Pair: {fp['pair'][0]} vs {fp['pair'][1]} (pred: {fp['predicted']:.4f})")
    
    print(f"\nFalse Negatives (Same user classified as different): {len(false_negatives)}")
    for fn in false_negatives[:3]:  # Show first 3
        print(f"  Pair: {fn['pair'][0]} vs {fn['pair'][1]} (pred: {fn['predicted']:.4f})")
    
    # Add Edge Case Analysis
    print("\nEdge Case Analysis (Scores near threshold):")
    threshold = threshold_info.get('threshold', 0.7)
    margin = 0.05  # Analyze cases within ±0.05 of threshold
    
    all_predictions = []
    for case in failures:
        all_predictions.append({
            'session1': case['pair'][0],
            'session2': case['pair'][1],
            'true_label': case['true_label'],
            'predicted': case['predicted'],
            'distance': abs(case['predicted'] - threshold)
        })
    
    edge_cases = pd.DataFrame(all_predictions)
    edge_cases = edge_cases[edge_cases['distance'] < margin].sort_values('distance')
    
    print(f"\nFound {len(edge_cases)} cases within ±{margin} of threshold {threshold:.4f}")
    if len(edge_cases) > 0:
        print("\nMost borderline cases:")
        for _, case in edge_cases.head().iterrows():
            label_text = "Same User" if case['true_label'] == 1 else "Different Users"
            print(f"  {case['session1']} vs {case['session2']}")
            print(f"    True: {label_text}, Score: {case['predicted']:.4f}")
            print(f"    Distance from threshold: {case['distance']:.4f}")
    
    # Calculate edge case statistics
    if len(edge_cases) > 0:
        print("\nEdge Case Statistics:")
        print(f"  Average score: {edge_cases['predicted'].mean():.4f}")
        print(f"  Score std dev: {edge_cases['predicted'].std():.4f}")
        print("  Distribution of true labels in edge cases:")
        print(edge_cases['true_label'].value_counts().to_string())
    
    # Threshold Analysis
    fpr = np.array(results['roc_analysis']['fpr'])
    tpr = np.array(results['roc_analysis']['tpr'])
    thresholds = np.array([float(t) if t != "Infinity" else np.inf 
                          for t in results['roc_analysis']['thresholds']])
    
    # Find optimal threshold
    optimal = find_optimal_threshold_from_roc(fpr, tpr, thresholds)
    
    print("\nThreshold Analysis:")
    print(f"Optimal Threshold: {float(optimal['threshold']):.4f}")
    print(f"True Positive Rate: {float(optimal['tpr']):.4f}")
    print(f"False Positive Rate: {float(optimal['fpr']):.4f}")
    print(f"Youden's J Statistic: {float(optimal['j_statistic']):.4f}")

if __name__ == "__main__":
    analyze_test_results()
