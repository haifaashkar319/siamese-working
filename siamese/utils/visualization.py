import os
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_results(results, output_dir='test_results'):
    """Generate visualizations for test results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # ROC Curve
    plt.figure(figsize=(10, 6))
    plt.plot(results['roc_analysis']['fpr'], results['roc_analysis']['tpr'])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve (AUC = {results['roc_analysis']['auc']:.3f})")
    plt.savefig(f"{output_dir}/roc_curve.png")
    plt.close()
    
    # Score Distribution
    plt.figure(figsize=(10, 6))
    failure_scores = [f['predicted'] for f in results['failure_analysis']]
    sns.histplot(failure_scores, bins=20)
    plt.axvline(results['threshold_info']['threshold'], color='r', linestyle='--')
    plt.title('Distribution of Failure Case Scores')
    plt.savefig(f"{output_dir}/failure_distribution.png")
    plt.close()
