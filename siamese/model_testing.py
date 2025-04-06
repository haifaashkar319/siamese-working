import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    classification_report
)
from utils.threshold_optimizer import find_optimal_threshold_from_roc, apply_threshold
from utils.data_validator import validate_test_data
from utils.visualization import visualize_results

class SiameseModelTester:
    def __init__(self, model, output_dir='test_results'):
        """Initialize tester with model and output directory."""
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def predict_with_confidence(self, X1, X2):
        """Predict with confidence scores."""
        predictions = self.model.predict([X1, X2])
        confidence = np.abs(predictions - 0.5) * 2  # Scale confidence to 0-1
        return predictions, confidence

    def analyze_edge_cases(self, predictions, Y_test, threshold, margin=0.05):
        """Analyze predictions near the decision boundary."""
        # Convert predictions to 1D array if needed
        predictions = predictions.ravel()
        
        # Create mask for edge cases
        edge_mask = np.abs(predictions - threshold) < margin
        
        # Filter predictions and labels
        edge_cases = predictions[edge_mask]
        edge_labels = Y_test[edge_mask]
        
        return {
            'count': len(edge_cases),
            'true_positives': np.sum((edge_cases >= threshold) & (edge_labels == 1)),
            'false_positives': np.sum((edge_cases >= threshold) & (edge_labels == 0)),
            'edge_cases': [{
                'prediction': float(pred),
                'true_label': int(label)
            } for pred, label in zip(edge_cases, edge_labels)],
            'stats': {
                'mean': float(np.mean(edge_cases)),
                'std': float(np.std(edge_cases))
            }
        }

    def evaluate_comprehensive(self, X1_test, X2_test, Y_test, test_pairs):
        """Run comprehensive evaluation using optimal threshold."""
        predictions, confidence = self.predict_with_confidence(X1_test, X2_test)
        fpr, tpr, thresholds = roc_curve(Y_test, predictions)
        
        # Find optimal threshold
        optimal = find_optimal_threshold_from_roc(fpr, tpr, thresholds)
        if optimal is None:
            raise ValueError("Could not find valid threshold in specified range")
        
        threshold = optimal['threshold']
        binary_predictions = apply_threshold(predictions, threshold)
        
        # Analyze edge cases
        edge_case_analysis = self.analyze_edge_cases(predictions, Y_test, threshold)
        
        return {
            'basic_metrics': self._compute_basic_metrics(Y_test, binary_predictions),
            'roc_analysis': self._compute_roc_analysis(fpr, tpr, thresholds),
            'confusion_matrix': self._compute_confusion_matrix(Y_test, binary_predictions),
            'failure_analysis': self._analyze_failures(predictions, Y_test, test_pairs, threshold),
            'confidence_scores': confidence.tolist(),
            'threshold_info': optimal,
            'edge_case_analysis': edge_case_analysis
        }

    def _compute_basic_metrics(self, Y, binary_predictions):
        return classification_report(Y, binary_predictions, output_dict=True)
    
    def _compute_roc_analysis(self, fpr, tpr, thresholds):
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, 'b-', label=f'ROC (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.savefig(f'{self.output_dir}/roc_curve.png')
        plt.close()
        
        return {'auc': roc_auc, 'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
    
    def _compute_confusion_matrix(self, Y, binary_predictions):
        cm = confusion_matrix(Y, binary_predictions)
        
        # Plot confusion matrix with improved styling
        plt.figure(figsize=(8, 6))
        sns.set_theme(style="white")  # Clean style
        
        # Create heatmap with better formatting
        ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Different', 'Same'],
            yticklabels=['Different', 'Same'],
            square=True,  # Make cells square
            cbar_kws={"shrink": .8}  # Adjust colorbar
        )
        
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        # Improve layout
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return cm.tolist()
    
    def _analyze_failures(self, predictions, Y, test_pairs, threshold):
        binary_predictions = apply_threshold(predictions, threshold)
        
        failures = []
        for i, (pred, true_label) in enumerate(zip(binary_predictions, Y)):
            if pred != true_label:
                failures.append({
                    'pair': test_pairs[i],
                    'true_label': int(true_label),
                    'predicted': float(predictions[i][0])
                })
        
        return failures
