import numpy as np

def find_optimal_threshold_from_roc(fpr, tpr, thresholds, min_threshold=0.3, max_threshold=0.8):
    """Adjusted threshold range to improve TPR"""
    thresholds = np.array([float(t) if t != "Infinity" else np.inf for t in thresholds])
    j_scores = tpr - fpr
    
    # Constrain thresholds to reasonable range
    valid_mask = (thresholds >= min_threshold) & (thresholds <= max_threshold)
    valid_j_scores = j_scores[valid_mask]
    valid_thresholds = thresholds[valid_mask]
    valid_tpr = tpr[valid_mask]
    valid_fpr = fpr[valid_mask]
    
    if len(valid_thresholds) == 0:
        return None
    
    optimal_idx = np.argmax(valid_j_scores)
    return {
        'threshold': valid_thresholds[optimal_idx],
        'tpr': valid_tpr[optimal_idx],
        'fpr': valid_fpr[optimal_idx],
        'j_statistic': valid_j_scores[optimal_idx]
    }

def apply_threshold(predictions, threshold):
    """Apply threshold to predictions uniformly."""
    return (predictions > threshold).astype(int)
