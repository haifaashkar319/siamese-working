import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, roc_auc_score

def perform_cross_validation(model_fn, X1, X2, Y, n_splits=5):
    """
    Perform k-fold cross-validation on Siamese network.
    
    Args:
        model_fn: Function that creates and returns a new model instance
        X1, X2: Input feature arrays for both branches
        Y: Labels
        n_splits: Number of folds for cross-validation
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X1)):
        print(f"\nTraining Fold {fold+1}/{n_splits}")
        
        # Split data
        X1_train, X1_val = X1[train_idx], X1[val_idx]
        X2_train, X2_val = X2[train_idx], X2[val_idx]
        y_train, y_val = Y[train_idx], Y[val_idx]
        
        # Create and train new model instance
        model = model_fn()
        model.fit(
            [X1_train, X2_train],
            y_train,
            validation_data=([X1_val, X2_val], y_val),
            epochs=30,
            batch_size=32,
            verbose=0
        )
        
        # Evaluate
        predictions = model.predict([X1_val, X2_val])
        accuracy = accuracy_score(y_val, (predictions > 0.76).astype(int))
        auc = roc_auc_score(y_val, predictions)
        
        fold_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'auc': auc
        })
        print(f"Fold {fold+1} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
    
    # Compute average metrics
    avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
    avg_auc = np.mean([r['auc'] for r in fold_results])
    std_accuracy = np.std([r['accuracy'] for r in fold_results])
    std_auc = np.std([r['auc'] for r in fold_results])
    
    return {
        'fold_results': fold_results,
        'average_accuracy': avg_accuracy,
        'average_auc': avg_auc,
        'std_accuracy': std_accuracy,
        'std_auc': std_auc
    }
