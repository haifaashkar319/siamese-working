import numpy as np

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
