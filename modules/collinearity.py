import numpy as np
import pandas as pd

def remove_collinearity(X_train, X_test, threshold=0.85):
    """
    Identifies highly correlated features in X_train and removes them from both sets.
    """
    corr_matrix = X_train.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Drop from both train and test
    X_train_clean = X_train.drop(columns=to_drop)
    X_test_clean = X_test.drop(columns=to_drop)
    
    return X_train_clean, X_test_clean