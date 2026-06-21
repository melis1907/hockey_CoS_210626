from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def select_optimal_features(X_train_clean, y_train, X_test_clean):
    """
    Uses RFECV with a Ridge estimator to extract the optimal features.
    """
    rfecv_estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0, random_state=42))
    ])
    
    # 3-fold inner CV to save computation time inside the nested loop
    selector = RFECV(
        rfecv_estimator, 
        step=2, 
        cv=3, 
        scoring='neg_mean_absolute_error', 
        min_features_to_select=2, 
        importance_getter='named_steps.ridge.coef_'
    )
    
    selector.fit(X_train_clean, y_train)
    
    optimal_features = X_train_clean.columns[selector.support_].tolist()
    
    X_train_optimal = X_train_clean[optimal_features]
    X_test_optimal = X_test_clean[optimal_features]
    
    return X_train_optimal, X_test_optimal, optimal_features