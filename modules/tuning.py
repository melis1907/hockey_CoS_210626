from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

def tune_and_predict(X_train_optimal, y_train, X_test_optimal):
    tuning_configurations = {
        "Lasso": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("model", Lasso(max_iter=5000))]),
            "grid": {'model__alpha': [0.1, 0.5, 1.0, 5.0]}
        },
        "Ridge": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
            "grid": {'model__alpha': [1.0, 10.0, 100.0]}
        },
        "XGBoost": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("model", XGBRegressor(random_state=42, n_estimators=50))]),
            "grid": {
                'model__learning_rate': [0.01, 0.05],
                'model__max_depth': [2, 3]
            }
        },
        "RandomForest": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor(random_state=42))]),
            "grid": {
                'model__n_estimators': [50, 100],
                'model__max_depth': [3, 5],
                'model__min_samples_leaf': [1, 5]
            }
        }
    }

    best_fold_mae = float('inf')
    best_fold_model = None
    
    # --- NEW: Track the winning name and params ---
    best_algorithm_name = ""
    best_algorithm_params = {}

    for name, config in tuning_configurations.items():
        grid_search = GridSearchCV(
            estimator=config["pipeline"],
            param_grid=config["grid"],
            cv=3, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train_optimal, y_train)
        
        if -grid_search.best_score_ < best_fold_mae:
            best_fold_mae = -grid_search.best_score_
            best_fold_model = grid_search.best_estimator_
            best_algorithm_name = name
            best_algorithm_params = grid_search.best_params_

    prediction = best_fold_model.predict(X_test_optimal)
    final_estimator = best_fold_model.named_steps['model']
    
    if hasattr(final_estimator, 'coef_'):
        weights = final_estimator.coef_
    elif hasattr(final_estimator, 'feature_importances_'):
        weights = final_estimator.feature_importances_
    else:
        weights = [0] * len(X_train_optimal.columns)
        
    fold_importances = dict(zip(X_train_optimal.columns, weights))
    
    # Pack the model info into a dictionary
    fold_model_info = {
        'Algorithm': best_algorithm_name,
        'Hyperparameters': best_algorithm_params
    }
    
    # Return all THREE outputs
    return prediction, fold_importances, fold_model_info