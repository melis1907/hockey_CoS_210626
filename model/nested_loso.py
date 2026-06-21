import pandas as pd
import numpy as np
import os
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tqdm import tqdm
import warnings

# Import your custom modules
from modules.collinearity import remove_collinearity
from modules.feature_selection import select_optimal_features
from modules.tuning import tune_and_predict

warnings.filterwarnings("ignore")
def process_single_marker(df, target_col, output_dir):
    """
    Executes the nested LOSO pipeline for a single biomarker,
    extracts feature stability, AND logs optimal hyperparameters per fold.
    """
    target_df = df.dropna(subset=[target_col]).reset_index(drop=True)

    if len(target_df) < 20:
        return None, None, None

    target_cols = [c for c in df.columns if c.startswith('delta_')]
    cols_to_exclude = ['Athlete ID', 'Window_End_Date', 'Date', 'is_off_day'] + target_cols
    
    X_full = target_df.drop(columns=cols_to_exclude, errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    y_full = target_df[target_col]
    groups = target_df['Athlete ID']

    logo = LeaveOneGroupOut()
    
    all_actuals = []
    all_predictions = []
    athlete_ids = []
    
    feature_stability_tracker = {}
    model_tracking_log = []

    for train_idx, test_idx in tqdm(logo.split(X_full, y_full, groups), total=groups.nunique(), desc=f"   Testing Athletes", leave=False):
        
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]
        test_athlete_id = groups.iloc[test_idx].iloc[0]

        X_train_clean, X_test_clean = remove_collinearity(X_train, X_test, threshold=0.85)
        X_train_opt, X_test_opt, _ = select_optimal_features(X_train_clean, y_train, X_test_clean)

        print(f"      [INFO] Athlete {test_athlete_id} | Training on {len(X_train_opt)} samples, Testing on {len(X_test_opt)} samples.")
        # CATCHING ALL THREE OUTPUTS HERE
        predictions, fold_importances, fold_model_info = tune_and_predict(X_train_opt, y_train, X_test_opt)
        
        all_actuals.extend(y_test.tolist())
        all_predictions.extend(predictions.tolist())
        athlete_ids.extend([test_athlete_id] * len(y_test))
        
        for feat, weight in fold_importances.items():
            if feat not in feature_stability_tracker:
                feature_stability_tracker[feat] = {'selections': 0, 'total_weight': 0.0}
            feature_stability_tracker[feat]['selections'] += 1
            feature_stability_tracker[feat]['total_weight'] += weight
            
        # Logging the model architecture and parameters
        clean_params = {k.replace('model__', ''): v for k, v in fold_model_info['Hyperparameters'].items()}
        
        model_tracking_log.append({
            'Holdout_Athlete_ID': test_athlete_id,
            'Winning_Algorithm': fold_model_info['Algorithm'],
            'Optimal_Hyperparameters': str(clean_params)
        })

    actuals, preds = np.array(all_actuals), np.array(all_predictions)
    r2 = r2_score(actuals, preds)
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))

    # 1. Export Predictions CSV
    results_df = pd.DataFrame({
        'Athlete ID': athlete_ids,
        'Actual_Delta': actuals,
        'Predicted_Delta': np.round(preds, 3),
        'Error': np.round(abs(actuals - preds), 3)
    })
    results_df.to_csv(os.path.join(output_dir, f"nested_loso_{target_col}.csv"), index=False)
    
    # 2. Export Feature Stability CSV
    total_folds = groups.nunique()
    stability_data = []
    for feat, stats in feature_stability_tracker.items():
        avg_weight = stats['total_weight'] / stats['selections']
        selection_pct = (stats['selections'] / total_folds) * 100
        stability_data.append({
            'Feature': feat,
            'Selection_Percentage': np.round(selection_pct, 2),
            'Average_Weight': np.round(avg_weight, 4)
        })
    if stability_data:
        stability_df = pd.DataFrame(stability_data).sort_values(by='Selection_Percentage', ascending=False)
        stability_df.to_csv(os.path.join(output_dir, f"nested_importances_{target_col}.csv"), index=False)
        
    # 3. Export Model and Hyperparameter Tracking CSV
    if model_tracking_log:
        hyperparameter_df = pd.DataFrame(model_tracking_log)
        hyperparameter_df.to_csv(os.path.join(output_dir, f"nested_hyperparameters_{target_col}.csv"), index=False)
    
    return r2, mae, rmse
def execute_nested_suite(dataset_path, output_dir, mode="SINGLE", target_single=None, custom_list=None):
    print(f"\n{'='*75}")
    print(f" EXECUTING MODULAR NESTED LOSO | Mode: {mode.upper()}")
    print(f"{'='*75}")

    df = pd.read_csv(dataset_path)
    all_deltas = [c for c in df.columns if c.startswith('delta_')]
    
    # Determine which markers to evaluate based on the selected mode
    if mode == "SINGLE":
        markers_to_run = [target_single]
    elif mode == "CUSTOM_LIST":
        markers_to_run = custom_list
    else:
        markers_to_run = all_deltas
        
    print(f"Target Count: {len(markers_to_run)} Biomarker(s) scheduled for processing.\n")
    
    os.makedirs(output_dir, exist_ok=True)
    master_results = []
    
    for target in markers_to_run:
        print(f"⚙️ Processing: {target}")
        
        if target not in df.columns:
            print(f"  [ERROR] {target} not found in dataset. Skipping.")
            continue
            
        r2, mae, rmse = process_single_marker(df, target, output_dir)
        
        if r2 is None:
            print(f"  [SKIP] Not enough valid windows for {target}.")
            continue
            
        print(f"  [RESULT] True Unbiased R²: {r2:.3f} | MAE: {mae:.3f} | RMSE: {rmse:.3f}\n")
        
        master_results.append({
            "Biomarker": target,
            "Unbiased_R_Squared": round(r2, 3),
            "Unbiased_MAE": round(mae, 3),
            "Unbiased_RMSE": round(rmse, 3)
        })
        
    # Generate the final master leaderboard
    if master_results:
        results_df = pd.DataFrame(master_results).sort_values(by="Unbiased_R_Squared", ascending=False)
        master_csv = os.path.join(output_dir, "master_nested_loso_leaderboard.csv")
        results_df.to_csv(master_csv, index=False)
        
        print(f"{'='*75}")
        print(f"[SUCCESS] Suite complete. Master leaderboard saved to: {master_csv}")
        print("\nTop Unbiased Markers:")
        print(results_df.head(5).to_string(index=False))

if __name__ == "__main__":
    dataset = 'final_ewma_ml_dataset.csv'
    output_directory = 'results/proofs_nested'
    
    # ==========================================
    # MISSION CONTROL: CONFIGURE YOUR RUN HERE
    # ==========================================
    
    # Options: "SINGLE", "CUSTOM_LIST", or "ALL"
    RUN_MODE = "CUSTOM_LIST" 
    
    # Used if RUN_MODE = "SINGLE"
    SINGLE_MARKER = 'delta_leptin' 
    
    # Used if RUN_MODE = "CUSTOM_LIST"
    # Replace these with your top 5 markers from the previous tests
    CUSTOM_MARKERS = [
        'delta_red_blood_count', 
        'delta_mchc', 
        'delta_mpv', 
        'delta_mcv', 
        'delta_hct'
    ]
    
    execute_nested_suite(
        dataset_path=dataset, 
        output_dir=output_directory, 
        mode=RUN_MODE, 
        target_single=SINGLE_MARKER, 
        custom_list=CUSTOM_MARKERS
    )