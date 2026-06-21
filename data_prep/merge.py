import pandas as pd
import numpy as np
import warnings

#This include some changes. Check the results before sending out. 

warnings.filterwarnings("ignore")

def merge_clinical_and_all_tracking_data(blood_csv, gps_ewma_csv, hat_csv, output_path):
    print(f"\n{'='*70}")
    print(" UNIVERSAL MERGE: ALL GPS METRICS -> BLOOD WINDOWS")
    print(f"{'='*70}")

    # 1. Load Data
    blood_df = pd.read_csv(blood_csv)
    gps_df = pd.read_csv(gps_ewma_csv)
    hat_df = pd.read_csv(hat_csv)

    # 2. Date Alignment
    blood_df['Date'] = pd.to_datetime(blood_df['Date'])
    gps_df['Date'] = pd.to_datetime(gps_df['Date'])
    hat_df['Date'] = pd.to_datetime(hat_df['date_hat'])
    hat_clean = hat_df.groupby(['participant_id', 'Date'])['score_hat'].mean().reset_index()
    hat_clean = hat_clean.rename(columns={'participant_id': 'Athlete ID'})

    # 3. Identify all GPS features dynamically
    exclude_gps_cols = ['Athlete ID', 'Date']
    gps_features = [c for c in gps_df.columns if c not in exclude_gps_cols]
    
    features = []

    for idx, row in blood_df.iterrows():
        athlete = row['Athlete ID']
        end_date = row['Date']
        window_days = row.get('days_since_last_test', 30)
        start_date = end_date - pd.Timedelta(days=window_days)
        
        # Filter windows
        window_gps = gps_df[(gps_df['Athlete ID'] == athlete) & 
                            (gps_df['Date'] > start_date) & 
                            (gps_df['Date'] <= end_date)]
        
        if window_gps.empty:
            continue
        
        # Calculate dynamic features
        row_features = {'Athlete ID': athlete, 'Window_End_Date': end_date}
        
        for feat in gps_features:
            # Always grab the state on the day of the blood draw
            row_features[f'TestDay_{feat}'] = window_gps.iloc[-1][feat]
            
            # Only sum the raw metrics (do not sum EWMA rolling averages)
            if 'EWMA' not in feat:
                row_features[f'Sum_{feat}'] = window_gps[feat].sum()

        # Add HAT
        window_hats = hat_clean[(hat_clean['Athlete ID'] == athlete) & 
                                (hat_clean['Date'] >= start_date) & 
                                (hat_clean['Date'] <= end_date)]
        row_features['Avg_HAT_Score'] = window_hats['score_hat'].mean() if not window_hats.empty else np.nan
        
        # Append targets
        for t in [c for c in blood_df.columns if 'delta_' in c]:
            row_features[t] = row[t]
            
        features.append(row_features)

    final_df = pd.DataFrame(features)
    
    # ML Best Practice: Leaving NaNs intact to prevent Data Leakage 
    # (Handle imputation later during the Train/Test split phase)
    
    final_df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Universal dataset built with {len(final_df)} windows and {len(final_df.columns)} columns.")
    print(f"[INFO] Final dataset saved to '{output_path}'")
    
    return final_df