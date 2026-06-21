import os
import pandas as pd
import numpy as np

def prepare_blood_targets(csv_path, output_path):
    print(f"\n[INFO] Loading Blood data from '{csv_path}'...")
    
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    #change column name to Date.
    df.rename(columns={'sample_date_blood': 'Date', 'participant_id': 'Athlete ID'}, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2. Replace 0.0 with NaN 
    cols_to_clean = df.columns.difference(['Athlete ID', 'Date'])
    # The csv file already has NaN values for 0.0, so this line is not needed.
    
    # 3. Handle duplicates: Group by Athlete and Date, take the mean of same-day tests
    df = df.groupby(['Athlete ID', 'Date'])[cols_to_clean].mean().reset_index()
    
    # Sort chronologically per athlete
    df = df.sort_values(by=['Athlete ID', 'Date']).reset_index(drop=True)
    
    # 4. Calculate the Delta (Change) for Biomarkers
    delta_df = df[['Athlete ID', 'Date']].copy()
    
    # Calculate days since last test
    delta_df['days_since_last_test'] = df.groupby('Athlete ID')['Date'].diff().dt.days
    
    # Calculate absolute change for each biomarker
    for col in cols_to_clean:
        delta_df[f'delta_{col}'] = df.groupby('Athlete ID')[col].diff()

    # 5. Drop the first test for each athlete 
    delta_df = delta_df.dropna(subset=['days_since_last_test']).reset_index(drop=True)
    
    # --- Create Output Folders Safely ---
    output_dir = os.path.dirname(output_path)
    if output_dir: 
        os.makedirs(output_dir, exist_ok=True)
    
    # Save targets
    delta_df.to_csv(output_path, index=False)
    print(f"[INFO] Blood target deltas saved to '{output_path}'")
    print(f"[INFO] Generated {len(delta_df)} prediction windows.")
    
    return df, delta_df