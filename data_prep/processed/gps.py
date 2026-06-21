import os
import pandas as pd
import numpy as np
from tqdm import tqdm 

# ==========================================
# Helper Function
# ==========================================
def clean_dataset(df):
    # Standardize specific string errors to null values
    df.replace('I am a legend', np.nan, inplace=True)
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
    return df

# ==========================================
# Main Processing Function
# ==========================================
def process_gps_data(input_path, meta_output_path, engineered_output_path):
    """
    Loads GPS data, cleans it, calculates EWMA features, 
    and splits the output into metadata and engineered feature files.
    """
    print(f"\n[INFO] Loading GPS data from '{input_path}'...")
    
    # 1. Load Datasets
    gps_df = pd.read_csv(input_path)

    # Standardize the identifier column if it exists
    if 'Player Name' in gps_df.columns:
        gps_df.rename(columns={'Player Name': 'Athlete ID'}, inplace=True)

    # 2. Data Standardization
    gps_df = clean_dataset(gps_df)

    # 3. Sort Chronologically
    gps_df = gps_df.sort_values(by=['Athlete ID', 'Date']).reset_index(drop=True)

    # 4. Feature Construction: All GPS Metrics
    exclude_cols = [
        'Athlete ID', 'Period Name', 'Period Number', 
        'Date', 'Day Name', 'Total Duration'
    ]

    # Dynamically grab every other column in the dataset (the Load Metrics)
    load_metrics = [col for col in gps_df.columns if col not in exclude_cols]

    acute_span = 7
    chronic_span = 28

    for metric in tqdm(load_metrics, desc="Calculating EWMA Features"):
        gps_df[metric] = pd.to_numeric(gps_df[metric], errors='coerce')
        gps_df[metric] = gps_df.groupby('Athlete ID')[metric].ffill()
        
        gps_df[f'{metric}_Acute_EWMA'] = gps_df.groupby('Athlete ID')[metric].transform(
            lambda x: x.ewm(span=acute_span, adjust=False).mean()
        )
        
        gps_df[f'{metric}_Chronic_EWMA'] = gps_df.groupby('Athlete ID')[metric].transform(
            lambda x: x.ewm(span=chronic_span, adjust=False).mean()
        )

    # 5. Split and Export Data
    print("Splitting data into Meta Data and Engineered files...")

    # File 1: Meta Data (Excluded columns + Date + Athlete ID)
    # Ensure we only try to keep columns that actually exist
    valid_exclude_cols = [col for col in exclude_cols if col in gps_df.columns]
    meta_data_df = gps_df[valid_exclude_cols]

    # File 2: Engineered Features 
    # We drop the text/descriptive columns, but KEEP Athlete ID, Date, AND Total Duration
    cols_to_drop = [col for col in exclude_cols if col not in ['Athlete ID', 'Date', 'Total Duration']]
    valid_cols_to_drop = [col for col in cols_to_drop if col in gps_df.columns]
    engineered_df = gps_df.drop(columns=valid_cols_to_drop)

    # --- Create Output Folders Safely ---
    os.makedirs(os.path.dirname(meta_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(engineered_output_path), exist_ok=True)

    # Save both files
    meta_data_df.to_csv(meta_output_path, index=False)
    engineered_df.to_csv(engineered_output_path, index=False)

    print(f"[INFO] Success! Meta Data saved to: '{meta_output_path}'")
    print(f"[INFO] Success! Engineered features saved to: '{engineered_output_path}'")
    
    return meta_data_df, engineered_df