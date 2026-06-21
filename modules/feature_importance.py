import pandas as pd
import numpy as np
import os
import glob

def analyze_feature_contributions(results_dir, target_col):
    print(f"\n{'='*70}")
    print(f" EXTRACTING MECHANICAL DRIVERS FOR: {target_col}")
    print(f"{'='*70}")
    
    # We aggregate the model's 'memory' from the nested run
    # (Note: You may need to adjust your nested_loso.py to save the 'coef_' 
    # for each fold if you want to be 100% precise, but this provides a strong proxy)
    
    # This script assumes you have saved the feature sets used in the nested run
    # For now, let's look at which features were selected most frequently across all 81 athletes
    
    # Example logic: 
    # If your nested loop saved the selected features for each fold:
    files = glob.glob(os.path.join(results_dir, f"feature_log_{target_col}_*.csv"))
    
    # If you didn't save logs yet, we can add a few lines to your nested loop 
    # to export the 'selector.support_' during the run.
    
    print(f"[INFO] Analyzing {len(files)} model folds...")
    # ... logic to average the impact of each GPS feature ...