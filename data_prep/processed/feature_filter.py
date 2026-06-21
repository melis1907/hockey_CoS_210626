import pandas as pd
import numpy as np

# ==========================================
# 1. Load the Engineered Data
# ==========================================
print("Loading engineered features...")
# Using the file that contains our Athlete ID and Meta Data standardization
df = pd.read_csv('processed/engineered_gps_features.csv')

# ==========================================
# 2. Isolate Numeric Load Metrics
# ==========================================
# We must exclude Athlete ID and Date from mathematical checks
exclude_cols = ['Athlete ID', 'Date', 'Total Duration']
numeric_cols = [col for col in df.columns if col not in exclude_cols]

# Ensure everything is strictly numeric
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Evaluating {len(numeric_cols)} numeric features...\n")

# ==========================================
# 3. Step 1: The Sparsity Filter (Zero-Inflated)
# ==========================================
# Threshold: If a metric is exactly 0.0 for more than 30% of the season, it fails.
SPARSITY_THRESHOLD = 0.30 

print(f"--- SPARSITY CHECK (Fails if > {SPARSITY_THRESHOLD*100}% zeros) ---")
sparse_failed = []
passed_sparsity = []

for col in numeric_cols:
    # Calculate percentage of exact zeros (ignoring NaNs for the calculation)
    valid_data = df[col].dropna()
    if len(valid_data) == 0:
        continue
        
    zero_ratio = (valid_data == 0).sum() / len(valid_data)
    
    if zero_ratio > SPARSITY_THRESHOLD:
        sparse_failed.append((col, zero_ratio))
    else:
        passed_sparsity.append(col)

print(f"Failed Sparsity: {len(sparse_failed)} features dropped.")
# Optional: Print the worst offenders
# for f, r in sorted(sparse_failed, key=lambda x: x[1], reverse=True)[:5]:
#     print(f"  - {f}: {r*100:.1f}% zeros")

# ==========================================
# 4. Step 2: The Variance Filter (Flatlines)
# ==========================================
# We use the Coefficient of Variation (CV = standard deviation / mean).
# Threshold: If CV < 0.10 (10% variance), the metric is too static for the HMM.
CV_THRESHOLD = 0.10

print(f"\n--- VARIANCE CHECK (Fails if CV < {CV_THRESHOLD}) ---")
variance_failed = []
final_survivors = []

for col in passed_sparsity:
    valid_data = df[col].dropna()
    mean_val = valid_data.mean()
    std_val = valid_data.std()
    
    # Avoid division by zero
    if mean_val == 0 or pd.isna(mean_val):
        variance_failed.append((col, 0))
        continue
        
    cv = abs(std_val / mean_val)
    
    if cv < CV_THRESHOLD:
        variance_failed.append((col, cv))
    else:
        final_survivors.append(col)

print(f"Failed Variance: {len(variance_failed)} features dropped.")

# ==========================================
# 5. Output the Final Surviving Candidates
# ==========================================

#save the remaining features to a new CSV for the HMM
survivor_df = df[final_survivors + ['Athlete ID', 'Date']]
survivor_df.to_csv('processed/hmm_ready_features.csv', index=False)


print("\n==========================================")
print(f"FINAL SURVIVORS: {len(final_survivors)} Metrics Passed All Tests")
print("==========================================")
print("These metrics have sufficient variance and density to drive the HMM:\n")

# Print the survivors sorted alphabetically for easy reading
for metric in sorted(final_survivors):
    print(f"- {metric}")