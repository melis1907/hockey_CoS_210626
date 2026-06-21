import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

warnings.filterwarnings("ignore")

# ==========================================
# CONSTANTS FOR PLOTTING
# ==========================================
STABLE_THRESHOLD = 20   # % for "stable" features
DISPERSED_THRESHOLD = 5 # % for "dispersed / weak signal" features

def save_all_features_csv(importance_csv, output_dir, original_marker_name):
    """Save a ranked CSV of ALL features for a marker, with a stability label column."""
    clean_name = original_marker_name.replace('delta_', '').upper().replace('_', ' ')

    if not os.path.exists(importance_csv):
        return

    df = pd.read_csv(importance_csv)
    df = df.sort_values(by='Average_Weight', ascending=False).reset_index(drop=True)
    df.insert(0, 'Rank', df.index + 1)

    def label_stability(freq):
        if freq >= STABLE_THRESHOLD:
            return 'Stable'
        elif freq >= DISPERSED_THRESHOLD:
            return 'Weak'
        else:
            return 'Noise'

    df['Stability'] = df['Selection_Percentage'].apply(label_stability)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"all_features_{clean_name.replace(' ', '_')}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [CSV] All features saved for {clean_name} → {csv_path}")


def plot_top_features(importance_csv, output_dir, original_marker_name, threshold=STABLE_THRESHOLD, dispersed=False):
    clean_name = original_marker_name.replace('delta_', '').upper().replace('_', ' ')
    
    if not os.path.exists(importance_csv):
        print(f"  [SKIP] Could not find importance data for {clean_name}.")
        return False

    df = pd.read_csv(importance_csv)
    df['Average_Absolute_Weight'] = df['Average_Weight'].abs()
    
    stable_features = df[df['Selection_Percentage'] >= threshold].copy()
    top_features = stable_features.sort_values(by='Average_Absolute_Weight', ascending=False).head(10)
    
    if top_features.empty:
        if not dispersed:
            print(f"  [INFO] No stable features found for {clean_name}. Model relied on dispersed weak signals.")
        return False

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    palette = 'magma' if dispersed else 'viridis'
    bars = sns.barplot(
        data=top_features, 
        y='Feature', 
        x='Average_Absolute_Weight', 
        palette=palette,
        edgecolor='black'
    )

    if dispersed:
        title = (f'Weak / Dispersed Signal Drivers for {clean_name}\n'
                 f'(Nested LOSO — Low Stability, threshold ≥{threshold}%)')
    else:
        title = (f'Primary Mechanical Drivers for {clean_name}\n'
                 f'(Nested LOSO Feature Stability, threshold ≥{threshold}%)')
    
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Average Absolute Impact Weight', fontsize=11)
    plt.ylabel('GPS / Subjective Feature', fontsize=11)

    if dispersed:
        plt.gcf().text(
            0.99, 0.01, f'WEAK SIGNAL — threshold ≥{threshold}%',
            ha='right', va='bottom', fontsize=9,
            color='red', alpha=0.5, fontstyle='italic'
        )
    
    for index, value in enumerate(top_features['Average_Absolute_Weight']):
        freq = top_features.iloc[index]['Selection_Percentage']
        plt.text(
            value + (top_features['Average_Absolute_Weight'].max() * 0.02), 
            index, 
            f"Selected in {freq:.0f}% of folds", 
            va='center', 
            fontsize=10, 
            color='black',
            fontweight='500'
        )

    plt.xlim(0, top_features['Average_Absolute_Weight'].max() * 1.3)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"drivers_{clean_name.replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    label = "dispersed-signal" if dispersed else "driver"
    print(f"  [SUCCESS] Generated {label} plot for {clean_name}")
    return True


def auto_generate_top_drivers(leaderboard_csv, results_dir, output_dir, top_n=5):
    print(f"\n{'='*75}")
    print(f" AUTOPILOT: GENERATING FEATURE IMPORTANCE VISUALS FOR TOP {top_n}")
    print(f"{'='*75}")

    if not leaderboard_csv or not os.path.exists(leaderboard_csv):
        print(f"[ERROR] Could not find the master leaderboard at {leaderboard_csv}.")
        return

    df_leaderboard = pd.read_csv(leaderboard_csv)

    sort_column = None
    if "Unbiased_R_Squared" in df_leaderboard.columns:
        sort_column = "Unbiased_R_Squared"
    elif "R_Squared" in df_leaderboard.columns:
        sort_column = "R_Squared"
    else:
        print("[ERROR] Could not find an R-Squared column to sort by.")
        return

    df_top = df_leaderboard.sort_values(by=sort_column, ascending=False).head(top_n)
    top_markers_list = df_top['Biomarker'].tolist()
    
    print(f"[INFO] Automatically identified Top {top_n} markers:")
    for m in top_markers_list:
        print(f" • {m}")
    print("-" * 50)

    dispersed_markers = []
    os.makedirs(output_dir, exist_ok=True)

    features_dir = os.path.join(output_dir, 'all_features')
    for marker in top_markers_list:
        importance_file = os.path.join(results_dir, f'nested_importances_{marker}.csv')
        save_all_features_csv(importance_file, features_dir, marker)
        plotted = plot_top_features(importance_file, output_dir, marker, threshold=STABLE_THRESHOLD, dispersed=False)
        if not plotted and os.path.exists(importance_file):
            dispersed_markers.append(marker)

    if dispersed_markers:
        dispersed_dir = os.path.join(output_dir, 'dispersed_signals')
        print(f"\n[INFO] Running dispersed-signal pass for {len(dispersed_markers)} marker(s)...")
        for marker in dispersed_markers:
            importance_file = os.path.join(results_dir, f'nested_importances_{marker}.csv')
            plot_top_features(importance_file, dispersed_dir, marker, threshold=DISPERSED_THRESHOLD, dispersed=True)

    print(f"\n{'='*75}")
    print("[COMPLETED] All visuals have been generated and saved.")
    print(f"  Stable drivers  : {output_dir}")
    print(f"  All features    : {features_dir}")
    if dispersed_markers:
        print(f"  Dispersed signals : {os.path.join(output_dir, 'dispersed_signals')}")
        
def plot_nested_leaderboard(csv_path, output_dir, top_n=5):
    print(f"\n{'='*75}")
    print(f" GENERATING LEADERBOARD VISUALIZATION (Top {top_n})")
    print(f"{'='*75}")

    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find {csv_path}. Ensure the master suite has finished running.")
        return

    # Load the leaderboard
    df = pd.read_csv(csv_path)
    
    # Sort securely by R_Squared descending and take the top N
    df = df.sort_values(by="Unbiased_R_Squared", ascending=False).head(top_n)
    
    if df.empty:
        print("[ERROR] Leaderboard is empty.")
        return

    # Clean up the marker names for the plot (e.g., 'delta_mchc' -> 'MCHC')
    df['Clean_Name'] = df['Biomarker'].str.replace('delta_', '').str.upper().str.replace('_', ' ')

    # Set up the plotting style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # Color code based on predictive viability (positive vs negative R2)
    colors = ['#1f77b4' if val > 0 else '#d62728' for val in df['Unbiased_R_Squared']]

    # Create a horizontal bar chart
    bars = sns.barplot(
        data=df,
        x='Unbiased_R_Squared',
        y='Clean_Name',
        palette=colors,
        edgecolor='black'
    )

    # Add the exact R2 values to the end of each bar for precision
    for i, bar in enumerate(bars.patches):
        width = bar.get_width()
        # Offset text slightly left if negative, right if positive
        x_offset = 0.005 if width > 0 else -0.015
        align = 'left' if width > 0 else 'right'
        
        plt.text(
            width + x_offset, 
            bar.get_y() + bar.get_height() / 2, 
            f'{width:.3f}', 
            va='center', 
            ha=align, 
            fontsize=11, 
            fontweight='bold'
        )

    # Draw a bold line at R2 = 0 to separate signal from noise
    plt.axvline(0, color='black', linewidth=1.5, linestyle='-')

    # Formatting
    plt.title('Predictive Viability of Blood Biomarkers from GPS Load\n(Strict Nested Leave-One-Subject-Out Validation)', 
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Unbiased $R^2$ Score (Higher is Better)', fontsize=12)
    plt.ylabel('Clinical Biomarker', fontsize=12)
    
    # Ensure the x-axis has a bit of padding so the text doesn't get cut off
    plt.xlim(min(df['Unbiased_R_Squared'].min() - 0.02, -0.05), df['Unbiased_R_Squared'].max() + 0.05)
    
    plt.tight_layout()

    # Save the output
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "top_5_markers_leaderboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[SUCCESS] Visual saved to: {plot_path}")