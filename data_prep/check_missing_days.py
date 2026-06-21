import pandas as pd
import argparse
from pathlib import Path

def analyze_missingness(input_csv, out_dir):
    df = pd.read_csv(input_csv)
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    
    # Calculate unique athletes and dates
    total_athletes = df["Player Name"].nunique()
    
    # Pivot to see which athletes have data on which dates
    # A value of 1 means data exists, NaN means missing
    pivot = df.pivot_table(index="Date", columns="Player Name", aggfunc='size')
    
    # Calculate % missing per date
    # (Total athletes - count of athletes with data) / Total athletes
    missing_pct = (pivot.isnull().sum(axis=1) / total_athletes) * 100
    missing_count = pivot.isnull().sum(axis=1)
    
    # Create the report DataFrame
    report = pd.DataFrame({
        "Missing_Count": missing_count,
        "Missing_Percentage": missing_pct.round(2)
    }).sort_index()
    
    # Save to CSV
    report_path = out_dir / "missing_data_report.csv"
    report.to_csv(report_path)
    
    print(f"[INFO] Report generated: {report_path}")
    print(report.head(10))
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="/Users/melisatbinek/Desktop/Data/Hockey_SK/data/GPS_File_synt.csv")
    parser.add_argument("--output", default="/Users/melisatbinek/Desktop/Data/Hockey_SK/Analysis_CoS_SK/output", help="/Users/melisatbinek/Desktop/Data/Hockey_SK/Analysis_CoS_SK/output")
    args = parser.parse_args()
    
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    analyze_missingness(args.input, out_dir)

if __name__ == "__main__":
    main()
    