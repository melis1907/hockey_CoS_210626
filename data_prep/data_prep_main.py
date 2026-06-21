
from processed.gps import process_gps_data
from processed.blood import prepare_blood_targets

from merge import merge_clinical_and_all_tracking_data

#The file paths:
#Make sure this matches your local directory structure. 
#You may need to adjust these paths based on where you store your data files.
gps_input_file = '/Users/melisatbinek/Desktop/Data/Hockey_SK/Analysis_CoS_SK/data/GPS_File_synt.csv'
gps_meta_out = 'processed_data/gps_Meta_Data.csv'
gps_engineered_out = 'processed_data/engineered_gps_features.csv'

# Blood Paths
blood_input_file = '/Users/melisatbinek/Desktop/Data/Hockey_SK/Analysis_CoS_SK/data/hockey_blood_measurments_synthetic.csv'
blood_deltas_out = 'processed_data/blood_deltas.csv'

# ==========================================
# Run Pipelines
# ==========================================
if __name__ == "__main__":
    print("Starting Data Processing Pipeline...")
    
    # 1. Process GPS Data
    meta_df, engineered_df = process_gps_data(
        input_path=gps_input_file,
        meta_output_path=gps_meta_out,
        engineered_output_path=gps_engineered_out
    )
    
    # 2. Process Blood Data
    raw_blood, blood_targets = prepare_blood_targets(
        csv_path=blood_input_file,
        output_path=blood_deltas_out
    )
    
    # 3. Merge Clinical and Tracking Data
    merged_df = merge_clinical_and_all_tracking_data(
        blood_csv = blood_deltas_out,
        gps_ewma_csv = gps_engineered_out,
        hat_csv = "data/hockey_hat_synthetic.csv",
        output_path = "processed_data/final_ewma_ml_dataset.csv"
    )
    
    print("\nPipeline Complete!")
