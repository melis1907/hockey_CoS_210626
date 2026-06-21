import subprocess
import time
import sys


PIPELINE_SCRIPTS = [
    "data_prep/data_prep_main.py",  # The script that processes GPS, Blood, and Merges them with HAT
    "model/run_nested_pipeline.py"  # The script that runs the nested LOSO and plots the drivers
]

def run_script(script_name):
    """Runs a python script and streams its output to the console."""
    print(f"\n{'='*75}")
    print(f"🚀 STARTING STAGE: {script_name}")
    print(f"{'='*75}\n")
    
    start_time = time.time()
    
    try:
        # Run the script and stream the output in real-time
        process = subprocess.Popen(
            [sys.executable, script_name], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Print the output line by line as it generates
        for line in process.stdout:
            print(line, end='')
            
        process.wait()
        
        # Check if the script crashed
        if process.returncode != 0:
            print(f"\n[FATAL ERROR] {script_name} failed with exit code {process.returncode}.")
            sys.exit(1) # Stop the entire master pipeline
            
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find '{script_name}'. Please check the filename.")
        sys.exit(1)
        
    elapsed_time = time.time() - start_time
    mins, secs = divmod(int(elapsed_time), 60)
    print(f"\n✅ [STAGE COMPLETE] {script_name} finished in {mins}m {secs}s.")

# ==========================================
# MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "#"*75)
    print(" INITIATING MASTER HOCKEY PIPELINE")
    print("#"*75)
    
    total_start_time = time.time()
    
    for script in PIPELINE_SCRIPTS:
        run_script(script)
        
    total_elapsed = time.time() - total_start_time
    t_mins, t_secs = divmod(int(total_elapsed), 60)
    
    print("\n" + "#"*75)
    print(f" 🎉 FULL PIPELINE EXECUTED SUCCESSFULLY IN {t_mins}m {t_secs}s")
    print(" All data processed, models trained, and visuals generated.")
    print("#"*75 + "\n")