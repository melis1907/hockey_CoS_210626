## Prerequisites & Installation

To avoid library conflicts, please recreate the project's exact Conda environment. 

### 1. Create the Environment File
Create a file named `env_hockey.yml` in the root of this repository and paste the following configuration into it:

```yaml
name: env_hockey
channels:
  - conda-forge
  - defaults
dependencies:
  # Core Python
  - python=3.14.6
  
  # Data Engineering & Manipulation
  - pandas=3.0.3
  - numpy=2.4.4
  
  # Machine Learning & Sequence Modeling
  - scikit-learn=1.9.0       # For Anomaly Detection (Isolation Forests)
  - hmmlearn=0.3.3           # For Gaussian Hidden Markov Models (HMM)
  - xgboost=3.2.0            # For Gradient Boosting Machines (GBM)
  
  # Statistical Validation
  - statsmodels=0.14.6       # For Linear Mixed-Effects Models (LMM)
  - scipy=1.18.0
  
  # Visualization (Optional but recommended for exploratory analysis)
  - matplotlib=3.11.0
  - seaborn=0.13.2
  
  # Notebook Environment
  - jupyter=1.1.1

  # Progress Bars
  - tqdm=4.68.3
