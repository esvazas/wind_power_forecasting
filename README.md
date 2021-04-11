# How to forecast wind-generated power?
 
The directory contains main steps for wind power forecasting applications and is a part of a Master thesis at KU Leuven, 2021. 
All pre-forecasting required steps are covered - data analysis, imputation and feature engineering. 
The notebooks can be used for comparison of various forecasting methodologies - Persistence, ARIMA, LSTM or ensembles (SVR or Decision trees).

** Note: Applications require examples that are not provided as the repository is continuously updated. Datasets will be provided later. **

# Directory content:
* datasets - directory containing generated datasets by each notebook
* solver - directory containing all required functions used in the notebooks
* notebooks - directory containing all workflow for wind power forecasting

# Directory tree:
```bash
notebooks
   |-- 1_data_analysis.ipynb
   |-- 2_outliers_removal.ipynb
   |-- 3_feature_engineering.ipynb
   |-- 4_imputation.ipynb
   |-- 5_ARIMA_forecasting.ipynb
   |-- 5_COND-LSTM_forecasting.ipynb
   |-- 5_LSTM_forecasting.ipynb
   |-- 5_Persistence_forecasting.ipynb
   |-- 5_ensembles_forecasting.ipynb
solver
   |-- arima.py
   |-- ensembles.py
   |-- get_hyperparameters.py
   |-- lstm.py
   |-- persistence.py
   |-- processing.py
.gitignore
LICENSE
README.md
setup.py
wind_power37.yml
```

# Usage
#### Create Anaconda environment
Use the conda environment file **wind_power37.yml** to install the required **wind_power37** environment and its modules.

```bash
conda env create -f wind_power37.yml
```
Activate `wind_power37` conda environment:
```bash
conda activate wind_power37
```
#### Create function package
From the root directory, create **solver** package which can be accessed from all the notebooks:
```bash
conda develop .
```



