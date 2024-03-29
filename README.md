# How to forecast wind-generated power?
 
The directory contains main steps for wind power forecasting applications and is a part of a Master thesis at KU Leuven, 2022. 
The dataset of interest is *La Haute Borne* wind farm in France. The repository details all preprocessing and forecasting steps - data analysis, imputation, feature engineering and modeling. 
The notebooks are used to compare various forecasting methodologies - Persistence, ARIMA, LSTM or ensembles (SVR or Decision trees).

# Directory content:
* solver - directory containing all required functions used in the notebooks
* notebooks - directory containing all workflow for wind power forecasting
* script - directory with external files important for forecasting

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
script
   |-- get_hyperparameters_LSTM.py
   |-- get_hyperparameters_ensemble.py
solver
   |-- arima.py
   |-- ensembles.py
   |-- lstm.py
   |-- persistence.py
   |-- processing.py
wind_power37.yml
.gitignore
LICENSE
README.md
setup.py
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



