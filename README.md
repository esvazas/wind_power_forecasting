# How to forecast wind-generated power by a wind turbine?
 
Directory contains important steps for wind power forecasting applications and is a part of a Master thesis at KU Leuven, 2021. 

** Note: the code is continiously updated. ***

# Directory content:
* datasets - directory containing generated datasets by each notebook
* solver - directory containing all required functions used in notebooks
* notebooks - directory containing main notebooks

# Directory tree:



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



