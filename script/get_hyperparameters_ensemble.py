import os
import json
import numpy as np


from sklearn.ensemble import BaggingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler


from solver import processing


feature_predict = 'P_avg'
features_train = ['P_avg']
TURBINE_ID = 'R80711'
# YEARS USED TO ESTIMATE HYPER-PARAMETERS
train_years = [2013, 2014, 2015, 2016]
# MOVING AVERAGE FOR TIME-SERIES
MA_CONSTANT = 3
# DEFINE METHOD OF INTEREST
method = 'DT' # 'SVR' or 'DT'
# FEATURE ENGINEERING PARAMETERS
N_OUT = 1
N_PAST = 48


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


if __name__ == '__main__':
    # define datasets
    DATA_DIR = os.path.join('../datasets/after_imputation', 'turbine_{}.csv'.format(TURBINE_ID))

    # read datasets
    dataset = processing.read_dataset(DATA_DIR)

    # define masks for training/validation and testing (will be used later)
    train_idx = dataset[dataset['Date_time'].dt.year.isin(train_years)].index

    # some stats:
    print("Number of duplicates: \t\t {}".format(len(dataset.index[dataset.index.duplicated()].unique())))
    print("Number of rows with nan: \t {}".format(np.count_nonzero(dataset.isnull())))

    # perform smoothing
    if feature_predict in features_train:
        dataset = processing.smooth(dataset, cols_to_smooth=features_train, ma_constant=MA_CONSTANT)
    else:
        dataset = processing.smooth(dataset, cols_to_smooth=features_train + [feature_predict], ma_constant=MA_CONSTANT)


    # split to training/validation/testing sets based on indices
    dataset_train = dataset[dataset.index.isin(train_idx)].copy()

    # define target mask for features
    target_idx = np.where(dataset_train.columns == feature_predict)[0][0]
    target_mask = np.zeros((dataset_train.shape[1])).astype(bool)
    target_mask[target_idx] = True
    # define input mask for features
    input_idx = [np.where(dataset_train.columns == feat_col)[0][0] for feat_col in features_train]
    input_mask = np.zeros((dataset_train.shape[1])).astype(bool)
    input_mask[input_idx] = True

    # Define scaler and fit only on training data
    scaler_output = MinMaxScaler()
    y_train = scaler_output.fit_transform(dataset_train.iloc[:, target_mask])
    # Define scaler and fit only on training data
    scaler_input = MinMaxScaler()
    X_train = scaler_input.fit_transform(dataset_train.iloc[:, input_mask])

    # Make small tests
    assert X_train.shape[0] == y_train.shape[0]

    # make supervised learning problem
    X_train_sup = processing.series_to_supervised(X_train, n_in=N_PAST, n_out=0, dropnan=True)
    y_train_sup = processing.series_to_supervised(y_train, n_in=N_PAST, n_out=N_OUT, dropnan=True).iloc[:, -1]

    # Align X with y
    X_train_sup = X_train_sup[X_train_sup.index.isin(y_train_sup.index)]

    # Set to numpy arrays
    X_train = X_train_sup.values
    y_train = y_train_sup.values

    # Define time series split
    tscv = TimeSeriesSplit(n_splits=3)
    print("Supervised shape: ", X_train.shape, y_train.shape)

    if method == 'SVR':
        print(method)
        # Run Grid-cross validation
        param = {
            'n_estimators': [4,16,64],
            'max_samples': [100, 500],
            'base_estimator__kernel': ['poly', 'rbf'],
            'base_estimator__epsilon':[0.1, 0.01, 0.001],
            'base_estimator__C': [1, 1e3, 1e6],
            'base_estimator__gamma': [1e-1, 1e-3, 1e-6],
            'base_estimator__degree':[5, 8]
                  }

        reg = BaggingRegressor(
            base_estimator=SVR(), verbose=3, n_jobs=2,
            bootstrap=True, oob_score=True)
        print(reg.get_params().keys())
        best_reg = GridSearchCV(estimator=reg,
                           param_grid=param, cv=tscv, n_jobs=2, verbose=2, scoring='neg_mean_squared_error')
        best_reg.fit(X_train, y_train)

    elif method == 'DT':
        print(method)
        # Run Grid-cross validation
        param = {
            'n_estimators':[10, 50, 200, 500],
            'max_samples':[100, 500, 1_000, 2_500],
            'base_estimator__max_features': ['auto', 'sqrt', 'log2'],
            'base_estimator__max_depth': [10, 50, 100, None],
            'base_estimator__min_samples_split': [2, 0.1, 0.25, 0.5]
                 }
        reg = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                               verbose=1, n_jobs=2,
                               bootstrap=True, oob_score=True)
        print(reg.get_params().keys())
        best_reg = GridSearchCV(estimator=reg,
                                param_grid=param, cv=tscv, n_jobs=2, verbose=2)
        best_reg.fit(X_train, y_train)

    else:
        raise Exception('Wrong method name!')

    print("Best hyper-parameters: \n\t {}".format(best_reg.best_params_))
    print("Best score: \n\t {}".format(best_reg.best_score_))
    print("---------  ------  ------  ---------- \n")
    print("All results: \n\t {}".format(best_reg.cv_results_))
    print("---------  ------  ------  ---------- \n")

    # store results to files
    dumped = json.dumps(best_reg.cv_results_, cls=NumpyEncoder)
    with open("gridSearch_{}_all_results.json".format(method), 'w') as fp:
        json.dump(dumped, fp)
    dumped = json.dumps(best_reg.best_score_, cls=NumpyEncoder)
    with open("gridSearch_{}_best_score.json".format(method), 'w') as fp:
        json.dump(dumped, fp)
    dumped = json.dumps(best_reg.best_params_, cls=NumpyEncoder)
    with open("gridSearch_{}_best_parameters.json".format(method), 'w') as fp:
        json.dump(dumped, fp)