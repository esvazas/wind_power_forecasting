import numpy as np

from tqdm import tqdm
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error

from solver.processing import eval_predictions, smape


def invert_scale(scaler, X, value):
    '''
    Perform inverse scaling for a forecasted value.
    Param: scaler - scaler object
    Param: X - input row
    Param: value - forecasted value
    Return: unscaled value
    '''
    new_row = [*X, *value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -len(value):]


def inverse_difference(history, yhat, interval):
    '''
    Invert differenced values.
    Param: history - time series to be predicted
    Param: yhat - predicted value
    Param: interval - difference interval
    Return: inverted difference value
    '''
    return yhat + history.iloc[-interval].values


def get_predictions(model, x_test):
    y_predict = model.predict(x_test)
    return y_predict


def get_predictions_for_weights(estimators, weights, x_test):
    y_predict = np.zeros((x_test.shape[0]))
    for est, w in zip(estimators, weights):
        y_predict = y_predict + get_predictions(est, x_test) * w
    return y_predict / np.sum(weights)


def train_homogeneous_ensemble(base_reg, X, y, s, n):
    reg = BaggingRegressor(base_estimator=base_reg, n_estimators=n,
                           max_samples=s, max_features=1.0,
                           bootstrap=True, oob_score=True, n_jobs=1)
    reg.fit(X, y)
    weights = np.zeros((n))
    for idx, est in tqdm(enumerate(reg.estimators_)):
        # Get oob samples
        mask = np.ones((X.shape[0]), np.bool)
        mask[reg.estimators_samples_[idx]] = 0
        x_val, y_val = X[mask, :], y[mask]
        weights[idx] = 1 / mean_squared_error(est.predict(x_val[:,:]), y_val[:])
    return reg.estimators_, weights


def print_evaluations(y_true, y_predict):
    ''' Compute RMSE, MAE and sMAPE for predictions. '''
    print("MAE: {} RMSE: {} sMAPE: {}".format(
        round(mean_absolute_error(y_true, y_predict), 3),
        round(np.sqrt(mean_squared_error(y_true, y_predict)), 3),
        round(smape(y_true, y_predict), 3)))


def make_predictions(reg_estimators, weights, scaler, X, y, print_eval=True):
    ''' Make one-step forecasts for the given ensembles'''
    predictions = list()
    for i in tqdm(range(X.shape[0]), disable=~print_eval):
        # define input
        X_input = X[i,:]
        # make one-step forecast
        yhat = get_predictions_for_weights(reg_estimators, weights, X_input.reshape(1,-1))
        # invert scaling
        yhat = scaler.inverse_transform(yhat.reshape(-1,1))
        # store forecast
        predictions.append(yhat)
    y_predict = np.array(predictions).flatten()
    # print predictions
    if print_eval:
        # rescale true values back
        y_true = scaler.inverse_transform(y.reshape(-1, 1))
        print("MAE: {} RMSE: {} sMAPE: {}".format(
            round(mean_absolute_error(y_true, y_predict), 3),
            round(np.sqrt(mean_squared_error(y_true, y_predict)), 3),
            round(smape(y_true, y_predict), 3)))
    return y_predict


def train_and_predict_heterogeneous_ensemble(
        X_train, y_train, X_test, y_test, s, n, base_reg1, base_reg2, scaler):
    '''
    Function to train two homogeneous ensembles base_reg1 and base_reg2 and combine their
    predictions as homogeneous ensembles.
    '''
    print("------ TRAIN AND PREDICT WITH HETEROGENEOUS ENSEMBLE ------")
    s1, s2 = s
    n1, n2 = n
    # build FIRST ensemble
    print("--- Case #1: HOMOGENEOUS ---")
    reg_estimators1, weights1 = train_homogeneous_ensemble(base_reg1, X_train, y_train, s1, n1)
    _ = make_predictions(reg_estimators1, weights1, scaler, X_test, y_test)

    # build SECOND ensemble
    print("--- Case #2: HOMOGENEOUS ---")
    reg_estimators2, weights2 = train_homogeneous_ensemble(base_reg2, X_train, y_train, s2, n2)
    _ = make_predictions(reg_estimators2, weights2, scaler, X_test, y_test)

    # build COMBINED ensemble with weighted voting
    reg_weights = np.concatenate([weights1, weights2])
    y_predict = np.zeros((X_test.shape[0]))
    for est, w in tqdm(zip(reg_estimators1 + reg_estimators2, reg_weights)):
        pred = make_predictions(
            [est], [w], scaler, X_test, y_test, print_eval=False) * w
        y_predict = y_predict + pred.flatten()
    y_predict = y_predict / np.sum(reg_weights)

    # scale outputs for evaluation
    y_true = scaler.inverse_transform(y_test.reshape(-1,1))
    print("--- Case #3: HETEROGENEOUS ---")
    print_evaluations(y_true, y_predict)

    return y_predict