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


def get_predictions(model, x_test, fh):
    for i in range(0, fh):
        y_predict = model.predict(x_test)
        x_test = np.roll(x_test, -1, axis=1)
        x_test[:,-1] = y_predict
    return y_predict


def get_predictions_for_weights(estimators, weights, x_test, fh):
    y_predict = np.zeros((x_test.shape[0]))
    for est, w in zip(estimators, weights):
        y_predict = y_predict + get_predictions(est, x_test, fh) * w
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


def make_predictions(reg_estimators, weights, scaler, X, y, fh=1):
    ''' Make one-step forecasts for the given ensembles'''
    predictions = list()
    for i in tqdm(range(len(y))):
        # define input
        X_input = X[i,:]
        # make one-step forecast
        yhat = get_predictions_for_weights(reg_estimators, weights, X_input.reshape(1,-1), fh)
        # invert scaling
        yhat = scaler.inverse_transform(yhat.reshape(-1,1))
        # store forecast
        predictions.append(yhat)
    y_predict = np.array(predictions).flatten()
    print(y_predict.shape, y.shape)
    print("MAE: {} RMSE: {} sMAPE: {}".format(
        round(mean_absolute_error(y, y_predict), 3),
        round(np.sqrt(mean_squared_error(y, y_predict)), 3),
        round(smape(y, y_predict), 3)))
    return y_predict


def train_and_predict_heterogeneous_ensemble(train, test, s, n, n_out, n_past, target_idx, scaler, raw_target):
    print("________------ TRAIN AND PREDICT WITH HETEROGENEOUS ENSEMBLE ------ ________")
    s1, s2 = s
    n1, n2 = n
    base_reg = DecisionTreeRegressor(min_samples_split=2, max_depth=50)
    reg_estimators1, weights1 = train_homogeneous_ensemble(base_reg, train, s1, n1, target_idx)
    y_predict = make_predictions(reg_estimators1, weights1, scaler, test, raw_target, target_idx, n_past)
    print("DT HOMO1 ---")
    print(y_predict.shape, raw_target.shape)
    true_values = raw_target.iloc[-test.shape[0]:]
    rmse, rmae, sm = eval_predictions(true_values, y_predict)

    base_reg = SVR(C=10000, epsilon=0.01, gamma=0.001, kernel='rbf')
    reg_estimators2, weights2 = train_homogeneous_ensemble(base_reg, train, s2, n2, target_idx)
    y_predict = make_predictions(reg_estimators2, weights2, scaler, test, raw_target, target_idx, n_past)
    print("svr HOMO1 ---")
    true_values = raw_target.iloc[-test.shape[0]:]
    rmse, rmae, sm = eval_predictions(true_values, y_predict)

    reg_weights = np.concatenate([weights1, weights2])
    y_predict = np.zeros((test.shape[0]))
    for est, w in zip(reg_estimators1 + reg_estimators2, reg_weights):
        pred = make_predictions(
            [est], [w], scaler, test, raw_target, target_idx, n_past, n_out) * w
        y_predict = y_predict + pred.flatten()
    y_predict = y_predict / np.sum(reg_weights)
    print("svr&DT Hetero ---")
    true_values = raw_target.iloc[-test.shape[0]:]
    rmse, rmae, sm = eval_predictions(true_values, y_predict)
    return y_predict