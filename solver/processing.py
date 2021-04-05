import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import tensorflow as tf
from tensorflow import keras


from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping


def smape(a, b):
    '''
    Calculates sMAPE
    :param a: actual values
    :param b: predicted values
    :return: sMAPE
    '''
    a = np.reshape(a, (-1,))
    b = np.reshape(b, (-1,))
    sm = np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()
    if np.isnan(sm):
        cond = np.abs(a) + np.abs(b)
        idx = np.argwhere(cond == 0)
        b = np.delete(b, idx)
        a = np.delete(a, idx)
        sm = np.mean(2.0 * np.abs(a - b) / (np.abs(a) + np.abs(b))).item()
    return sm


def read_datasets(turbine_link=None, weather_link=None):
    ''' Read passed links of dataframes. '''
    if weather_link is not None:
        weather_data = pd.read_csv(weather_link, encoding='utf-8')
        weather_data['Date_time'] = pd.to_datetime(weather_data['date_time'], utc=True, errors='coerce')
    if turbine_link is not None:
        turbine_data = pd.read_csv(turbine_link, sep='\t', encoding='utf-8')
        turbine_data['Date_time'] = pd.to_datetime(turbine_data['Date_time'], utc=True, errors='coerce')
    return weather_data, turbine_data


def fit_lstm(X, y, X_val, y_val, batch_size, nb_epochs, neurons, n_past):
    '''
    Fit LSTM model on the X, y data with the defined paramters
    :param X: training inputs (2D array)
    :param y: training outputs (1D array)
    :param X_val: validation inputs (2D array)
    :param y_val: validation outputs (1D array)
    :param batch_size: batch size for the LSTM training
    :param nb_epochs: number of epochs to train
    :param neurons: number of neurons in LSTM layers
    :param n_past: number of past samples for every observation
    :return: LSTM model
    '''
    # shape input data
    X = X.reshape(-1, n_past, int(X.shape[1] / n_past))
    X_val = X_val.reshape(-1, n_past, int(X_val.shape[1] / n_past))

    # define model
    model = Sequential()
    model.add(LSTM(neurons, activation='relu', input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
    model.add(LSTM(neurons, activation='relu', input_shape=(X.shape[1], X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    # fit model
    model.fit(
        X, y, epochs=nb_epochs, batch_size=batch_size,
        verbose=1, validation_data=(X_val, y_val))
    return model


def make_predictions(model, X, y, scaler, n_past):
    # shape input data
    X = X.reshape(-1, n_past, int(X.shape[1]/n_past))
    # make predictions
    y_predict = model.predict(X)
    # scale outputs back
    y_true = scaler.inverse_transform(y.reshape(-1,1)).flatten()
    y_predict = scaler.inverse_transform(y_predict.reshape(-1,1)).flatten()
    # evaluate predictions
    print("MAE: {} RMSE: {} sMAPE: {}".format(
        round(mean_absolute_error(y_true, y_predict), 3),
        round(np.sqrt(mean_squared_error(y_true, y_predict)),3),
        round(smape(y_true, y_predict),3)))
    return y_predict, y_true


def plot_predictions(y_true, y_predict, N_OUT=1, plot_samples=800, save=False):
    ''' Plot predictions and true values. '''
    fig = plt.figure(figsize=(14, 5))
    plt.plot(y_true[:plot_samples], label='observed')
    plt.plot(y_predict[:plot_samples], label='predicted')
    plt.title("Prediction for {} Hour Horizon".format(N_OUT))
    plt.xlabel("Hour")
    plt.ylabel("P_avg")
    fig.tight_layout()
    plt.legend()
    plt.show()
    if save:
        plt.savefig('predictions_horizon{}.pdf'.format(N_OUT), dpi=450)
