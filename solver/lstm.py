import tensorflow as tf
from tensorflow import keras
from solver.processing import smape

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from keras.callbacks import EarlyStopping, TensorBoard


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
    history = model.fit(
        X, y, epochs=nb_epochs, batch_size=batch_size,
        verbose=1, validation_data=(X_val, y_val),
        callbacks=[
            TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
            EarlyStopping(monitor='val_loss', patience=10, mode='auto')
        ])
    return model, history


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