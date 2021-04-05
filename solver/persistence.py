import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from solver.processing import smape


def eval_predictions(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    sm = smape(y_test.reshape(-1, 1), y_predict.reshape(-1,1))
    return mae, rmse, sm


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
    mae, rmse, sm = eval_predictions(actual, predicted)
    return mae, rmse, sm


# sample persistence model
def sample_persistence(history, fh):
    # get the data for the prior sample
    return history[int(-1 * fh)]


# daily persistence model
def daily_persistence(history, fh):
    # get the data for the prior sample
    return history[int(-24 * fh)]


# weekly persistence model
def weekly_persistence(history, fh):
    # get sample from the prior week
    return history[int(-(24 *7) * fh)]


# one year ago persistence model
def year_ago_persistence(history, fh):
    if fh == 1:
        return history[-8784]
    elif fh == 2:
        return history[-8784 - 8760]
    elif fh == 3:
        return history[-8784 - 8760 - 8760]
    elif fh == 4:
        return history[-8784 - 8760 - 8760 - 8760]
    else:
        return None


# evaluate a single model
def evaluate_model(model_func, train, test, fh):
    # history is a list of weekly data
    history = [x for x in train]
    # walk-forward validation over each week
    y_true = np.zeros(shape=(test.shape[0]))
    predictions = np.zeros(shape=(test.shape[0]))
    for i in range(len(test)):
        # predict the week
        yhat = model_func(history, fh)
        # store the predictions
        predictions[i] = yhat
        y_true[i] = test[i]
        # get real observation
        # and add to history for predicting the next week
        history.append(test[i])

    # Remove nan samples
    nan_pred = np.argwhere(np.isnan(predictions))
    nan_y_true = np.argwhere(np.isnan(y_true))
    nan_idx = np.union1d(nan_pred, nan_y_true)
    pred = np.delete(predictions, nan_idx)
    y_true = np.delete(y_true, nan_idx)

    # evaluate forecasts
    mae, mse, sm = evaluate_forecasts(y_true, pred)
    return predictions, mae, mse, sm


def plot_predictions(y_true, y_predict, N_OUT=1, plot_samples=800, save=False):
    ''' Plot predictions and true values. '''
    color = sns.color_palette("rocket", 5)

    plt.figure(figsize=(14, 5))
    y_predict.iloc[:plot_samples].plot(color=color[-2], label="Predictions")
    y_true.iloc[:plot_samples].plot(label='True values')
    plt.grid(which='both')
    plt.ylabel('Power (W)')
    plt.title("{}-steps ahead Naive predictions".format(N_OUT))
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig("Naive_{}step_predictions_1H.pdf".format(N_OUT), dpi=450)
    plt.show()