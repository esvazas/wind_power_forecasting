import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from solver.processing import smape


def eval_predictions(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    sm = smape(y_test.reshape(-1, 1), y_predict.reshape(-1,1))
    return mae, rmse, sm


def ARIMAForecasting(data, best_pdq, step):
    ''' ARIMA model fit and recursive forecasting. '''
    model = ARIMA(data, order=best_pdq)
    model_fit = model.fit()
    return model_fit.forecast(steps=step)[-1]


def make_predictions(y_train, y_test, N_OUT, MAX_WINDOW=1200, best_pdq=(4,1,1)):
    ''' Moving window model evaluation = at each sample ARIMA on MAX_WINDOW data is built. '''

    # define lists
    y_predict = np.empty(shape=(y_test.shape[0] - N_OUT))
    y_true = np.empty(shape=(y_test.shape[0] - N_OUT))

    # loop through every sample
    data = y_train
    for t in tqdm(range(len(y_test) - N_OUT)):
        # get predictions
        prediction = ARIMAForecasting(data, best_pdq, step=N_OUT)
        # store prediction and data used to train
        y_predict[t] = prediction
        # compare with value after horizon
        y_true[t] = y_test[t + N_OUT - 1]
        # add subsequent sample
        data = np.append(data, y_test[t])

        if data.shape[0] > MAX_WINDOW:
            data = np.delete(data, 0, axis=0)
    return y_predict, y_true


def plot_predictions(y_true, y_predict, N_OUT=1, plot_samples=800, save=False):
    ''' Plot predictions and true values. '''
    # define colors
    color = sns.color_palette("rocket", 5)

    # make plot
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