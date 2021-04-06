import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error

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


def read_dataset(data_link=None):
    ''' Read dataset from a passed link. '''
    if data_link is not None:
        data = pd.read_csv(data_link, encoding='utf-8', sep='\t')
        data['Date_time'] = pd.to_datetime(data['Date_time'], utc=True, errors='coerce')
        return data


def read_datasets(turbine_link=None, weather_link=None):
    ''' Read passed links of DataFrames. '''
    if weather_link is not None:
        weather_data = pd.read_csv(weather_link, encoding='utf-8')
        weather_data['Date_time'] = pd.to_datetime(weather_data['date_time'], utc=True, errors='coerce')
    if turbine_link is not None:
        turbine_data = pd.read_csv(turbine_link, sep='\t', encoding='utf-8')
        turbine_data['Date_time'] = pd.to_datetime(turbine_data['Date_time'], utc=True, errors='coerce')
    return weather_data, turbine_data


def smooth(df, cols_to_smooth, ma_constant=3):
    df[cols_to_smooth] = df[cols_to_smooth].rolling(ma_constant).mean()
    return df.dropna()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
        data: Sequence of observations as a list or NumPy array.
        n_in: Number of lag observations as input (X).
        n_out: Number of observations as output (y).
        dropnan: Boolean whether or not to drop rows with NaN values.
    Returns:
        Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def eval_predictions(y_test, y_predict):
    ''' Evaluate model performance. '''
    mae = mean_absolute_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    sm = smape(y_test.reshape(-1, 1), y_predict.reshape(-1,1))
    return mae, rmse, sm


def plot_series_predictions(y_true, y_predict, N_OUT=1, method='Naive', plot_samples=800, save=False):
    ''' Plot predictions and true values for pandas series where index is time. '''
    color = sns.color_palette("rocket", 5)

    plt.figure(figsize=(14, 5))
    y_predict.iloc[:plot_samples].plot(color=color[-2], label="Predictions")
    y_true.iloc[:plot_samples].plot(label='True values')
    plt.grid(which='both')
    plt.ylabel('Power (W)')
    plt.title("{}-steps ahead {} predictions".format(N_OUT, method))
    plt.tight_layout()
    plt.legend()
    if save:
        plt.savefig("{}_{}step_predictions_1H.pdf".format(method, N_OUT), dpi=450)
    plt.show()


def plot_predictions(y_true, y_predict, N_OUT=1, plot_samples=400, save=False):
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


def plot_history(history, save=False):
    if 'val_loss' in history.history.keys():
        fig = plt.figure(figsize=(12, 5))
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.grid()
        plt.tight_layout()
        plt.legend()
        if save:
            plt.savefig('model_loss.pdf', dpi=450)
        plt.show()
