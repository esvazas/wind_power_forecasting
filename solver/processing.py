import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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