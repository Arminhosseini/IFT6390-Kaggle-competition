import numpy as np
import pandas as pd


class preprocessing():
    def __init__(self) -> None:
        self.data_min = None
        self.data_max = None
        pass

    def train_valid(self, data):
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        data['day_of_year'] = data['time'].dt.dayofyear

        data['day_of_year'] = (data['day_of_year']*np.pi)/180
        data['sin_time'] = np.sin(data['day_of_year'])
        data['cos_time'] = np.cos(data['day_of_year'])

        date = data["time"]
        label = data['Label']
        data.drop(columns=['time', 'day_of_year', 'Label'], inplace=True)

        data_min = data.min()
        data_max = data.max()

        data = (data - data_min) / (data_max - data_min)

        for column in data.columns.drop(['sin_time', 'cos_time']):
            data[column + '_pow'] = np.square(data[column])

        X_train = data[date.dt.year <= 2005]
        X_val = data[date.dt.year > 2005]
        y_train = label[date.dt.year <= 2005]
        y_val = label[date.dt.year > 2005]

        return X_train, y_train, X_val, y_val

    def train(self, data):
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        data['day_of_year'] = data['time'].dt.dayofyear

        data['day_of_year'] = (data['day_of_year']*np.pi)/180
        data['sin_time'] = np.sin(data['day_of_year'])
        data['cos_time'] = np.cos(data['day_of_year'])

        y_train = data['Label']
        data.drop(columns=['time', 'day_of_year', 'Label'], inplace=True)

        self.data_min = data.min()
        self.data_max = data.max()
        X_train = (data - self.data_min) / (self.data_max - self.data_min)

        for column in X_train.columns.drop(['sin_time', 'cos_time']):
            X_train[column + '_pow'] = np.square(X_train[column])

        return X_train, y_train

    def test(self, data):
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        data['day_of_year'] = data['time'].dt.dayofyear

        data['day_of_year'] = (data['day_of_year']*np.pi)/180
        data['sin_time'] = np.sin(data['day_of_year'])
        data['cos_time'] = np.cos(data['day_of_year'])

        data.drop(columns=['time', 'day_of_year'], inplace=True)

        X_test = (data - self.data_min) / (self.data_max - self.data_min)

        for column in X_test.columns.drop(['sin_time', 'cos_time']):
            X_test[column + '_pow'] = np.square(X_test[column])

        return X_test
