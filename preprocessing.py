import numpy as np
import pandas as pd


class preprocessing():
    """
    A class for preprocessing data for machine learning models.

    Methods
    -------
    train_valid(data: pd.DataFrame) -> tuple
        Preprocesses the input data and returns the training and validation sets.
    train(data: pd.DataFrame) -> tuple
        Preprocesses the input data and returns the training set.
    test(data: pd.DataFrame) -> tuple
        Preprocesses the input data and returns the test set.
    """

    def __init__(self) -> None:
        # Stores the minimum and maximum values of the training set for normalization
        self.data_min = None
        self.data_max = None
        pass

    # Preprocesses the input data and returns the training and validation sets
    def train_valid(self, data: pd.DataFrame) -> tuple:
        # Convert the time column to datetime format
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        # Convert the day of the year to sine and cosine values
        data['day_of_year'] = data['time'].dt.dayofyear

        data['day_of_year'] = (data['day_of_year']*np.pi)/180
        data['sin_time'] = np.sin(data['day_of_year'])
        data['cos_time'] = np.cos(data['day_of_year'])

        # delete the unnecessary columns
        date = data["time"]
        label = data['Label']
        data.drop(columns=['time', 'day_of_year', 'Label'], inplace=True)

        # Normalize the data
        data_min = data.min()
        data_max = data.max()

        data = (data - data_min) / (data_max - data_min)

        # Add the square of each feature to the data
        for column in data.columns.drop(['sin_time', 'cos_time']):
            data[column + '_pow'] = np.square(data[column])

        # Split the data into training and validation sets
        X_train = data[date.dt.year <= 2005]
        X_val = data[date.dt.year > 2005]
        y_train = label[date.dt.year <= 2005]
        y_val = label[date.dt.year > 2005]

        return X_train, y_train, X_val, y_val

    # Preprocesses the input data and returns the training set
    def train(self, data: pd.DataFrame) -> tuple:
        # Convert the time column to datetime format
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        # Convert the day of the year to sine and cosine values
        data['day_of_year'] = data['time'].dt.dayofyear

        data['day_of_year'] = (data['day_of_year']*np.pi)/180
        data['sin_time'] = np.sin(data['day_of_year'])
        data['cos_time'] = np.cos(data['day_of_year'])

        # delete the unnecessary columns
        y_train = data['Label']
        data.drop(columns=['time', 'day_of_year', 'Label'], inplace=True)

        # Normalize the data
        self.data_min = data.min()
        self.data_max = data.max()
        X_train = (data - self.data_min) / (self.data_max - self.data_min)

        # Add the square of each feature to the data
        for column in X_train.columns.drop(['sin_time', 'cos_time']):
            X_train[column + '_pow'] = np.square(X_train[column])

        return X_train, y_train

    def test(self, data: pd.DataFrame) -> tuple:
        # Convert the time column to datetime format
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        # Convert the day of the year to sine and cosine values
        data['day_of_year'] = data['time'].dt.dayofyear

        data['day_of_year'] = (data['day_of_year']*np.pi)/180
        data['sin_time'] = np.sin(data['day_of_year'])
        data['cos_time'] = np.cos(data['day_of_year'])

        # delete the unnecessary columns
        data.drop(columns=['time', 'day_of_year'], inplace=True)

        # Normalize the data
        X_test = (data - self.data_min) / (self.data_max - self.data_min)

        # Add the square of each feature to the data
        for column in X_test.columns.drop(['sin_time', 'cos_time']):
            X_test[column + '_pow'] = np.square(X_test[column])

        return X_test
