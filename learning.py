import pandas as pd
from logistic_regression import *


class Learning_data():
    def __init__(self, train_data: pd.DataFrame, test_data: pd.DataFrame, epoch_num: int, learning_rate: float, alpha: float) -> None:
        self.train_data = train_data
        self.test_data = test_data
        self.epoch_num = epoch_num
        self.learning_rate = learning_rate
        self.alpha = alpha
        return

    def test_preprocessing(self, test) -> pd.DataFrame:
        test['time'] = pd.to_datetime(
            test['time'], format='%Y%m%d')
        test['year'] = test["time"].dt.year
        test['month'] = test["time"].dt.month
        test['day'] = test["time"].dt.day
        test = test.drop(columns='time')
        test = (test - self.train_min) / (self.train_max - self.train_min)
        return test

    def pre_proc_to_submit(self, data: pd.DataFrame) -> tuple:
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        data['year'] = data["time"].dt.year
        data['month'] = data["time"].dt.month
        data['day'] = data["time"].dt.day

        data = data.drop(columns='time')

        X_train = data.loc[:, data.columns != 'Label']
        y_train = data.loc[:, 'Label']

        x_train_min = X_train.min()
        x_train_max = X_train.max()
        X_train = (X_train - x_train_min) / (x_train_max - x_train_min)

        self.train_min = x_train_min
        self.train_max = x_train_max

        return X_train, y_train

    def pre_proc_valid(self, data: pd.DataFrame) -> tuple:
        data['time'] = pd.to_datetime(
            data['time'], format='%Y%m%d')

        data['year'] = data["time"].dt.year
        data['month'] = data["time"].dt.month
        data['day'] = data["time"].dt.day

        data = data.drop(columns='time')

        train_data = data[data['year'] <= 2005]
        valid_data = data[data['year'] > 2005]

        X_train = train_data.loc[:, train_data.columns != 'Label']
        y_train = train_data.loc[:, 'Label']
        X_val = valid_data.loc[:, train_data.columns != 'Label']
        y_val = valid_data.loc[:, 'Label']

        x_train_min = X_train.min()
        x_train_max = X_train.max()
        X_train = (X_train - x_train_min) / (x_train_max - x_train_min)
        X_val = (X_val - x_train_min) / (x_train_max - x_train_min)

        return X_train, y_train, X_val, y_val

    def learn_data(self, type: str):
        lr = logistic_regression()
        if type == 'test':
            X_train, y_train, X_val, y_val = self.pre_proc_valid(
                self.train_data)
            lr.fit(X_train, y_train, self.epoch_num,
                   self.learning_rate, self.alpha)
            train_pred = lr.predict(X_train)
            train_acc = lr.accuracy(y_train, train_pred)
            print(f'Training accuracy : {train_acc}')
            val_pred = lr.predict(X_val)
            val_acc = lr.accuracy(y_val, val_pred)
            print(f'Validation accuracy : {val_acc}')

        if type == 'submit':
            X_train, y_train = self.pre_proc_to_submit(self.train_data)
            lr.fit(X_train, y_train, self.epoch_num,
                   self.learning_rate, self.alpha)
            X_test = self.test_preprocessing(self.test_data)
            X_test["Label"] = lr.predict(X_test)
            X_test["Label"].to_csv('submission.csv', header=True)
