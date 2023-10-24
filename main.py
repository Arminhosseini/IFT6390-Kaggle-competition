import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import copy
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from logistic_regression import *

data = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

data['time'] = pd.to_datetime(
    data['time'], format='%Y%m%d')

data['year'] = data["time"].dt.year
data['month'] = data["time"].dt.month
data['day'] = data["time"].dt.day

data = data.drop(columns='time')

train_data = data[data['year'] > 2005]
valid_data = data[data['year'] <= 2005]

X_train = train_data.loc[:, train_data.columns != 'Label']
y_train = train_data.loc[:, 'Label']
X_val = valid_data.loc[:, train_data.columns != 'Label']
y_val = valid_data.loc[:, 'Label']

X_train = (X_train - X_train.mean()) / X_train.std()
X_val = (X_val - X_val.mean()) / X_val.std()


lr = logistic_regression()
lr.fit(X_train, y_train, 2000, 0.001)
pred = lr.predict(X_val)
predictions = lr.predict(X_val)
lr.accuracy(y_val, predictions)

test = pd.read_csv('Dataset/test.csv')
df = pd.DataFrame(data={'SNo': test['SNo']})
test['time'] = pd.to_datetime(
    test['time'], format='%Y%m%d')
test['year'] = test["time"].dt.year
test['month'] = test["time"].dt.month
test['day'] = test["time"].dt.day
test = test.drop(columns='time')
test = (test - test.mean()) / test.std()

test_predictions = lr.predict(test)

df['Label'] = test_predictions
df.to_csv('submission.csv', index=False)
