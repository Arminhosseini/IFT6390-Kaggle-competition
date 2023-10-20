import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import copy
from sklearn.linear_model import LogisticRegression
from logistic_regression import *

train_data = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")

msk = np.random.rand(len(train_data)) < 0.8
train = train_data[msk]
validation = train_data[~msk]

X_train = train.iloc[:, :-1]
Y_train = train.iloc[:, -1]
X_val = validation.iloc[:, :-1]
Y_val = validation.iloc[:, -1]

y_train = copy.deepcopy(Y_train)
y_val = copy.deepcopy(Y_val)

y_train = y_train.replace({1: 3, 2: 3})
y_val = y_val.replace({1: 3, 2: 3})

lr = logistic_regression()
lr.fit(X_train, y_train, 150, 0.001)
pred = lr.predict(X_val)
