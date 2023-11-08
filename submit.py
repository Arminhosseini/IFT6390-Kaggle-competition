import pandas as pd
import numpy as np
from IPython.display import display, clear_output
from logistic_regression import logistic_regression
from preprocessing import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Dataset/train.csv', index_col=0)
test_data = pd.read_csv('Dataset/test.csv', index_col=0)

prepro = preprocessing()
X_train, y_train = prepro.train(data)
X_test = prepro.test(test_data)

lr = logistic_regression()
loss = lr.fit(X_train, y_train, 2500, 0.10, 0.99, 0.00001, 0.98, 0, 0)
train_pred = lr.predict(X_train)
train_acc = lr.accuracy(y_train, train_pred)
print(f'{loss}\n{train_acc}')

X_test["Label"] = lr.predict(X_test)
X_test["Label"].to_csv('submission.csv', header=True)
