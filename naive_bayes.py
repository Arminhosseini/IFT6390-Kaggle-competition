
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from preprocessing import preprocessing
import numpy as np


preprocessing = preprocessing()
data = pd.read_csv('Dataset/train.csv', index_col=0)
# Load and preprocess dataset
X_train, y_train, X_val, y_val = preprocessing.train_valid(data)

alpha_values = np.logspace(1e-6, -9, num=1000)

for alpha in alpha_values:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)
    print(
        f"Train accuracy for alpha {alpha}: {mnb.score(X_train, y_train)}")
    print(
        f"Validation accuracy for alpha {alpha}: {mnb.score(X_val, y_val)}")
