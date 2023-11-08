
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from preprocessing import preprocessing
import pandas as pd


preprocessing = preprocessing()
data = pd.read_csv('Dataset/train.csv', index_col=0)
# Load and preprocess dataset
X_train, y_train, X_val, y_val = preprocessing.train_valid(data)

kernel = ['linear', 'rbf', 'poly']
C = [1, 10]
gamma = [0.1, 1, 10]

for k in kernel:
    for c in C:
        for g in gamma:
            svc = svm.SVC(kernel=k, C=c, gamma=g)
            svc.fit(X_train, y_train)
            print(
                f"Train accuracy for kernel {k}, C {c}, gamma {g}: {svc.score(X_train, y_train)}")
            print(
                f"Validation accuracy for kernel {k}, C {c}, gamma {g}: {svc.score(X_val, y_val)}")
