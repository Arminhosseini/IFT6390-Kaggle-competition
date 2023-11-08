
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from preprocessing import preprocessing
import pandas as pd


preprocessing = preprocessing()
data = pd.read_csv('Dataset/train.csv', index_col=0)
test_data = pd.read_csv('Dataset/test.csv', index_col=0)
# Load and preprocess dataset
X_train, y_train = preprocessing.train(data)
X_test = preprocessing.test(test_data)

svc = svm.SVC(kernel='rbf', C=1, gamma=0.1)
svc.fit(X_train, y_train)
X_test["Label"] = svc.predict(X_test)
X_test["Label"].to_csv('submission.csv', header=True)
