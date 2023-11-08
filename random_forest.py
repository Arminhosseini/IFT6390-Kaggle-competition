import pandas as pd
import numpy as np
from comet_ml import Experiment
from comet_ml import Optimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neural_network import MLPClassifier
from dotenv import load_dotenv
from sklearn.naive_bayes import GaussianNB
from preprocessing import *
import os
load_dotenv()

exp = Experiment(
    api_key=os.environ.get('COMET_API_KEY'),
    project_name='ift6758-project-milestone2',
    workspace='ift6758-b09-project'
)

# setting the hyperparameters we are tuning
model_params = {"n_estimators": {
    "type": "integer",
    "scaling_type": "uniform",
    "min": 100,
    "max": 300},
    "criterion": {
    "type": "categorical",
    "values": ["gini", "entropy"]},
    "min_samples_leaf": {
    "type": "discrete",
    "values": [1, 3, 5, 7, 9]},
    "max_depth": {
    "type": "integer",
    "scaling_type": "uniform",
    "min": 10,
    "max": 100},
    "min_samples_split": {
    "type": "discrete",
    "values": [2, 5, 10]}}

# setting the spec for bayes algorithm
spec = {"maxCombo": 20,
        "objective": "minimize",
        "metric": "loss",
        "minSampleSize": 500,
        "retryLimit": 20,
        "retryAssignLimit": 0}

# defining the configuration dictionary
config_dict = {"algorithm": "bayes",
               "spec": spec,
               "parameters": model_params,
               "name": "Bayes Optimization",
               "trials": 1}


# initializing the comet ml optimizer
opt = Optimizer(api_key=os.environ.get('COMET_API_KEY'),
                config=config_dict,
                project_name="ift6390-kaggle-competition",
                workspace="armin-hsn")


data = pd.read_csv('Dataset/train.csv', index_col=0)

preprocessing = preprocessing()

X_train, y_train, X_val, y_val = preprocessing.train_valid(data)

for experiment in opt.get_experiments():
    # initializing random forest
    # setting the parameters to be optimized with get_parameter
    random_forest = RandomForestClassifier(
        n_estimators=experiment.get_parameter("n_estimators"),
        criterion=experiment.get_parameter("criterion"),
        max_depth=experiment.get_parameter("max_depth"),
        min_samples_split=experiment.get_parameter("min_samples_split"),
        min_samples_leaf=experiment.get_parameter("min_samples_leaf"),
        random_state=25)

    # training the model and making predictions
    random_forest.fit(X_train, y_train)
    valid_pred = random_forest.predict(X_val)
    train_pred = random_forest.predict(X_train)

    # logging the random state and accuracy of each model
    experiment.log_parameter("random_state", 25)
    experiment.log_metric("Validation accuracy",
                          accuracy_score(y_val, valid_pred))
    experiment.log_metric("Training accuracy",
                          accuracy_score(y_train, train_pred))

    experiment.end()


best_experiment = opt.get_best_experiment()

best_n_estimators = best_experiment.get_parameter("n_estimators")
best_criterion = best_experiment.get_parameter("criterion")
best_max_depth = best_experiment.get_parameter("max_depth")
best_min_samples_split = best_experiment.get_parameter("min_samples_split")
best_min_samples_leaf = best_experiment.get_parameter("min_samples_leaf")

test_data = pd.read_csv('Dataset/test.csv', index_col=0)

X_train, y_train = preprocessing.train(data)
X_test = preprocessing.test(test_data)
best_model = best_experiment.get_model()

best_model.fit(X_train, y_train)

X_test["Label"] = best_model.predict(X_test)
X_test["Label"].to_csv('submission.csv', header=True)
