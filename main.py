import pandas as pd
import numpy as np
import sys
import argparse
from IPython.display import display, clear_output
from logistic_regression import logistic_regression
from preprocessing import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import argparse

# Read the data
data = pd.read_csv('Dataset/train.csv', index_col=0)
test_data = pd.read_csv('Dataset/test.csv', index_col=0)
# Initialize the classes
prepro = preprocessing()
lr = logistic_regression()


parser = argparse.ArgumentParser(description='Kaggle competition')
parser.add_argument(
    'model', type=str, help='Model to use: logistic or randomforest or svm')
parser.add_argument('action', type=str,
                    help='Action to perform: hptuning or submit')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='Learning rate for logistic regression', required=False)
parser.add_argument('--rho', type=float, default=0.99,
                    help='Rho for logistic regression', required=False)
parser.add_argument('--alpha', type=float, default=0.00001,
                    help='Alpha for logistic regression', required=False)
parser.add_argument('--loggamma', type=float, default=0.98,
                    help='Gamma for logistic regression', required=False)
parser.add_argument('--n_estimators', type=int, default=100,
                    help='Number of estimators for random forest', required=False)
parser.add_argument('--max_depth', type=int, default=5,
                    help='Max depth for random forest', required=False)
parser.add_argument('--min_samples_split', type=int, default=2,
                    help='Min samples split for random forest', required=False)
parser.add_argument('--min_samples_leaf', type=int, default=1,
                    help='Min samples leaf for random forest', required=False)
parser.add_argument('--kernel', type=str, default='rbf',
                    help='Kernel for SVM', required=False)
parser.add_argument('--degree', type=int, default=3,
                    help='Degree for SVM', required=False)
parser.add_argument('--C', type=float, default=1,
                    help='C for SVM', required=False)
parser.add_argument('--svmgamma', type=float, default=0.1,
                    help='Gamma for SVM', required=False)

args = vars(parser.parse_args())


try:
    # Check the arguments and run the corresponding code

    # This part will run if we want to use the logistic regression model and tune its hyperparameters
    if args['model'] == 'logistic' and args['action'] == 'hptuning':
        # Split the data into train and validation sets
        X_train, y_train, X_val, y_val = prepro.train_valid(data)
        # Define the hyperparameters to tune
        gammas = [0.94, 0.96, 0.98, 0.99]
        alphas = [0.0001, 0.00001, 0.00005]
        rhos = [0.95, 0.97, 0.99]
        etas = [0.1, 0.01, 0.3, 0.6]
        # Create a dataframe to store the results
        df = pd.DataFrame(
            columns=['eta', 'rho', 'gamma', 'alpha', 'loss', 'train_acc', 'valid_acc'])
        # Loop through all the hyperparameters and store the results in the dataframe
        for rho in rhos:
            for gamma in gammas:
                for eta in etas:
                    for alpha in alphas:
                        # Call the fit function to train the model with respect to the hyperparameters
                        loss = lr.fit(X_train, y_train, 2500, eta,
                                      rho, alpha, gamma, X_val, y_val)
                        # Calculate the accuracy of the model on the train and validation sets
                        train_pred = lr.predict(X_train)
                        train_acc = lr.accuracy(y_train, train_pred)
                        val_pred = lr.predict(X_val)
                        val_acc = lr.accuracy(y_val, val_pred)
                        # Store the results in the dataframe
                        new_row = {'eta': eta, 'rho': rho, 'gamma': gamma, 'alpha': alpha,
                                   'loss': loss, 'train_acc': train_acc, 'valid_acc': val_acc}
                        df.loc[len(df)] = new_row
                        # Sort the dataframe by the loss in order to find the best hyperparameters
                        df = df.sort_values(by=['loss'])
                        # Display the dataframe
                        clear_output(wait=True)
                        display(df)

    # This part will run if we want to use the logistic regression model and submit the results
    elif args['model'] == 'logistic' and args['action'] == 'submit':
        # Preprocess the data
        X_train, y_train = prepro.train(data)
        X_test = prepro.test(test_data)

        # loss = lr.fit(X_train, y_train, 2500, 0.10, 0.99, 0.00001, 0.98, 0, 0)
        # Call the fit function to train the model with respect to the hyperparameters given as arguments
        loss = lr.fit(X_train, y_train, 2500,
                      learning_rate=args['learning_rate'], rho=args['rho'], alpha=args['alpha'], gamma=args['loggamma'], x_val=0, y_val=0, submission=True)
        # Calculate the accuracy of the model on the train set
        train_pred = lr.predict(X_train)
        train_acc = lr.accuracy(y_train, train_pred)
        print(f'Loss: {loss}\nTrain Accuracy: {train_acc}')
        # Predict the labels of the test set
        X_test["Label"] = lr.predict(X_test)
        # Save the results in a csv file
        X_test["Label"].to_csv('submission.csv', header=True)

    # This part will run if we want to use the random forest model and tune its hyperparameters
    elif args['model'] == 'randomforest' and args['action'] == 'hptuning':
        # Split the data into train and validation sets
        X_train, y_train, X_val, y_val = prepro.train_valid(data)
        # Define the hyperparameters to tune
        n_estimators = [100, 200, 300, 400, 500]
        max_depths = [5, 10, 15, 20, 25, 30]
        min_samples_splits = [2, 5, 10, 15, 20]
        min_samples_leafs = [1, 2, 5, 10, 15]
        # Create a dataframe to store the results
        df = pd.DataFrame(
            columns=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leafs', 'train_acc', 'valid_acc'])
        # Loop through all the hyperparameters and store the results in the dataframe
        for n_estimator in n_estimators:
            for max_depth in max_depths:
                for min_samples_split in min_samples_splits:
                    for min_samples_leaf in min_samples_leafs:
                        # Call the fit function to train the model with respect to the hyperparameters
                        rfc = RandomForestClassifier(n_estimators=n_estimator,
                                                     max_depth=max_depth,
                                                     min_samples_split=min_samples_split,
                                                     min_samples_leaf=min_samples_leaf,
                                                     criterion='entropy',
                                                     random_state=42)
                        rfc.fit(X_train, y_train)
                        # Calculate the accuracy of the model on the train and validation sets
                        train_acc = accuracy_score(
                            y_train, rfc.predict(X_train))
                        val_acc = accuracy_score(y_val, rfc.predict(X_val))
                        # Store the results in the dataframe
                        new_row = {'n_estimators': n_estimator, 'max_depth': max_depth, 'min_samples_split': min_samples_split,
                                   'min_samples_leafs': min_samples_leaf, 'train_acc': train_acc, 'valid_acc': val_acc}
                        df.loc[len(df)] = new_row
                        # Sort the dataframe by the validation accuracy in order to find the best hyperparameters
                        df = df.sort_values(by=['valid_acc'])
                        # Display the dataframe
                        clear_output(wait=True)
                        display(df)

    # This part will run if we want to use the random forest model and submit the results
    elif args['model'] == 'randomforest' and args['action'] == 'submit':
        # Preprocess the data
        X_train, y_train = prepro.train(data)
        X_test = prepro.test(test_data)
        # Call the fit function to train the model with respect to the hyperparameters given as arguments
        rfc = RandomForestClassifier(n_estimators=args['n_estimators'],
                                     max_depth=args['max_depth'],
                                     min_samples_split=args['min_samples_split'],
                                     min_samples_leaf=args['min_samples_leaf'],
                                     criterion='entropy',
                                     random_state=42)
        rfc.fit(X_train, y_train)
        # Calculate the accuracy of the model on the train set
        train_acc = accuracy_score(y_train, rfc.predict(X_train))
        print(f'Train Accuracy: {train_acc}')
        # Predict the labels of the test set
        X_test["Label"] = rfc.predict(X_test)
        # Save the results in a csv file
        X_test["Label"].to_csv('submission.csv', header=True)

    # This part will run if we want to use the SVM model and tune its hyperparameters
    elif args['model'] == 'svm' and args['action'] == 'hptuning':
        # Split the data into train and validation sets
        X_train, y_train, X_val, y_val = prepro.train_valid(data)
        # Define the hyperparameters to tune
        Cs = [0.1, 1, 10, 100, 1000]
        gammas = [0.1, 0.01, 0.001, 0.0001]
        degrees = [1, 2, 3, 4, 5]
        # Create a dataframe to store the results
        df = pd.DataFrame(
            columns=['kernel', 'C', 'gamma', 'train_acc', 'valid_acc'])
        # Loop through all the hyperparameters and store the results in the dataframe
        for c in Cs:
            for gamma in gammas:
                # Call the fit function to train the model with respect to the hyperparameters
                svc = SVC(kernel='rbf', C=c, gamma=gamma)
                svc.fit(X_train, y_train)
                # Calculate the accuracy of the model on the train and validation sets
                train_acc = accuracy_score(y_train, svc.predict(X_train))
                val_acc = accuracy_score(y_val, svc.predict(X_val))
                # Store the results in the dataframe
                new_row = {'kernel': 'rbf', 'C': c, 'gamma': gamma, 'degree': 0,
                           'train_acc': train_acc, 'valid_acc': val_acc}
                df.loc[len(df)] = new_row
                # Sort the dataframe by the validation accuracy in order to find the best hyperparameters
                df = df.sort_values(by=['valid_acc'])
                # Display the dataframe
                clear_output(wait=True)
                display(df)
        for degree in degrees:
            # Call the fit function to train the model with respect to the hyperparameters
            svc = SVC(kernel='poly', degree=degree)
            svc.fit(X_train, y_train)
            # Calculate the accuracy of the model on the train and validation sets
            train_acc = accuracy_score(y_train, svc.predict(X_train))
            val_acc = accuracy_score(y_val, svc.predict(X_val))
            # Store the results in the dataframe
            new_row = {'kernel': 'poly', 'C': 0, 'gamma': 0, 'degree': degree,
                       'train_acc': train_acc, 'valid_acc': val_acc}
            df.loc[len(df)] = new_row
            # Sort the dataframe by the validation accuracy in order to find the best hyperparameters
            df = df.sort_values(by=['valid_acc'])
            # Display the dataframe
            clear_output(wait=True)
            display(df)

    # This part will run if we want to use the SVM model and submit the results
    elif args['model'] == 'svm' and args['action'] == 'submit':
        # Preprocess the data
        X_train, y_train = prepro.train(data)
        X_test = prepro.test(test_data)
        # Call the fit function to train the model with respect to the hyperparameters given as arguments
        if args['kernel'] == 'poly':
            svc = SVC(kernel='poly', degree=args['degree'])
        elif args['kernel'] == 'rbf':
            svc = SVC(kernel='rbf', C=args['C'], gamma=args['svmgamma'])
        svc.fit(X_train, y_train)
        # Calculate the accuracy of the model on the train set
        train_acc = accuracy_score(y_train, svc.predict(X_train))
        print(f'Train Accuracy: {train_acc}')
        # Predict the labels of the test set
        X_test["Label"] = svc.predict(X_test)
        # Save the results in a csv file
        X_test["Label"].to_csv('submission.csv', header=True)


except IndexError:
    print('You have to enter the correct arguments')
