from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from skopt import forest_minimize
from skopt.space import Real, Integer, Categorical
from preprocessing import preprocessing

# Define the hyperparameter search space
space = [
    Integer(10, 1000, name='n_estimators'),
    Integer(1, 50, name='max_depth'),
    Integer(2, 10, name='min_samples_split'),
    Integer(1, 10, name='min_samples_leaf'),
    Categorical(['entropy'], name='criterion')
]

data = pd.read_csv('Dataset/train.csv', index_col=0)
preprocessing = preprocessing()
X_train, y_train, X_val, y_val = preprocessing.train_valid(data)


# Define the objective function to minimize

iteration = 0


def objective(params):
    global iteration
    iteration += 1
    n_estimators, max_depth, min_samples_split, min_samples_leaf, criterion = params
    print(f"Iteration {iteration}: {params}")
    clf = RandomForestClassifier(n_estimators=n_estimators,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 criterion=criterion,
                                 random_state=42)
    clf.fit(X_train, y_train)
    print(f"Accuracy: {clf.score(X_val, y_val)}")
    return -clf.score(X_val, y_val)  # negative accuracy to minimize


# Perform hyperparameter tuning with Latin Hypercube method
result = forest_minimize(objective, space, n_calls=50, random_state=42)

# Print the best hyperparameters found
print(f"\nBest hyperparameters: {result.x}")
print(f"Best accuracy: {-result.fun}")
