import numpy as np
import matplotlib.pyplot as plt


class logistic_regression():
    def __init__(self) -> None:
        self.loss = 0

    def sigmoid(self, z):
        if z >= 0:
            exp = np.exp(-z)
            return 1/(exp+1)
        else:
            exp = np.exp(z)
            return exp/(exp+1)

    def compute_sigmoid(self, z):
        sigmoid_func = np.vectorize(self.sigmoid)
        res = sigmoid_func(z)
        return res

    def compute_loss(self, y_hat, y):
        m = y_hat.shape[0]
        L = (-1/m) * np.sum((y * np.log(y_hat + 1e-9)) +
                            ((1 - y) * np.log(1 - y_hat + 1e-9)))
        return L

    def gradients(self, x, y, predictions):
        # w = np.mean(np.matmul(x.T, (predictions - y)))
        # b = np.mean(predictions - y)
        # return w, b
        difference = predictions - y
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def fit(self, X, y, epochs, learning_rate):
        self.bias = 0
        self.weights = np.zeros(X.shape[1])

        for epoch in range(epochs):
            z = np.matmul(X, self.weights.T) + self.bias
            predictions = self.compute_sigmoid(z)
            self.loss = self.compute_loss(predictions, y)
            weights, bias = self.gradients(X, y, predictions)
            self.weights = self.weights - learning_rate * weights
            self.bias = self.bias - learning_rate * bias

    def predict(self, x):
        z = np.matmul(x, self.weights.T) + self.bias
        probabilities = self.compute_sigmoid(z)
        return probabilities

    def accuracy(self, y, predictions):
        return np.mean(y == predictions)
