import numpy as np
import matplotlib.pyplot as plt


class logistic_regression():
    def __init__(self) -> None:
        self.loss = 0

    def softmax(self, z):
        p = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
        return p

    def one_hot_encoding(self, y):
        onehot = []
        for target in y:
            if target == 0:
                encode = np.array([1, 0, 0])
            elif target == 1:
                encode = np.array([0, 1, 0])
            elif target == 2:
                encode = np.array([0, 0, 1])

            onehot.append(encode)
        return np.array(onehot)

    def compute_loss(self, y_hat, y):
        n = len(y)
        f = y * np.log(y_hat)
        f = f.to_numpy()
        loss = -(np.sum(f))/n
        # add the Lasso penalty term
        # lasso_penalty = self.alpha * np.sum(np.abs(self.weights))
        # loss += lasso_penalty
        return loss

    def gradients(self, x, y, predictions):
        n = len(y)
        # w = -(np.dot(x.T, (y - predictions)))/n
        w = (np.dot(x.T, (predictions - y)))/n
        b = np.mean(predictions - y)
        # add the Lasso penalty term
        # lasso_penalty = self.alpha * np.sign(self.weights)
        # w += lasso_penalty
        return w, b

    def show_err(self, epoch, error):
        plt.figure(figsize=(9, 4))
        plt.plot(epoch, error, "m-")
        plt.xlabel("Number of Epoch")
        plt.ylabel("Error")
        plt.title("Error Minimization")
        plt.show()

    def fit(self, X, y, epochs, learning_rate, alpha):
        self.bias = 0
        num_epoch = []
        error = []
        self.alpha = alpha
        y = self.one_hot_encoding(y)
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1], y.shape[1]))

        for epoch in range(epochs):
            z = np.matmul(X, self.weights) + self.bias
            predictions = self.softmax(z)
            self.loss = self.compute_loss(predictions, y)
            weights, bias = self.gradients(X, y, predictions)
            self.weights = self.weights - learning_rate * weights
            self.bias = self.bias - learning_rate * bias
            if epoch % 100 == 0:
                num_epoch.append(epoch)
                error.append(self.loss)
                print(f'epoch {epoch}, Error = {self.loss}')

        self.show_err(num_epoch, error)

    def predict(self, x):
        z = np.matmul(x, self.weights) + self.bias
        probabilities = self.softmax(z)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def accuracy(self, y, predictions):
        return np.mean(y == predictions)
