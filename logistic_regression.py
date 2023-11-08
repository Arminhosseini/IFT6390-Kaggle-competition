import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Logistic Regression Class


class logistic_regression():
    """
    A logistic regression model for multi-class classification.

    Attributes:
    -----------
    loss : float
        The current loss value of the model.

    Methods:
    --------
    softmax(z)
        Computes the softmax function of the input array.
    one_hot_encoding(y)
        Converts the input array of labels to one-hot encoded vectors.
    compute_loss(y_hat, y)
        Computes the cross-entropy loss between the predicted and true labels.
    gradients(x, y, predictions)
        Computes the gradients of the loss with respect to the weights and bias.
    show_err(epoch, error, val_error)
        Plots the training and validation error over the number of epochs.
    show_acc(epoch, tr_acc, val_acc)
        Plots the training and validation accuracy over the number of epochs.
    fit(X, Y, epochs, learning_rate, rho, alpha, gamma, x_val, y_val)
        Trains the model on the input data and returns the final loss value.
    compute_val_loss(x, y)
        Computes the validation loss of the model on the input data.
    predict(x)
        Predicts the labels of the input data using the trained model.
    accuracy(y, predictions)
        Computes the accuracy of the model on the input data.
    """
    # Constructor

    def __init__(self) -> None:
        self.loss = 0

    # sotmax function for multi-class classification
    def softmax(self, z: pd.DataFrame) -> pd.DataFrame:
        p = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
        return p

    # one-hot encoding of the labels
    def one_hot_encoding(self, y: pd.Series) -> np.ndarray:
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

    # compute the cross-entropy loss
    def compute_loss(self, y_hat: pd.DataFrame, y: np.ndarray) -> float:
        n = len(y)
        f = y * np.log(y_hat)
        f = f.to_numpy()
        loss = -(np.sum(f))/n
        return loss

    # compute the gradients of the loss with respect to the weights and bias and regularize them with L2 penalty
    def gradients(self, x: pd.DataFrame, y: np.ndarray, predictions: pd.DataFrame) -> tuple:
        n = len(y)
        w = (np.dot(x.T, (predictions - y)))/n
        b = np.mean(predictions - y)
        # add the L2 penalty term
        w += self.alpha * self.weights
        b += self.alpha * self.bias
        return w, b

    # plot the training and validation error over the number of epochs
    def show_err(self, epoch: list, error: list, val_error: list) -> None:
        plt.figure(figsize=(9, 4))
        plt.plot(epoch, error, color='green', label='Train')
        plt.plot(epoch, val_error, color='red', label='Validation')
        plt.xlabel("Number of Epoch")
        plt.ylabel("Error")
        plt.title("Error Minimization")
        plt.legend()
        plt.show()

    # plot the training and validation accuracy over the number of epochs
    def show_acc(self, epoch: list, tr_acc: list, val_acc: list) -> None:
        plt.figure(figsize=(9, 4))
        plt.plot(epoch, tr_acc, color='green', label='Train')
        plt.plot(epoch, val_acc, color='red', label='Validation')
        plt.xlabel("Number of Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Maximization")
        plt.legend()
        plt.show()

    # train the model on the input data and return the final loss value based on momentum and adaptive learning rate
    def fit(self, X: pd.DataFrame, Y: pd.Series, epochs: int, learning_rate: float, rho: float, alpha: float, gamma: float, x_val: pd.DataFrame, y_val: pd.Series, submission: bool = False) -> float:
        self.bias = 0
        num_epoch = []
        error = []
        val_error = []
        train_acc = []
        val_acc = []
        self.alpha = alpha
        Vb = 0
        Sb = 0
        Vw = 0
        Sw = 0
        y = self.one_hot_encoding(Y)
        self.weights = np.random.uniform(-1, 1, size=(X.shape[1], y.shape[1]))

        for epoch in range(epochs):
            z = np.matmul(X, self.weights) + self.bias
            predictions = self.softmax(z)
            self.loss = self.compute_loss(predictions, y)
            weights, bias = self.gradients(X, y, predictions)
            Vb = rho * Vb + (1 - rho) * bias
            Sb = gamma * Sb + (1 - gamma) * (bias ** 2)
            self.bias -= learning_rate * (Vb / (np.sqrt(Sb) + 1e-07))

            Vw = rho * Vw + (1 - rho) * weights
            Sw = gamma * Sw + (1 - gamma) * (weights ** 2)
            self.weights -= learning_rate * (Vw / (np.sqrt(Sw) + 1e-07))

            if not submission:
                if epoch % 50 == 0:
                    self.loss = self.compute_loss(predictions, y)
                    self.compute_val_loss(x_val, y_val)
                    num_epoch.append(epoch)
                    error.append(self.loss)
                    val_error.append(self.loss_val)
                    train_pred = self.predict(X)
                    train_acc.append(self.accuracy(Y, train_pred))
                    val_pred = self.predict(x_val)
                    val_acc.append(self.accuracy(y_val, val_pred))

        if not submission:
            self.show_err(num_epoch, error, val_error)
            self.show_acc(num_epoch, train_acc, val_acc)

        return self.loss

    # compute the validation loss of the model on the input data
    def compute_val_loss(self, x: pd.DataFrame, y: pd.Series) -> None:
        y = self.one_hot_encoding(y)
        z = np.matmul(x, self.weights) + self.bias
        predictions = self.softmax(z)
        loss = self.compute_loss(predictions, y)
        self.loss_val = loss

    # predict the labels of the input data using the trained model
    def predict(self, x) -> np.ndarray:
        z = np.matmul(x, self.weights) + self.bias
        probabilities = self.softmax(z)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    # compute the accuracy of the model on the input data
    def accuracy(self, y, predictions) -> float:
        return np.mean(y == predictions)
