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
        return loss

    def gradients(self, x, y, predictions):
        n = len(y)
        # w = -(np.dot(x.T, (y - predictions)))/n
        w = (np.dot(x.T, (predictions - y)))/n
        b = np.mean(predictions - y)
        # add the L2 penalty term
        w += self.alpha * self.weights
        b += self.alpha * self.bias
        return w, b

    def show_err(self, epoch, error, val_error):
        plt.figure(figsize=(9, 4))
        plt.plot(epoch, error, color='green', label='Train')
        plt.plot(epoch, val_error, color='red', label='Validation')
        plt.xlabel("Number of Epoch")
        plt.ylabel("Error")
        plt.title("Error Minimization")
        plt.legend()
        plt.show()

    def show_acc(self, epoch, tr_acc, val_acc):
        plt.figure(figsize=(9, 4))
        plt.plot(epoch, tr_acc, color='green', label='Train')
        plt.plot(epoch, val_acc, color='red', label='Validation')
        plt.xlabel("Number of Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Maximization")
        plt.legend()
        plt.show()

    def fit(self, X, Y, epochs, learning_rate, rho, alpha, gamma, x_val, y_val):
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
            # self.weights = self.weights - learning_rate * weights
            # self.bias = self.bias - learning_rate * bias

            # momentum = rho * momentum + learning_rate * weights
            # self.weights -= momentum

            # momentum_weight = (momentum_weight * rho) + \
            #     (1 - rho) * (weights ** 2)
            # self.weights = self.weights - \
            #     (learning_rate/np.sqrt(momentum_weight + 1e-07)) * weights
            # momentum_loss = (momentum_loss * rho) + (1 - rho) * bias
            # self.bias = self.bias - learning_rate * (momentum_loss)

            Vb = rho * Vb + (1 - rho) * bias
            Sb = gamma * Sb + (1 - gamma) * (bias ** 2)
            self.bias -= learning_rate * (Vb / (np.sqrt(Sb) + 1e-07))

            Vw = rho * Vw + (1 - rho) * weights
            Sw = gamma * Sw + (1 - gamma) * (weights ** 2)
            self.weights -= learning_rate * (Vw / (np.sqrt(Sw) + 1e-07))

        #     if epoch % 50 == 0:
        #         self.loss = self.compute_loss(predictions, y)
        #         self.compute_val_loss(x_val, y_val)
        #         num_epoch.append(epoch)
        #         error.append(self.loss)
        #         val_error.append(self.loss_val)
        #         train_pred = self.predict(X)
        #         train_acc.append(self.accuracy(Y, train_pred))
        #         val_pred = self.predict(x_val)
        #         val_acc.append(self.accuracy(y_val, val_pred))

        # self.show_err(num_epoch, error, val_error)
        # self.show_acc(num_epoch, train_acc, val_acc)

        return self.loss

    def compute_val_loss(self, x, y):
        y = self.one_hot_encoding(y)
        z = np.matmul(x, self.weights) + self.bias
        predictions = self.softmax(z)
        loss = self.compute_loss(predictions, y)
        self.loss_val = loss

    def predict(self, x):
        z = np.matmul(x, self.weights) + self.bias
        probabilities = self.softmax(z)
        predictions = np.argmax(probabilities, axis=1)
        return predictions

    def accuracy(self, y, predictions):
        return np.mean(y == predictions)
