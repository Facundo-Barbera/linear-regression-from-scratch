from random import randint
from typing import List

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class LinearRegression:

    def __init__(self, data, betas: List | None = None, bias: int | None = None,
                 learning_rate: float | None = None, epochs: int = 10000):
        self.data = data
        self.betas = [randint(0, 10) for i in range(len(self.data[0]))] if betas is None else betas
        self.bias = randint(0, 10) if bias is None else bias
        self.learning_rate = self._auto_learning_rate() if learning_rate is None else learning_rate
        self.epochs = epochs
        self.loss_history: List[float] = []
        self.status = 'initialized'

    # Public methods #

    def fit(self):
        self._fit()

    def predict(self, x_data: List | None = None):
        return self._predict(x_data=x_data)

    # Private methods #

    def _auto_learning_rate(self):
        n = len(self.data[0][0])
        scale = sum(sum(x ** 2 for x in feature) / n for feature in self.data[0]) + 1
        return 1 / scale

    def _predict(self, x_data: List | None = None):
        x_data = x_data if x_data else self.data[0]
        y_pred = []
        for j in range(len(x_data[0])):
            total = self.bias
            for i in range(len(x_data)):
                total += self.betas[i] * x_data[i][j]
            y_pred.append(total)
        return y_pred

    def _fit(self):
        for epoch in range(self.epochs):
            y_pred = self._predict()
            self.loss_history.append(self._mse(self.data[1], y_pred))
            self._adjust_betas_bias(y_pred)

        self.status = 'fitted'

    def _adjust_betas_bias(self, y_pred: List):
        for i in range(len(self.betas)):
            total = 0
            for j in range(len(self.data[1])):
                total += (y_pred[j] - self.data[1][j]) * self.data[0][i][j]
            self.betas[i] -= (self.learning_rate / len(self.data[0][0])) * total

        self.bias -= (self.learning_rate / len(self.data[0][0]) * (sum(y_pred) - sum(self.data[1])))

    # Static methods #

    @staticmethod
    def _mse(y_true: List, y_pred: List):
        return (sum((y_pred[i] - y_true[i]) ** 2 for i in range(len(y_true)))) / len(y_true)

    # Magic methods #

    def __repr__(self):
        if self.status == 'initialized':
            return 'LinearRegression has not been fitted yet.'
        if self.status == 'fitted':
            return f'LinearRegression has been fitted with betas: {self.betas} and bias: {self.bias}'
        else:
            return 'LinearRegression has an unknown status.'


if __name__ == "__main__":
    X = [[i for i in range(100)]]
    y = [randint(30, 60) for i in range(100)]

    data_list = [X, y]

    lr = LinearRegression(data_list, epochs=100000)
    lr.fit()
    print(lr)

    if plt:
        plt.scatter(X[0], y)
        plt.plot(X[0], lr.predict(), color='red')
        plt.show()
