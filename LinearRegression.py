import numpy as np


class LinearRegression:
    def __init__(self, inp_shape):
        self.w = np.random.uniform(size=(inp_shape,))
        self.b = np.random.uniform(size=(1,))

    def predict(self, X):
        return np.dot(X, self.w) + self.b

    def loss(self, x, y):
        predictions = self.predict(x)
        return np.mean((predictions - y) ** 2) / 2

    def derivative(self, X, y):
        predictions = self.predict(X)
        dw = np.mean(np.dot(X.T, (predictions - y)))
        db = np.mean((predictions - y))
        return dw, db

    def update_weights(self, dw, db, lr):
        self.w -= lr * dw
        self.b -= lr * db

    def fit(self, X, y, lr, num_epochs):
        for i in range(num_epochs):
            dw, db = self.derivative(X, y)
            self.update_weights(dw, db, lr)
            current_loss = self.loss(X, y)
            if i % 100 == 0:
                print(f"Epoch {i}: Loss = {current_loss}")


if __name__ == "__main__":
    np.random.seed(0)
    x_train = np.random.rand(100, 4)
    y_train = 3 * x_train @ np.array([1, 2, 3, 4]) + 5 + np.random.randn(100) * 0.5

    model = LinearRegression(inp_shape=4)
    model.fit(x_train, y_train, num_epochs=1000, lr=0.01)
    predictions = model.predict(x_train)
    print(predictions[:5])
    print(y_train[:5])
