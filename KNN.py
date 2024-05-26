import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.x_train = X
        self.y_train = y

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        distances = []
        for x_train in self.x_train:
            dist = np.linalg.norm(x - x_train)
            distances.append(dist)

        top_k = np.argsort(distances)[:self.k]
        pred_labels = [self.y_train[i] for i in top_k]
        return Counter(pred_labels).most_common()[0][0]
