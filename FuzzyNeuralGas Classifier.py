# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:23:25 2024

@author: S.M.H Mousavi
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class FuzzySupervisedNeuralGas:
    def __init__(self, n_units=3, max_iter=100, eta_start=0.05, eta_end=0.01, lambda_start=30, lambda_end=0.1, n_classes=3):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = None
        self.unit_labels = None
        self.n_classes = n_classes

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data, labels):
        n_samples, n_features = data.shape
        self.units = np.random.rand(self.n_units, n_features)
        self.unit_labels = np.zeros((self.n_units, self.n_classes))

        for i in range(self.max_iter):
            eta = self._update_learning_rate(i)
            lambd = self._update_neighborhood_range(i)

            indices = np.random.permutation(n_samples)
            for index in indices:
                x = data[index]
                label = labels[index]
                dists = np.linalg.norm(self.units - x, axis=1)
                ranking = np.argsort(dists)
                
                for rank, idx in enumerate(ranking):
                    influence = np.exp(-rank / lambd)
                    self.units[idx] += eta * influence * (x - self.units[idx])
                    if rank == 0:  # Update fuzzy memberships for the closest unit
                        self.unit_labels[idx, label] += influence
            
            # Normalize fuzzy memberships to sum to 1
            self.unit_labels /= self.unit_labels.sum(axis=1, keepdims=True)

            print(f"Iteration {i+1}/{self.max_iter}")

    def predict(self, data):
        dists = np.linalg.norm(self.units[:, np.newaxis, :] - data, axis=2)
        closest_units = np.argmin(dists, axis=0)
        fuzzy_outputs = self.unit_labels[closest_units]
        # Defuzzification: Choosing the class with the maximum membership value
        return np.argmax(fuzzy_outputs, axis=1)

# Load and preprocess data
iris = load_iris()
data = iris.data
target = iris.target

scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, test_size=0.3, random_state=42)

# Train Fuzzy Supervised Neural Gas
sng = FuzzySupervisedNeuralGas(n_units=100, max_iter=100, n_classes=len(np.unique(target)))
sng.train(X_train, y_train)

# Predict and evaluate
y_pred = sng.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(' ')
print(f'Test Accuracy: {accuracy:.4f}')
