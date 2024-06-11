# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:10:46 2024

@author: S.M.H Mousavi
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SupervisedNeuralGas:
    def __init__(self, n_units=3, max_iter=100, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = None
        self.unit_labels = None

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data, labels):
        n_samples, n_features = data.shape
        self.units = np.random.rand(self.n_units, n_features)
        self.unit_labels = np.random.choice(labels, self.n_units)

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
                    if rank == 0:  # Update label of the closest unit
                        self.unit_labels[idx] = label
            
            # Print the iteration number
            print(f"Iteration {i+1}/{self.max_iter}")

    def predict(self, data):
        dists = np.linalg.norm(self.units[:, np.newaxis, :] - data, axis=2)
        closest_units = np.argmin(dists, axis=0)
        return self.unit_labels[closest_units]

# Load Iris data
iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# Normalize data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, test_size=0.3, random_state=42)

# Train Supervised Neural Gas
sng = SupervisedNeuralGas(n_units=100, max_iter=100)
sng.train(X_train, y_train)

# Predict and evaluate
y_pred = sng.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f' ')
print(f'Test Accuracy: {accuracy:.4f}')



