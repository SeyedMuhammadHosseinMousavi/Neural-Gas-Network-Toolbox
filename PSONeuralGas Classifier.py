# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 00:44:26 2024

@author: S.M.H Mousavi
"""

# Load libraries
from pyswarm import pso
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Define the Neural Gas Model
class SupervisedNeuralGas:
    def __init__(self, n_units=100, max_iter=100, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = np.random.rand(n_units, 4)  # Assuming iris dataset
        self.unit_labels = np.zeros(n_units)

    def train(self, data, labels):
        for i in range(self.max_iter):
            print(f"Training iteration {i+1}")
            # Simulated training logic

    def predict(self, data):
        return np.random.randint(0, 3, size=data.shape[0])

# Objective function for PSO
def objective_function(params):
    n_units, eta_start = int(params[0]), params[1]
    model = SupervisedNeuralGas(n_units=n_units, eta_start=eta_start)
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Current params: {params}, Accuracy: {accuracy}")
    return -accuracy

# Load and preprocess data
iris = load_iris()
data = iris.data
target = iris.target
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)
X_train, X_test, y_train, y_test = train_test_split(data_normalized, target, test_size=0.2, random_state=42)

# Set bounds for PSO and execute
lb = [3, 0.01]  # Lower bounds of n_units and eta_start
ub = [50, 0.9]  # Upper bounds
xopt, fopt = pso(objective_function, lb, ub, swarmsize=50, omega=0.5, phip=0.5, phig=0.5, maxiter=100, debug=True)

print("Best parameters found: ", xopt)
print("Best accuracy achieved: ", -fopt)
