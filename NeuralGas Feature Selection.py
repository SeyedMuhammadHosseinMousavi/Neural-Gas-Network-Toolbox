# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:08:11 2024

@author: S.M.H Mousavi
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

class NeuralGas:
    def __init__(self, n_units=3, max_iter=1000, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = None
        self.feature_importance = None

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data):
        n_samples, n_features = data.shape
        self.units = np.random.rand(self.n_units, n_features)
        self.feature_importance = np.zeros(n_features)

        for i in range(self.max_iter):
            eta = self._update_learning_rate(i)
            lambd = self._update_neighborhood_range(i)

            indices = np.random.permutation(n_samples)
            for index in indices:
                x = data[index]
                dists = np.linalg.norm(self.units - x, axis=1)
                ranking = np.argsort(dists)
                
                for rank, idx in enumerate(ranking):
                    influence = np.exp(-rank / lambd)
                    self.feature_importance += eta * influence * np.abs(x - self.units[idx])
                    self.units[idx] += eta * influence * (x - self.units[idx])

    def get_feature_importance(self):
        return self.feature_importance / np.sum(self.feature_importance)

# Load Iris data
iris = load_iris()
data = iris.data
target = iris.target
feature_names = iris.feature_names

# Normalize data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Train Neural Gas
ng = NeuralGas(n_units=3, max_iter=100)
ng.train(data_normalized)

# Get and print feature importance
feature_importance = ng.get_feature_importance()
for name, importance in zip(feature_names, feature_importance):
    print(f'Feature: {name}, Importance: {importance:.4f}')

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.bar(feature_names, feature_importance)
plt.title('Feature Importance using Neural Gas')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.6, random_state=42)

# Train and evaluate XGBoost with all features
model_all_features = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model_all_features.fit(X_train, y_train)
y_pred_all_features = model_all_features.predict(X_test)
accuracy_all_features = accuracy_score(y_test, y_pred_all_features)
print(f'Accuracy with all features: {accuracy_all_features:.4f}')

# Select top features based on importance
top_features_indices = np.argsort(feature_importance)[-2:]  # Select top 2 features as an example
X_train_selected = X_train[:, top_features_indices]
X_test_selected = X_test[:, top_features_indices]

# Train and evaluate XGBoost with selected features
model_selected_features = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model_selected_features.fit(X_train_selected, y_train)
y_pred_selected_features = model_selected_features.predict(X_test_selected)
accuracy_selected_features = accuracy_score(y_test, y_pred_selected_features)
print(f'Accuracy with NGN selected features: {accuracy_selected_features:.4f}')
