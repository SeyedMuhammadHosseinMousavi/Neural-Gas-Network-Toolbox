# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:56:07 2024

@author: S.M.H Mousavi
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from sklearn.preprocessing import StandardScaler
from PIL import Image

class NeuralGas:
    def __init__(self, n_units=10, max_iter=100, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = None

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data):
        n_samples, n_features = data.shape
        self.units = np.random.rand(self.n_units, n_features)

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
                    self.units[idx] += eta * influence * (x - self.units[idx])

    def assign_clusters(self, data):
        dists = np.linalg.norm(self.units[:, np.newaxis, :] - data, axis=2)
        return np.argmin(dists, axis=0)

# Load an image
image_path = 'scor2.jpg'
image = img_as_float(io.imread(image_path))
pixels = image.reshape(-1, 3)  # Reshape image to a 2D array of pixels

# Normalize pixel data
scaler = StandardScaler()
pixels_normalized = scaler.fit_transform(pixels)

# Train Neural Gas
ng = NeuralGas(n_units=5, max_iter=20)
ng.train(pixels_normalized)

# Assign clusters to pixels
cluster_assignment = ng.assign_clusters(pixels_normalized)

# Reshape cluster assignments into the image shape
segmented_image = cluster_assignment.reshape(image.shape[0], image.shape[1])

# Plot the original and segmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='viridis')
plt.title('NGN Segmented Image')
plt.axis('off')

plt.show()
