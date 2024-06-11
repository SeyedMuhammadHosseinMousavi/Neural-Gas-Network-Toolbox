import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class SupervisedNeuralGas:
    def __init__(self, n_units_per_class=3, max_iter=100, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units_per_class = n_units_per_class
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = {}
        self.unit_labels = {}

    def _update_learning_rate(self, i):
        return self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)

    def _update_neighborhood_range(self, i):
        return self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

    def train(self, data, labels):
        unique_labels = np.unique(labels)
        n_samples, n_features = data.shape

        for label in unique_labels:
            self.units[label] = np.random.rand(self.n_units_per_class, n_features)
            indices = np.where(labels == label)[0]
            data_class = data[indices]

            for i in range(self.max_iter):
                eta = self._update_learning_rate(i)
                lambd = self._update_neighborhood_range(i)

                np.random.shuffle(indices)
                for index in indices:
                    x = data[index]
                    dists = np.linalg.norm(self.units[label] - x, axis=1)
                    ranking = np.argsort(dists)
                    
                    for rank, idx in enumerate(ranking):
                        influence = np.exp(-rank / lambd)
                        self.units[label][idx] += eta * influence * (x - self.units[label][idx])

                # Print the iteration number
                if (i + 1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.max_iter} for class {label}")

    def generate_synthetic_data(self, n_samples=100, noise_level=0.1):
        synthetic_data = []
        synthetic_labels = []
        for label in self.units:
            n_units, n_features = self.units[label].shape
            samples_per_unit = n_samples // n_units

            for unit in self.units[label]:
                for _ in range(samples_per_unit):
                    noise = np.random.randn(n_features) * noise_level
                    synthetic_data.append(unit + noise)
                    synthetic_labels.append(label)

        return np.array(synthetic_data), np.array(synthetic_labels)

# Load Iris data
iris = load_iris()
data = iris.data
target = iris.target

# Normalize data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Train Supervised Neural Gas
sng = SupervisedNeuralGas(n_units_per_class=10, max_iter=100)
sng.train(data_normalized, target)

# Generate synthetic data
synthetic_data, synthetic_labels = sng.generate_synthetic_data(n_samples=300, noise_level=0.1)

# Plot the original and synthetic data for comparison
plt.figure(figsize=(14, 7))

# Plot original data
plt.subplot(1, 2, 1)
for label in np.unique(target):
    plt.scatter(data_normalized[target == label, 0], data_normalized[target == label, 1], label=f'Class {label}')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

# Plot synthetic data
plt.subplot(1, 2, 2)
for label in np.unique(synthetic_labels):
    plt.scatter(synthetic_data[synthetic_labels == label, 0], synthetic_data[synthetic_labels == label, 1], label=f'Class {label}')
plt.title('Synthetic Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.show()
