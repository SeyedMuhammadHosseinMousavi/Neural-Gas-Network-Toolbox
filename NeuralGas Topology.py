import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class NeuralGas:
    def __init__(self, n_units=100, max_iter=300, eta_start=0.5, eta_end=0.01, lambda_start=30, lambda_end=0.1):
        self.n_units = n_units
        self.max_iter = max_iter
        self.eta_start = eta_start
        self.eta_end = eta_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.units = np.random.rand(n_units, 2)  # Initialize units in 2D space

    def train_step(self, data, i):
        eta = self.eta_start * (self.eta_end / self.eta_start) ** (i / self.max_iter)
        lambda_val = self.lambda_start * (self.lambda_end / self.lambda_start) ** (i / self.max_iter)

        np.random.shuffle(data)
        for point in data:
            dists = np.linalg.norm(self.units - point, axis=1)
            ranking = np.argsort(dists)
            for rank, idx in enumerate(ranking):
                influence = np.exp(-rank / lambda_val)
                self.units[idx] += eta * influence * (point - self.units[idx])

def generate_random_shape(num_points=100):
    angles = np.sort(2 * np.pi * np.random.rand(num_points))
    radii = 0.5 + np.random.rand(num_points) * 0.5  # Random radii from 0.5 to 1.0
    x = radii * np.cos(angles)
    y = radii * np.sin(angles)
    return np.column_stack((x, y))

# Generate a random 2D shape
shape_points = generate_random_shape(num_points=100)

# Initialize Neural Gas
ng = NeuralGas(n_units=70, max_iter=300)

# Set up the figure, the axis, and the plot elements
fig, ax = plt.subplots(figsize=(8, 8))
data_scatter = ax.scatter(shape_points[:, 0], shape_points[:, 1], alpha=0.5, label='Shape Points')
units_scatter, = ax.plot([], [], 'ro', label='NGN Units')
ax.legend()

def init():
    units_scatter.set_data([], [])
    return units_scatter,

def update(frame):
    ng.train_step(shape_points, frame)
    units_scatter.set_data(ng.units[:, 0], ng.units[:, 1])
    ax.set_title(f'Iteration {frame+1}/{ng.max_iter}')
    return units_scatter,

ani = FuncAnimation(fig, update, frames=range(ng.max_iter), init_func=init, blit=True, interval=50)

plt.show()
