# Hasan F. Durkaya
# 504241526

import numpy as np
import matplotlib.pyplot as plt

# Data Generation
np.random.seed(42)
cluster_count = 3
samples = [300, 350, 400]

# Cluster parameters, means and covariances.
means = [np.array([2, 2]), np.array([6, 6]), np.array([10, 2])]
isotropic_variance = 1.0                                                        # Uniform variance for all features
covariances = [np.eye(2) * isotropic_variance for _ in range(cluster_count)]

# Generating data points.
data = np.vstack([
    np.random.multivariate_normal(mean, cov, n)
    for mean, cov, n in zip(means, covariances, samples)
])

# EM Algorithm
samples, features = data.shape

# Initial parameters
weights = np.ones(cluster_count) / cluster_count  
means = np.random.uniform(data.min(axis=0), data.max(axis=0), size=(cluster_count, features))
covariances = [np.eye(features) * isotropic_variance for _ in range(cluster_count)]

# Gaussian Density Function
def gaussian_density(x, mean, cov):
    size = len(x)
    det = np.linalg.det(cov)
    norm_const = 1.0 / (np.power((2 * np.pi), float(size) / 2) * np.sqrt(det))
    x_mu = x - mean
    inv_cov = np.linalg.inv(cov)
    result = np.exp(-0.5 * np.dot(x_mu.T, np.dot(inv_cov, x_mu)))
    return norm_const * result

# iterations, tolerance and log likelihoods
max_iter = 100
tolerance = 1e-4
log_likelihoods = []

#EM Algorithm Loop
for iteration in range(max_iter):
    
    # E-step: Compute responsibilities, expected contributions.
    responsibilities = np.zeros((samples, cluster_count))
    for i in range(samples):
        for j in range(cluster_count):
            responsibilities[i, j] = weights[j] * gaussian_density(data[i], means[j], covariances[j])
        responsibilities[i, :] /= np.sum(responsibilities[i, :])

    # M-step: Update parameters.
    new_weights = np.mean(responsibilities, axis=0)
    new_means = np.array([
        np.sum(responsibilities[:, j][:, np.newaxis] * data, axis=0) / np.sum(responsibilities[:, j])
        for j in range(cluster_count)
    ])
    new_covariances = [
        np.eye(features) * isotropic_variance  # Same isotropic variance for each cluster
        for j in range(cluster_count)
    ]

    # Log likelihood calculation.
    log_likelihood = np.sum([
        np.log(np.sum([
            weights[j] * gaussian_density(data[i], means[j], covariances[j])
            for j in range(cluster_count)
        ]))
        for i in range(samples)
    ])
    log_likelihoods.append(log_likelihood)

    # Checking if algorithm has converged.
    if iteration > 0 and np.abs(log_likelihood - log_likelihoods[-2]) < tolerance:
        break

    #Updating parameters for next iteration.
    weights, means, covariances = new_weights, new_means, new_covariances

# Results and Visualization part.
x, y = np.meshgrid(
    np.linspace(data[:, 0].min() - 1, data[:, 0].max() + 1, 100),
    np.linspace(data[:, 1].min() - 1, data[:, 1].max() + 1, 100)
)
grid = np.c_[x.ravel(), y.ravel()]
z = np.zeros((grid.shape[0], cluster_count))

for j in range(cluster_count):
    for i, point in enumerate(grid):
        z[i, j] = gaussian_density(point, means[j], covariances[j])

plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], s=10, label="Data points", alpha=0.5)
for j in range(cluster_count):
    z_j = z[:, j].reshape(x.shape)
    plt.contour(x, y, z_j, levels=5, alpha=0.8)
plt.scatter(means[:, 0], means[:, 1], c='red', label="Cluster Means", marker='x')
plt.title("Gaussian Mixture Contours after EM")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.savefig("q1_results.png")
