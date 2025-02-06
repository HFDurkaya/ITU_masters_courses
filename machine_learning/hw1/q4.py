import numpy as np
import matplotlib.pyplot as plt


def variance(x):
    mean = np.mean(x)
    return np.sum((x - mean) ** 2) / (len(x) - 1)


def covariance(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)

    return (np.sum(np.multiply(x - mean_x, y - mean_y))) / (len(x) - 1)


def plot_normal_distribution(x):
    mean = np.mean(x)
    std = np.std(x)
    x.sort()
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    plt.figure()
    plt.plot(x, y, color='red', label='Normal Distribution')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()


def scatter_plot(x, y):
    plt.figure()
    plt.scatter(x, y, color='red', label='Age-Weight')
    plt.xlabel('Age')
    plt.ylabel('Weight(lbs)')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    X = np.array([69,74,68,70,72,67,66,70, 76,68,72,79,74,67,66,71,74,75,75,76])
    Y = np.array([153,175,155,135,172,150,115,137,200,130,140, 265,185,112,140, 150,165,185,210,220])

    cov = covariance(X, Y)

    vals, counts = np.unique(X, return_counts=True)
    index = np.argmax(counts)

    print(f'Mean of X: {X.mean()}')
    print(f'Median of X: {np.median(X)}')
    print(f'Mode of X: {vals[index]}')
    print(f'Variance of Y: {variance(Y)}')
    print(f'Probability of observing age higher than 80: {(np.sum(X > 80) / len(X)) * 100}%')
    print(f'Covariance Matrix: {np.array([[variance(X), cov], [cov, variance(Y)]])}')
    print(f'Correlation: {cov/(variance(X) ** 0.5 * variance(Y) ** 0.5)}')

    plot_normal_distribution(X)
    scatter_plot(X, Y)