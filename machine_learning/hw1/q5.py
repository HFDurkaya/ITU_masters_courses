import matplotlib.pyplot as plt
import numpy as np

def laplace_distribution(x):
    return 0.5 * np.exp(-np.abs(x))

if __name__ == '__main__':
    x = np.linspace(-15, 15, 1000)
    plt.figure()
    plt.plot(x, laplace_distribution(x), color='red', label='Laplace Distribution')
    plt.legend()
    plt.show()
