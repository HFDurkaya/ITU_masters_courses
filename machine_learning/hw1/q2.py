"""
    @Author: A. Esad Uğur
    @Date: 09.11.2024
    @ID: 504241539
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def plot_data_and_prediction(x, y, y_pred, title):
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Prediction line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Data points and Prediction line - {title}')
    # plt.show()


def plot_data_and_prediction2(x, y, coeffs, title):
    plt.figure()
    plt.scatter(x, y, color='blue', label='Data points')
    xx = np.linspace(np.floor(min(x)), np.ceil(max(x)), 100)
    plt.plot(xx, predict(coeffs, xx), color='red', label='Prediction line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title(f'Data points and Prediction line - {title}')
    # plt.show()


def plot_errors(train_err, val_err, title):
    plt.figure()
    plt.plot(range(1, 9), train_err, color='blue', label='Train Loss')
    plt.plot(range(1, 9), val_err, color='red', linestyle='dashed', label='Validation Loss')
    plt.xlabel('Polynomial Order')
    plt.ylabel('Average MSE')
    plt.legend()
    plt.title(f'Error vs polynomial order - {title}')
    # plt.show()


def least_squares_regression(x, y, degree):
    # Create the design matrix
    X = np.vander(x, degree + 1)
    
    # Compute the coefficients using the normal equation (X^T * X) * beta = X^T * y
    X_transpose = X.T
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_y = np.dot(X_transpose, y)
    coeffs = np.linalg.solve(X_transpose_X, X_transpose_y)
    
    return coeffs


def mean_squared_error(y_true, y_pred):
    #y_true = np.sort(y_true)
    #y_pred = np.sort(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def predict(coeffs, x):
    y_pred = np.dot(np.vander(x, len(coeffs)), coeffs)
    return y_pred


def k_fold_cross_validation(data, k):
    data_shuffled = data.sample(frac=1)

    folds = np.array_split(data_shuffled, k)
    models_train_err = []
    models_val_err = []

    for degree in range(1, 9):
        avg_train_err = 0
        avg_val_err = 0
        
        for i, fold in enumerate(folds):
            train_data = pd.concat([f for j, f in enumerate(folds) if j != i])
            val_data = fold

            x_train = train_data['x:']
            y_train = train_data['y']
            x_val = val_data['x:']
            y_val = val_data['y']

            coeffs = least_squares_regression(x_train, y_train, degree)
            y_train_pred = predict(coeffs, x_train)
            y_val_pred = predict(coeffs, x_val)

            #if i == 4:
                #plot_data_and_prediction(data['x:'], data['y'], predict(coeffs, data['x:']), str(degree))
                #plot_data_and_prediction2(data_shuffled['x:'], data_shuffled['y'], coeffs, str(degree))

            train_err = mean_squared_error(y_train, y_train_pred)
            val_err = mean_squared_error(y_val, y_val_pred)

            avg_train_err += train_err
            avg_val_err += val_err
        
        avg_train_err /= (k)
        avg_val_err /= (k)

        models_train_err.append(avg_train_err)
        models_val_err.append(avg_val_err)
    plt.show()
    return models_train_err, models_val_err


def sklearn_k_fold_cross_validation(data, k):
    kf = KFold(n_splits=k, shuffle=True, random_state=1)
    x = data['x:'].values.reshape(-1, 1)
    y = data['y'].values

    models_train_err = []
    models_val_err = []

    for degree in range(1, 9):
        avg_train_err = 0
        avg_val_err = 0

        poly = PolynomialFeatures(degree)
        x_poly = poly.fit_transform(x)

        for train_index, val_index in kf.split(x):
            x_train, x_val = x_poly[train_index], x_poly[val_index]
            y_train, y_val = y[train_index], y[val_index]

            model = LinearRegression()
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_val_pred = model.predict(x_val)

            train_err = mean_squared_error(y_train, y_train_pred)
            val_err = mean_squared_error(y_val, y_val_pred)

            avg_train_err += train_err
            avg_val_err += val_err

        avg_train_err /= k
        avg_val_err /= k

        models_train_err.append(avg_train_err)
        models_val_err.append(avg_val_err)

    plt.figure()
    plt.plot(range(1, 9), models_train_err, color='blue', label='Train Loss')
    plt.plot(range(1, 9), models_val_err, color='red', linestyle='dashed', label='Validation Loss')
    plt.xlabel('Polynomial Order')
    plt.ylabel('Average MSE')
    plt.legend()
    plt.title('Error vs polynomial order - sklearn k-fold')
    plt.show()

    return models_train_err, models_val_err


def question_a():
    """
        Function for the part a of question 2.
        Reads all the csv files and computes the least squares regression and plots them.
    """
    df_linear = pd.read_csv('linear.csv')
    df_outlier = pd.read_csv('outlier.csv')
    df_poly = pd.read_csv('poly.csv')

    x_linear = df_linear['x:']
    x_outlier = df_outlier['x:']
    x_poly = df_poly['x:']
    y_linear = df_linear['y']
    y_outlier = df_outlier['y']
    y_poly = df_poly['y']

    coeffs_linear = least_squares_regression(x_linear, y_linear, 1)
    coeffs_outlier = least_squares_regression(x_outlier, y_outlier, 1)
    coeffs_poly = least_squares_regression(x_poly, y_poly, 1)

    plot_data_and_prediction(x_linear, y_linear, predict(coeffs_linear, x_linear), 'Linear')
    plot_data_and_prediction(x_outlier, y_outlier, predict(coeffs_outlier, x_outlier), 'Outlier')
    plot_data_and_prediction(x_poly, y_poly, predict(coeffs_poly, x_poly), 'Poly')

    plt.show()


def question_b():
    """
        Function for the part b of question 2.
    """
    linear = pd.read_csv('linear.csv')
    outlier = pd.read_csv('outlier.csv')
    poly = pd.read_csv('poly.csv')

    linear_train_errs, linear_val_errs = k_fold_cross_validation(linear, 5)
    outlier_train_errs, outlier_val_errs = k_fold_cross_validation(outlier, 5)
    poly_train_errs, poly_val_errs = k_fold_cross_validation(poly, 5)

    plot_errors(linear_train_errs, linear_val_errs, 'linear')
    plot_errors(outlier_train_errs, outlier_val_errs, 'outlier')
    plot_errors(poly_train_errs, poly_val_errs, 'poly')

    plt.show()

if __name__ == '__main__':
    question_a()
    question_b()
