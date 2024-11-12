import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random


def get_column(data, index):
    result = data[:, index]
    return result


def prepare_data(file_name_dataset):
    data = np.genfromtxt(file_name_dataset, delimiter=',',
                         skip_header=1).tolist()
    data = np.array(data)
    tv_data = get_column(data, 0)
    radio_data = get_column(data, 1)
    newspaper_data = get_column(data, 2)
    sales_data = get_column(data, 3)
    X = np.array([tv_data, radio_data, newspaper_data])
    y = np.array(sales_data)
    return X, y


def initialize_params():
    # In reality
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    # b = 0
    w1, w2, w3, b = (0.016992259082509283,
                     0.0070783670518262355, -0.002307860847821344, 0)
    return w1, w2, w3, b


def predict(x1, x2, x3, w1, w2, w3, b):
    return x1*w1 + x2*w2 + x3*w3 + b


def compute_loss_mse(y, y_hat):
    loss = (y_hat - y)**2
    return loss


def compute_loss_mae(y, y_hat):
    return abs(y_hat-y)


def compute_gradient_wi(xi, y, y_hat):
    return 2*xi*(y_hat-y)


def compute_gradient_b(y, y_hat):
    return 2*(y_hat - y)


def update_weight_wi(wi, dl_dwi, lr):
    w = wi - dl_dwi*lr
    return w


def update_weight_b(b, dl_db, lr):
    b = b - dl_db*lr
    return b


def implement_linear_regression(x_data, y_data, epoch_max=50, lr=1e-5):
    # Stochastic Gradient Descent
    losses = []
    w1, w2, w3, b = initialize_params()
    N = len(y_data)
    for _ in range(epoch_max):
        for i in range(N):
            x1 = x_data[0][i]
            x2 = x_data[1][i]
            x3 = x_data[2][i]

            y = y_data[i]
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)

            loss = compute_loss_mse(y, y_hat)

            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)

            w1 = update_weight_wi(w1, dl_dw1, lr)
            w2 = update_weight_wi(w2, dl_dw2, lr)
            w3 = update_weight_wi(w3, dl_dw3, lr)
            b = update_weight_b(b, dl_db, lr)

            losses.append(loss)
    return (w1, w2, w3, b, losses)


file_path = 'MODULE4\WEEK1_28092024\q_advertising.csv'
X, y = prepare_data(file_path)
listt = [sum(X[0][:5]), sum(X[1][:5]), sum(X[2][:5]), sum(y[:5])]
print(listt)
print(compute_gradient_b(y=2.0, y_hat=0.5))
print(update_weight_wi(wi=1.0, dl_dwi=-0.5, lr=1e-5))
print(update_weight_b(b=0.5, dl_db=-1.0, lr=1e-5))
(w1, w2, w3, b, losses) = implement_linear_regression(X, y)
print(w1, w2, w3)
plt.plot(losses[:100])
plt.xlabel("#iteration")
plt.ylabel("Loss")
plt.show()
