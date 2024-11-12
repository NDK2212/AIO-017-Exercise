# 3
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


def implement_linear_regression_nsamples(x_data, y_data, epoch_max=50, lr=1e-5):
    losses = []
    w1, w2, w3, b = initialize_params()
    N = len(y_data)
    for _ in range(epoch_max):

        loss_total = 0.0
        dw1_total = 0.0
        dw2_total = 0.0
        dw3_total = 0.0
        db_total = 0.0
        for i in range(N):
            x1 = x_data[0][i]
            x2 = x_data[1][i]
            x3 = x_data[2][i]
            y = y_data[i]
            y_hat = predict(x1, x2, x3, w1, w2, w3, b)
            loss = compute_loss_mse(y, y_hat)
            loss_total += loss
            dl_dw1 = compute_gradient_wi(x1, y, y_hat)
            dl_dw2 = compute_gradient_wi(x2, y, y_hat)
            dl_dw3 = compute_gradient_wi(x3, y, y_hat)
            dl_db = compute_gradient_b(y, y_hat)
            dw1_total += dl_dw1
            dw2_total += dl_dw2
            dw3_total += dl_dw3
            db_total += dl_db
        w1 = w1 - (dw1_total / N) * lr
        w2 = w2 - (dw2_total / N) * lr
        w3 = w3 - (dw3_total / N) * lr
        b = b - (db_total / N) * lr
        losses.append(loss_total / N)
    return (w1, w2, w3, b, losses)


file_path = 'MODULE4\WEEK1_28092024\q_advertising.csv'
X, y = prepare_data(file_path)
(w1, w2, w3, b, losses) = implement_linear_regression_nsamples(
    X, y, epoch_max=50, lr=1e-5)
print(losses)
print(w1, w2, w3)
plt.plot(losses)
plt.xlabel("#epoch")
plt.ylabel("MSE Loss")
plt.show()
