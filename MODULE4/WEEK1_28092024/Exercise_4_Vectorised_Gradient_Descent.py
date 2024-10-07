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

    X = [[1, x1, x2, x3]
         for x1, x2, x3 in zip(tv_data, radio_data, newspaper_data)]
    y = sales_data
    return X, y


def initialise_params():
    # In reality
    # bias = 0
    # w1 = random.gauss(mu=0.0, sigma=0.01)
    # w2 = random.gauss(mu=0.0, sigma=0.01)
    # w3 = random.gauss(mu=0.0, sigma=0.01)
    return [0, -0.01268850433497871, 0.004752496982185252, 0.0073796171538643845]


def predict(x_features, weights):
    result = np.dot(x_features, weights)
    return result


def compute_loss(y_hat, y):
    return (y_hat - y)**2


def compute_gradient_w(x_features, y, y_hat):
    return 2*np.dot(x_features, (y_hat-y))


def update_weight(weights, dl_dweights, lr):
    weights = weights - dl_dweights*lr
    return weights


def implement_linear_regression(x_feature, y_output, epoch_max=50, lr=1e-5):
    losses = []
    weights = initialise_params()
    N = len(y_output)
    for epoch in range(epoch_max):
        print("epoch", epoch)
        for i in range(N):
            features_i = x_feature[i]
            y = y_output[i]

            y_hat = predict(features_i, weights)
            loss = compute_loss(y_hat, y)
            dl_dweights = compute_gradient_w(features_i, y, y_hat)
            weights = update_weight(weights, dl_dweights, lr)
            losses.append(loss)
    return weights, losses


file_path = 'MODULE4\WEEK1_28092024\q_advertising.csv'
X, y = prepare_data(file_path)
W, L = implement_linear_regression(X, y)
print(L[9999])
plt.plot(L[0:100])
plt.xlabel('#Iteration')
plt.ylabel('Loss')
plt.show()
