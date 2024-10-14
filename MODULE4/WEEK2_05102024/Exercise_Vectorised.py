import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

data = np.genfromtxt(
    'MODULE4\WEEK2_05102024\Advertising copy.csv', delimiter=',', skip_header=1)
print(data.shape)
num_samples = data.shape[0]
X = data[:, :-1]
y = data[:, -1]


def mean_normalisation(X):  # min_max scaler
    N = X.shape[0]
    maxx = np.max(X)
    minn = np.min(X)
    avg = np.mean(X)
    X_normalised = (X - avg) / (maxx - minn)
    X_b = np.c_[np.ones((N, 1)), X_normalised]
    return X_b


def predict(x, theta):
    return x@theta


def compute_loss_mse(y, y_hat):
    return np.mean(((y_hat - y)**2)/2)


def compute_gradient(x, y, y_hat):
    k = (y_hat - y)
    return (x.T)@k / x.shape[0]


def update_theta(theta, dl_dtheta, lr):
    theta = theta - dl_dtheta*lr
    return theta


def stochastic_gradient_descent(X_b, y, n_epochs=50, learning_rate=0.00001):
    # thetas = np.random.rand(4,1) # real application
    thetas = np.array([[1.16270837], [-0.81960489],
                      [1.39501033], [0.29763545]])
    thetas_path = [thetas]
    losses = []
    N = X_b.shape[0]

    for epoch in range(n_epochs):
        for i in range(N):
            # select random number in N
            # random_index = np.random.randint(N) # In real application, you should use this code
            random_index = i  # for this assignment only
            xi = X_b[random_index:random_index+1]
            yi = y[random_index:random_index+1]

            yi_hat = predict(xi, thetas)
            loss = compute_loss_mse(yi, yi_hat)
            dl_dthetas = compute_gradient(xi, yi, yi_hat)
            thetas = update_theta(thetas, dl_dthetas, learning_rate)
            thetas_path.append(thetas)
            losses.append(loss)
    return thetas_path, losses


X_b = mean_normalisation(X)
sgd_theta, losses = stochastic_gradient_descent(
    X_b, y, n_epochs=50, learning_rate=0.01)
print(np.sum(losses))
x_axis = list(range(500))
plt.plot(x_axis, losses[:500], color="r")
plt.show()


def mini_batch_gradient_descent(X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01):
    # thetas = np.random.rand(4,1) # real application
    thetas = np.asarray([[1.16270837], [-0.81960489],
                        [1.39501033], [0.29763545]])
    thetas_path = [thetas]
    losses = []
    N = X_b.shape[0]

    for epoch in range(n_epochs):
        shuffled_indices = np.asarray([21, 144, 17, 107, 37, 115, 167, 31, 3, 132, 179, 155, 36, 191, 182, 170, 27, 35, 162, 25, 28, 73, 172, 152, 102, 16, 185, 11, 1, 34, 177, 29, 96, 22, 76, 196, 6, 128, 114, 117, 111, 43, 57, 126, 165, 78, 151, 104, 110, 53, 181, 113, 173, 75, 23, 161, 85, 94, 18, 148, 190, 169, 149, 79, 138, 20, 108, 137, 93, 192, 198, 153, 4, 45, 164, 26, 8, 131, 77, 80, 130, 127, 125, 61, 10, 175, 143, 87, 33, 50, 54, 97, 9, 84, 188, 139, 195,
                                      72, 64, 194, 44, 109, 112, 60, 86, 90, 140, 171, 59, 199, 105, 41, 147, 92, 52, 124, 71, 197, 163, 98, 189, 103, 51, 39, 180, 74, 145, 118, 38, 47, 174, 100, 184, 183, 160, 69, 91, 82, 42, 89, 81, 186, 136, 63, 157, 46, 67, 129, 120, 116, 32, 19, 187, 70, 141, 146, 15, 58, 119, 12, 95, 0, 40, 83, 24, 168, 150, 178, 49, 159, 7, 193, 48, 30, 14, 121, 5, 142, 65, 176, 101, 55, 133, 13, 106, 66, 99, 68, 135, 158, 88, 62, 166, 156, 2, 134, 56, 123, 122, 154])
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, N, minibatch_size):
            xi = X_b_shuffled[i:i+minibatch_size]
            yi = y_shuffled[i:i+minibatch_size]

            yi_hat = predict(xi, thetas)
            loss = compute_loss_mse(yi, yi_hat)
            dl_dthetas = compute_gradient(xi, yi, yi_hat)
            thetas = update_theta(thetas, dl_dthetas, learning_rate)
            thetas_path.append(thetas)
            losses.append(loss)
    return thetas_path, losses


X_b = mean_normalisation(X)
mbgd_thetas, losses = mini_batch_gradient_descent(
    X_b, y, n_epochs=50, minibatch_size=20, learning_rate=0.01)
x_axis = list(range(200))
print(round(np.sum(losses), 2))
plt.plot(x_axis, losses[:200], color='r')
plt.show()


def batch_gradient_descent(X_b, y, n_epochs=100, learning_rate=0.01):
    # thetas = np.random.rand(4,1) # real application
    thetas = np.asarray([[1.16270837], [-0.81960489],
                        [1.39501033], [0.29763545]])
    thetas_path = [thetas]
    losses = []
    N = X_b.shape[0]

    for epoch in range(n_epochs):
        yi_hat = predict(X_b, thetas)
        loss = compute_loss_mse(y, yi_hat)
        dl_dthetas = compute_gradient(X_b, y, yi_hat)
        thetas = update_theta(thetas, dl_dthetas, learning_rate)
        thetas_path.append(thetas)
        losses.append(loss)
    return thetas_path, losses


X_b = mean_normalisation(X)
bgd_thetas, losses = batch_gradient_descent(
    X_b, y, n_epochs=100, learning_rate=0.01)

# in loss cho 100 sample đầu
x_axis = list(range(100))
plt.plot(x_axis, losses[:100], color="r")
plt.show()
