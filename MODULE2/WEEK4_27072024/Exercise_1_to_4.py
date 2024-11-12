import numpy as np
import math

# Question 1


def compute_mean(x):
    return np.mean(x, axis=0)

# Question 2


def compute_median(x):
    x = np.sort(x)
    if len(x) % 2 == 1:
        return x[(len(x)+1)//2 - 1]
    else:
        return (1/2) * (x[((len(x))//2) - 1] + x[((len(X))//2) + 1 - 1])

# Question 3


def compute_std(x):
    mean = compute_mean(x)
    variance = (1/len(x))*(np.sum((x - mean)**2))
    std = math.sqrt(variance)
    return std

# Question 4


def compute_correlation_coefficient(x, y):
    n = len(x)
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    deviations_x = x - mean_x
    deviations_y = x - mean_y
    numerator = np.sum(deviations_x*deviations_y) / n
    denominator = compute_std(x)*compute_std(y)
    return numerator/denominator


X = np.asarray([-2, -5, -11, 6, 4, 15, 9])
Y = np.asarray([4, 25, 121, 36, 16, 225, 81])
print("Correlation : ", compute_correlation_coefficient(X, Y))
print(compute_mean(X))
print(compute_mean(Y))
print(compute_std(X))
print(compute_std(Y))
print(np.corrcoef(X, Y)[0][1])
print(compute_correlation_coefficient(X, Y))
