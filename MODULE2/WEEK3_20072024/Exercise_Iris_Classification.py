import numpy as np
import math


def compute_len_class0(x):
    return (1/(math.sqrt(math.pi*2*variance_for_class0)))*math.exp(-((x - mean_for_class0)**2) / (2*(variance_for_class0)))


def compute_len_class1(x):
    return (1/(math.sqrt(math.pi*2*variance_for_class1)))*math.exp(-((x - mean_for_class1)**2) / (2*(variance_for_class1)))


data = np.array([
    [1.4, 0],
    [1.0, 0],
    [1.3, 0],
    [1.9, 0],
    [2.0, 0],
    [1.8, 0],
    [3.0, 1],
    [3.8, 1],
    [4.1, 1],
    [3.9, 1],
    [4.2, 1],
    [3.4, 1]
])

data_for_class0 = data[np.nonzero(data[:, 1] == 0)]
print(data_for_class0)
mean_for_class0 = np.sum(
    data_for_class0[:, 0], axis=0) / len(data_for_class0[:, 0])
print(mean_for_class0)
variance_for_class0 = (1/len(data_for_class0[:, 0])) * np.sum(
    (data_for_class0[:, 0] - mean_for_class0)**2, axis=0)
print(variance_for_class0)

data_for_class1 = data[np.nonzero(data[:, 1] == 1)]
print(data_for_class1)
mean_for_class1 = np.sum(
    data_for_class1[:, 0], axis=0) / len(data_for_class1[:, 0])
print(mean_for_class1)
variance_for_class1 = (1/len(data_for_class1[:, 0])) * np.sum(
    (data_for_class1[:, 0] - mean_for_class1)**2, axis=0)
print(variance_for_class1)
