import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import math


def compute_mean(x):
    return np.mean(x, axis=0)


def compute_std(x):
    mean = compute_mean(x)
    variance = (1/len(x))*(np.sum((x - mean)**2))
    std = math.sqrt(variance)
    return std
# Hàm tính toán hệ số tương quan


def correlation(x, y):
    X = np.array(x)
    Y = np.array(y)
    numerator = compute_mean((X - compute_mean(X))*(Y - compute_mean(Y)))
    denominator = compute_std(X)*compute_std(Y)
    return numerator / denominator


# Question 5
data = pd.read_csv("MODULE2\WEEK4_27072024\one_advertising.csv")
x = data['TV']
y = data['Radio']
corr_xy = correlation(x, y)
print(f"Correlation between TV and Radio: {round(corr_xy, 2)}")

# Question 6
features = ['TV', 'Radio', 'Newspaper']

for feature_1 in features:
    for feature_2 in features:
        correlation_value = correlation(data[feature_1], data[feature_2])
        print(
            f"Correlation between {feature_1} and {feature_2}: {round(correlation_value, 2)}")

# Question 7
x = data['Radio']
y = data['Newspaper']
print(np.corrcoef(x, y))
print(data.corr())


# Question 09
plt.figure(figsize=(10, 8))
data_corr = data.corr()
sns.heatmap(data_corr, annot=False, fmt=".2f", linewidth=.5)
plt.show()
