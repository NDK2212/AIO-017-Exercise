import numpy as np
import math


def create_train_data():
    data_np = np.array([["Sunny", "Hot", "High", "Weak", "No"],
                        ["Sunny", "Hot", "High", "Strong", "No"],
                        ["Overcast", "Hot", "High", "Weak", "Yes"],
                        ["Rain", "Mild", "High", "Weak", "Yes"],
                        ["Rain", "Cool", "Normal", "Weak", "Yes"],
                        ["Rain", "Cool", "Normal", "Strong", "No"],
                        ["Overcast", "Cool", "Normal", "Strong", "Yes"],
                        ["Overcast", "Mild", "High", "Weak", "No"],
                        ["Sunny", "Cool", "Normal", "Weak", "Yes"],
                        ["Rain", "Mild", "Normal", "Weak", "Yes"]
                        ])
    return data_np


def compute_prior_probability(train_data):
    y_unique = ['No', 'Yes']
    prior_probability = np.zeros(len(y_unique))
    prior_probability[0] = train_data[np.nonzero(
        train_data[:, 4] == y_unique[0])].shape[0] / train_data.shape[0]
    prior_probability[1] = train_data[np.nonzero(
        train_data[:, 4] == y_unique[1])].shape[0] / train_data.shape[0]
    return prior_probability


def compute_conditional_probability(train_data):
    y_unique = ['No', 'Yes']
    conditional_probability = []
    list_x_name = []

    for i in range(train_data.shape[1] - 1):
        x_unique = np.unique(train_data[:, i])
        list_x_name.append(x_unique)
        x_conditional_probability = np.zeros((len(y_unique), len(x_unique)))
        for result in range(len(y_unique)):
            for feature in range(len(x_unique)):
                x_conditional_probability[result, feature] = train_data[np.nonzero((train_data[:, i] == x_unique[feature]) & (
                    train_data[:, 4] == y_unique[result]))].shape[0] / train_data[np.nonzero(train_data[:, 4] == y_unique[result])].shape[0]
        conditional_probability.append(x_conditional_probability)
    return conditional_probability, list_x_name


def get_index_from_value(feature_name, list_features):
    return np.nonzero(list_features == feature_name)[0][0]


def train_naive_bayes(train_data):
    prior_probability = compute_prior_probability(train_data)
    conditional_probability, list_x_name = compute_conditional_probability(
        train_data)
    return prior_probability, conditional_probability, list_x_name


def prediction_play_tennis(x, list_x_name, prior_probability, conditional_probability):
    x1 = get_index_from_value(x[0], list_x_name[0])
    x2 = get_index_from_value(x[1], list_x_name[1])
    x3 = get_index_from_value(x[2], list_x_name[2])
    x4 = get_index_from_value(x[3], list_x_name[3])

    p0 = prior_probability[0] * conditional_probability[0][0, x1] * conditional_probability[1][0,
                                                                                               x2] * conditional_probability[2][0, x3] * conditional_probability[3][0, x4]
    p1 = prior_probability[1] * conditional_probability[0][1, x1] * conditional_probability[1][1,
                                                                                               x2] * conditional_probability[2][1, x3] * conditional_probability[3][1, x4]

    if p0 > p1:
        y_pred = 0
    else:
        y_pred = 1

    return y_pred


train_data = create_train_data()
print(train_data)
prior_probability = compute_prior_probability(train_data)
print("P(play tennis = No):", prior_probability[0])
print("P(play tennis = Yes):", prior_probability[1])
conditional_probability, list_x_name = compute_conditional_probability(
    train_data)
print("Conditional_probability:", conditional_probability)
print("List_x_name:", list_x_name)
print("x1 = ", list_x_name[0])
print("x2 = ", list_x_name[1])
print("x3 = ", list_x_name[2])
print("x4 = ", list_x_name[3])
outlook = list_x_name[0]
i1 = get_index_from_value("Overcast", outlook)
i2 = get_index_from_value("Rain", outlook)
i3 = get_index_from_value("Sunny", outlook)
print(i1, i2, i3)
x1 = get_index_from_value("Sunny", list_x_name[0])
print("P('Outlook'= 'Sunny'| 'Play Tennis' = 'Yes') = ",
      np.round(conditional_probability[0][1, x1], 2))
print("P('Outlook'= 'Sunny'| 'Play Tennis' = 'No') = ",
      np.round(conditional_probability[0][0, x1], 2))

x = ['Sunny', 'Cool', 'High', 'Strong']
pred = prediction_play_tennis(
    x, list_x_name, prior_probability, conditional_probability)
if pred:
    print("Ad should go!")
else:
    print("Ad should not go!")
