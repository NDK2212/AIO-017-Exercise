import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

# Thiết lập giá trị k và label_nums
k = 3
label_nums = 2

# Đặt seed để đảm bảo tính tái lập của số ngẫu nhiên
np.random.seed(10)

# Tạo đối tượng Generator
rng = np.random.default_rng(10)  # default_rng() sinh các số ngẫu nhiên

# Sinh dữ liệu ngẫu nhiên cho x_data (90 hàng, 4 cột)
x_data = rng.random((90, 4))

# In thử phần tử đầu tiên của x_data
print("x_data[0]:", x_data[0])

# Sinh dữ liệu ngẫu nhiên cho y_data (90 hàng, 1 cột) với giá trị trong phạm vi [0, label_nums)
y_data = rng.integers(0, label_nums, size=(90, 1))

# Sinh dữ liệu ngẫu nhiên cho x_test (30 hàng, 4 cột)
x_test = rng.random((30, 4))

# In thử dữ liệu x_test
print("x_test[0]:", x_test[0])

print(x_test[0])
distance = np.linalg.norm(
    np.abs(x_test[:, np.newaxis, :] - x_data[np.newaxis, :, :]), axis=2)
print(distance[0])
k_min_distance = np.argsort(distance, axis=1)[:, :k]
print(k_min_distance[0])
distance_to_idx = y_data[k_min_distance[:]].reshape(-1, 3)
print(distance_to_idx)
count_label = np.empty((30, 1), dtype=dict)
for i in range(30):
    count_label[i] = {np.count_nonzero(
        distance_to_idx[i] == j): j for j in range(label_nums)}
label = np.empty((30, 1), dtype=int)
print(dict(count_label[0, 0]).keys())
for i in range(30):
    label[i] = count_label[i, 0][np.max(list(dict(count_label[i, 0]).keys()))]
print(label)
