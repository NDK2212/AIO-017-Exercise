import pandas as pd
import numpy as np

# Read the CSV file into a DataFrame
df = pd.read_csv(
    'C:\\Users\\Admin\\Desktop\\AIO2024\\AIO-017-Exercise\\MODULE2\\WEEK1_06072024\\Advertising.csv'
)

# Convert the DataFrame to a NumPy array
dataa = df.to_numpy()
print(dataa.shape)

# 15: Find the maximum sale value
max_sale_value = dataa.max(axis=0)[3]
print(max_sale_value)
print(np.nonzero(dataa[:, 3] == max_sale_value))  # Use np.nonzero

# 16: Calculate the average of the first column
print(np.sum(dataa, axis=0)[0] / dataa.shape[0])

# 17: Count the number of sales values greater than or equal to 20
sales_upper_20 = (dataa[:, 3] >= 20).sum()
print(sales_upper_20)

# 18: Filter data and calculate the average of the first column where sales are >= 15
data = dataa[:, [1, 3]]
print(data)
data1 = data[np.nonzero(data[:, 1] >= 15)]  # Use np.nonzero
data1 = data1[:, 0]
result = data1.sum()
countt = len(data1)
print(result / countt)

# 19: Calculate the average of the second column and sum of the third column
news = np.sum(dataa[:, 2], axis=0) / dataa.shape[0]
print(news)
neww = dataa[:, [2, 3]]
neww1 = neww[np.nonzero(neww[:, 0] > news)]  # Use np.nonzero
neww1 = neww1[:, 1]
print(neww1.sum())

# 20: Categorize sales as Good, Average, or Bad
A = np.sum(dataa[:, 3], axis=0) / dataa.shape[0]
print(A)
ne = dataa[:, 3]
new_arr = np.array(ne, dtype=str)
new_arr[np.nonzero(ne > A)] = 'Good'
new_arr[np.nonzero(ne == A)] = 'Average'
new_arr[np.nonzero(ne < A)] = 'Bad'
print(new_arr[7:10])

# 21: Categorize sales relative to the closest average
A = np.sum(dataa[:, 3], axis=0) / dataa.shape[0]
print(A)
na = dataa[:, 3]
nb = abs(na - A)
nb.sort()
B = nb[0]
if len(np.nonzero(na == A - B)[0]) != 0:  # Use np.nonzero
    B = A - nb[0]
elif len(np.nonzero(na == A + B)[0]) != 0:  # Use np.nonzero
    B = A + nb[0]
ne = dataa[:, 3]
new_arr = np.array(ne, dtype=str)
new_arr[np.nonzero(ne > B)] = 'Good'
new_arr[np.nonzero(ne == B)] = 'Average'
new_arr[np.nonzero(ne < B)] = 'Bad'
print(new_arr[7:10])
