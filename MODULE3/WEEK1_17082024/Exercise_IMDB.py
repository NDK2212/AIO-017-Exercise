import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1: Read data
dataset_path = 'MODULE3\WEEK1_17082024\IMDB-Movie-Data.csv'
data = pd.read_csv(dataset_path)
# 2: View the data
print(data.head(5))
print(data.tail(5))

# 3: Understand some basic information about the data
data.info()
data.describe()
# 4: Data Selection - Indexing and Slicing data
print(data.columns)
director_series = data['Director']
print(type(director_series))
print(director_series)
director_df = data[['Director']]
print(type(director_df))
print(director_df)
# truyền vào data trong ngoặc vuông là 1 list of columns => tạo dataframe (dù chỉ có 1 column cũng thế)
#                                     tên 1 column => tạo series
some_cols = data[['Title', 'Rating', 'Revenue (Millions)']]
print(some_cols)
data.loc[0:2, 'Votes':'Metascore']
# 5: Data Selection – Based on Conditional filtering
filtered_data_1 = data[(data['Year'] == 2016) & (data['Rating'] > 6.0)]
print(filtered_data_1.shape)
# => có 202 phim sản xuất vào năm 2016 và được RATING > 6.0
# lấy 10% những phim có rating cao nhất
filtered_data_2 = data[data['Rating'] >= data['Rating'].quantile(0.9)]
print(filtered_data_2)
data.nlargest(n=5, columns='Rating')
# 6: Groupby Operations
data.groupby('Genre')[['Rating', 'Metascore']].min()
# 7: Sorting operation
data.groupby('Director')[['Rating']].mean().sort_values(
    ['Rating'], ascending=False).head()
# 8: View missing values
data.isnull().sum()
# 9: Deal with missing values - Deleting
# xóa cột, thêm inplace bằng True thì lệnh mới được thực hiện
data.drop('Metascore', axis=1).head()

# xóa hàng, thêm inplace bằng True thì lệnh mới được thực hiện
print(data.shape)
data.dropna()
print(data.shape)
# 10: Deal with missing values - Filling
metascore_mean = data['Metascore'].mean()
data['Metascore'].fillna(metascore_mean, inplace=True)
print(data.isnull().sum())
data[data.isnull().any(axis=1)]
# 11: apply


def classify_revenue(revenue):
    if revenue >= 300:
        return 'bomb'
    elif revenue >= 150:
        return 'hit'
    elif revenue >= 100:
        return 'normal'
    else:
        return 'low'


data['revenue_level'] = data['Revenue (Millions)'].apply(classify_revenue)
print(data.head())
