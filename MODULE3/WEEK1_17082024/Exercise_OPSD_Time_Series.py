import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

opsd = pd.read_csv(
    'MODULE3\WEEK1_17082024\opsd_germany_daily.csv')
opsd.set_index('Date', inplace=True)
print(opsd.head())
opsd = pd.read_csv(
    'MODULE3\WEEK1_17082024\opsd_germany_daily.csv', index_col=0, parse_dates=True)

opsd['Year'] = opsd.index.year
opsd['Month'] = opsd.index.month_name()  # opsd.index.month
opsd['Weekday Name'] = opsd.index.day_name()
opsd.sample(5, random_state=0)
opsd.loc['2017-08-10':'2017-08-15']
opsd.loc['2012-02']
opsd['2012':'2014']
sns.set(rc={'figure.figsize': (11, 4)})
opsd['Consumption'].plot(linewidth=0.5)
opsd.loc['2016', 'Consumption'].plot()
cols_plot = ['Consumption', 'Solar', 'Wind']
axes = opsd[cols_plot].plot(marker='.', alpha=0.5,
                            linestyle='None', figsize=(11, 9), subplots=True)
for ax in axes:
    ax.set_ylabel('Daily Totals (GWh)')
plt.show()
# Seasonality
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True)
for name, ax in zip(['Consumption', 'Wind', 'Solar'], axes):
    sns.boxplot(data=opsd, x='Month', y=name, ax=ax)
#   ax.set_ylabel('GWh')
#   ax.set_title(name)
#   if ax != axes[-1]:
#     ax.set_xlabel('')
# plt.show()
# Frequencies
# freq: bước nhảy, ví dụ 'W': week,'h':giờ,'Q': quý,'M': tháng, 'YE':year, 'B': working day
pd.date_range('2024-08-25', '2024-09-06', freq='B')
time_sample = pd.to_datetime(['2014-02-02', '2014-02-05', '2014-02-08'])
consume_sample = opsd.loc[time_sample, 'Consumption'].copy()
print(consume_sample)
consume_freq = consume_sample.asfreq('D')
print(consume_freq)
consume_freq = consume_sample.asfreq('D', method='ffill')
print(consume_freq)
consume_freq = consume_sample.asfreq('D', method='bfill')
print(consume_freq)
# Resampling
data_columns = ['Consumption', 'Wind', 'Solar', 'Wind+Solar']
opsd_weekly_mean = opsd[data_columns].resample('Y').mean()
print(opsd_weekly_mean)
# Rolling windows
opsd[data_columns].rolling(7, center=True).mean()
# Trends
opsd_7d = opsd[data_columns].rolling(7, center=True).mean()

# Calculate 365-day rolling mean
opsd_365d = opsd[data_columns].rolling(
    window=365,
    center=True,
    min_periods=360
).mean()


# Assuming `apsd_daily`, `apsd_7d`, and `apsd_365d` are your DataFrames or Series containing the data
fig, ax = plt.subplots()

# Plot daily, 7-day rolling mean, and 365-day rolling mean time series
ax.plot(opsd["Consumption"], marker='.',
        markersize=0.6, linestyle='None', label='Daily')
ax.plot(opsd_7d["Consumption"], linewidth=2, label='7-d Rolling Mean')
ax.plot(opsd_365d["Consumption"], color='0.2',
        linewidth=3, label='Trend (365-d Rolling Mean)')

# Set x-ticks to yearly interval and add legend and labels
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.legend()
ax.set_xlabel('Year')
ax.set_ylabel('Consumption (GWh)')
ax.set_title('Trends in Electricity Consumption')

plt.show()
