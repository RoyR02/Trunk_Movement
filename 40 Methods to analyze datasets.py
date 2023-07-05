import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Define the features and target
X = df.drop(['Status'], axis=1)
y = df['Status']

# Label encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# 1. Get basic statistics of the numerical features
basic_stats = X.describe()
print("Basic statistics of numerical features:\n", basic_stats)

# 2. Get the number of missing values in each column
missing_values = X.isna().sum()
print("Number of missing values in each column:\n", missing_values)

# 3. Get the correlation matrix
corr_matrix = X.corr()
print("Correlation matrix:\n", corr_matrix)

# 4. Get the covariance matrix
cov_matrix = X.cov()
print("Covariance matrix:\n", cov_matrix)

# 5. Get the skewness of each column
skewness = X.skew()
print("Skewness of each column:\n", skewness)

# 6. Get the kurtosis of each column
kurtosis = X.kurtosis()
print("Kurtosis of each column:\n", kurtosis)

# 7. Get the number of unique values in each column
num_unique = X.nunique()
print("Number of unique values in each column:\n", num_unique)

# 8. Get the mode of each column
mode = X.mode()
print("Mode of each column:\n", mode)

# 9. Get the median of each column
median = X.median()
print("Median of each column:\n", median)

# 10. Get the mean of each column
mean = X.mean()
print("Mean of each column:\n", mean)

# 11. Get the standard deviation of each column
std_dev = X.std()
print("Standard deviation of each column:\n", std_dev)

# 12. Get the variance of each column
variance = X.var()
print("Variance of each column:\n", variance)

# 13. Get the minimum value of each column
minimum = X.min()
print("Minimum value of each column:\n", minimum)

# 14. Get the maximum value of each column
maximum = X.max()
print("Maximum value of each column:\n", maximum)

# 15. Get the range of each column
range_val = X.max() - X.min()
print("Range of each column:\n", range_val)

# 16. Get the interquartile range of each column
interquartile_range = X.quantile(0.75) - X.quantile(0.25)
print("Interquartile range of each column:\n", interquartile_range)

# 17. Get the z-score of each value
z_score = (X - X.mean()) / X.std()
print("Z-score of each value:\n", z_score)

# 18. Get the percentage change between consecutive values
percentage_change = X.pct_change()
print("Percentage change between consecutive values:\n", percentage_change)

# 19. Get the cumulative sum of each column
cumulative_sum = X.cumsum()
print("Cumulative sum of each column:\n", cumulative_sum)

# 20. Get the cumulative product of each column
cumulative_prod = X.cumprod()
print("Cumulative product of each column:\n", cumulative_prod)

# 21. Get the rolling mean of each column
rolling_mean = X.rolling(window=3).mean()
print("Rolling mean of each column:\n", rolling_mean)

# 22. Get the exponential moving average of each column
ema = X.ewm(span=3).mean()
print('Exponential Moving Average:')
print(ema)

# 23. Get the correlation between each pair of columns
corr = X.corr()
print('Correlation Matrix:')
print(corr)

# 24. Get the covariance between each pair of columns
cov = X.cov()
print('Covariance Matrix:')
print(cov)

# 25. Get the number of unique values in each column
nunique = X.nunique()
print('Number of Unique Values in Each Column:')
print(nunique)

# 26. Get the counts of each unique value in each column
counts = X.apply(pd.Series.value_counts)
print('Counts of Unique Values in Each Column:')
print(counts)

# 27. Get the skewness and kurtosis of each column
skew_kurtosis = X.aggregate(['skew', 'kurtosis'])
print('Skewness and Kurtosis of Each Column:')
print(skew_kurtosis)

# 28. Get the histogram of each column
fig1, ax1 = plt.subplots(figsize=(10,10))
X.hist(ax=ax1)
ax1.set_title('Histogram of Each Column')
plt.show()

# 29. Get the boxplot of each column
fig2, ax2 = plt.subplots(figsize=(10,10))
X.boxplot(ax=ax2)
ax2.set_title('Boxplot of Each Column')
plt.show()

# 30. Get the violin plot of each column
fig3, ax3 = plt.subplots(figsize=(10,10))
sns.violinplot(data=X, inner="stick", palette="pastel", ax=ax3)
ax3.set_title('Violin Plot of Each Column')
plt.show()

# 31. Get the bar plot of each column
fig4, ax4 = plt.subplots(figsize=(10,10))
X.plot(kind='bar', ax=ax4)
ax4.set_title('Bar Plot of Each Column')
plt.show()

# 32. Get the stacked bar plot of each column
fig5, ax5 = plt.subplots(figsize=(10,10))
X.plot(kind='bar', stacked=True, ax=ax5)
ax5.set_title('Stacked Bar Plot of Each Column')
plt.show()

# 33. Get the line plot of each column
fig6, ax6 = plt.subplots(figsize=(10,10))
X.plot(kind='line', ax=ax6)
ax6.set_title('Line Plot of Each Column')
plt.show()

# 34. Get the area plot of each column
fig7, ax7 = plt.subplots(figsize=(10,10))
X.plot(kind='area', ax=ax7)
ax7.set_title('Area Plot of Each Column')
plt.show()

# 35. Get the scatter plot of each pair of columns
fig8 = sns.pairplot(X)
fig8.fig.suptitle('Scatter Plot of Each Pair of Columns')
plt.show()

#38. Get the heatmap of each column
heatmap = sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
heatmap.set_title('Heatmap')
plt.show()

