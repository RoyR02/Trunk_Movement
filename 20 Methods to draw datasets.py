import pandas as pd
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


# 1. Pair plot
sns.pairplot(df, hue='Status')
plt.title('Pair plot')
plt.show()

# 2. Box plot
df.boxplot(by='Status', figsize=(10,10))
plt.title('Box plot')
plt.show()

# 3. Correlation heatmap
corr = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, cmap='coolwarm', annot=True)
plt.title('Correlation heatmap')
plt.show()

# 8. Violin plot
sns.violinplot(X='Status', y='age', data=df)
plt.title('Violin plot')
plt.show()


# 14. Pie chart
df['Status'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie chart')
plt.show()



# 18. Radar chart
categories = list(df.columns)[1:]
values = df.iloc[0][1:].tolist()
values += values[:1]
angles = [n / float(len(categories)) * 2 * 3.141 for n in range(len(categories))]
angles += angles[:1]
ax = plt.subplot(111, polar=True)
ax.plot(angles, values, linewidth=1, linestyle='solid')
ax.fill(angles, values, 'b', alpha=0.1)
ax.set_thetagrids(angles[:-1], categories)
ax.set_title('Radar chart', fontsize=20)
plt.title('Radar chart')
plt.show()

#

# 22. Parallel coordinate plot
from pandas.plotting import parallel_coordinates
parallel_coordinates(df, 'Status')
plt.title('Parallel coordinate plot')
plt.show()

# 23. Andrews curves
from pandas.plotting import andrews_curves
andrews_curves(df, 'Status')
plt.title('Andrews curves')
plt.show()



# 24. Heatmap with dendrogram
sns.clustermap(df.corr(), cmap='coolwarm', standard_scale=1)
plt.show()
