from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score

# Load iris dataset

import pandas as pd

import warnings
warnings.filterwarnings("ignore")


# Add any other classifiers you want to try here

# Load data
dataset=pd.read_csv("AD_train.csv")
X=dataset.drop(['Status'] , axis=1)
y=dataset['Status']
print (X)
print(y)

# Define the target variable and features
target = 'Status'
features = [col for col in dataset.columns if col != target]


# Standardize features
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Compute pairwise correlations between features
correlations = X.corr()
# Train a random forest classifier to estimate feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns)

# Print results
print("Pairwise feature correlations:")
print(correlations)
print("\nFeature importances:")
print(importances)
