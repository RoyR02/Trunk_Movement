import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.5,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)

print(models)

