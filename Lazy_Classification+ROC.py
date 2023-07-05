from lazypredict.Supervised import LazyClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# Load data
dataset=pd.read_csv("AD_train.csv")
X=dataset.drop(['Status'] , axis=1)
y=dataset['Status']
print (X)
print(y)

# encode target variable
le = LabelEncoder()
y = le.fit_transform(y.astype(str))
# Define the target variable and features
target = 'Status'
features = [col for col in dataset.columns if col != target]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.27 ,random_state =123)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)

# plot ROC curves
plt.figure(figsize=(8, 8))
plt.plot([0, 1], [0, 1], 'k--')
for name, y_pred in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='{} ({:.2f})'.format(name, roc_auc))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic curve')
plt.legend(loc='best')
plt.show()

print(models)