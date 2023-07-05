# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import time

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Define the features and target
X = df.drop(['Status'], axis=1)
y = df['Status']

# Label encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Normalize the features
X_train = normalize(X_train)
X_test = normalize(X_test)

# Initialize LazyClassifier model
clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)

# Fit the model and calculate time consumed
start_time = time.time()
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
end_time = time.time()

# Print the results
print(models)
print("Time consumed:", end_time-start_time)