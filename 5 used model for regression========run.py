# Import required libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor

import time

# Load the training dataset
df_train = pd.read_csv('AD_train.csv')

# Define the features and target for training
X_train = df_train.drop(['Status'], axis=1)
y_train = df_train['Status']

# Label encode the target variable for training
le = LabelEncoder()
y_train = le.fit_transform(y_train)

# Scale the features using StandardScaler for training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Define the 5 regression models
models = [ElasticNetCV(), RandomForestRegressor(), SVR(kernel='rbf'), BaggingRegressor(), KNeighborsRegressor()]

# Train and evaluate each model using KFold cross-validation
for model in models:
    start_time = time.time()
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_list, mae_list, r2_list = [], [], []
    for train_index, test_index in kfold.split(X_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        mse_list.append(mean_squared_error(y_test_fold, y_pred))
        mae_list.append(mean_absolute_error(y_test_fold, y_pred))
        r2_list.append(r2_score(y_test_fold, y_pred))
    end_time = time.time()

    # Print the model performance metrics
    print(f"Model: {model.__class__.__name__}")
    print(f"Mean squared error: {sum(mse_list)/len(mse_list):.4f}")
    print(f"Mean absolute error: {sum(mae_list)/len(mae_list):.4f}")
    print(f"R-squared score: {sum(r2_list)/len(r2_list):.4f}")
    print(f"Training time: {end_time-start_time:.4f} seconds\n")
