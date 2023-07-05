# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import time

import warnings
warnings.filterwarnings("ignore")

# Load the dataset
df = pd.read_csv('AD_train.csv')

# Preprocess the dataset
X = df.drop('Status', axis=1)
y = df['Status']

# One-hot encode the 'Status' column
y_encoded = pd.get_dummies(y)

# Convert y_encoded back to a 1D array
y = y_encoded.values.argmax(axis=1)

# Normalize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the objective function for Optuna
def objective(trial):
    # Define the parameters to optimize
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 100)
    activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
    alpha = trial.suggest_loguniform('alpha', 1e-5, 1e-2)
    
    # Create the classifier with the suggested parameters
    clf = MLPClassifier(
        hidden_layer_sizes=(hidden_layer_size,),
        activation=activation,
        alpha=alpha,
        random_state=42
    )

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train, y_train): # add y_train here
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        clf.fit(X_train_fold, y_train_fold)
        y_val_pred = clf.predict(X_val_fold)
        accuracy = accuracy_score(y_val_fold, y_val_pred)
        cv_scores.append(accuracy)

    # Calculate the average accuracy
    accuracy_avg = sum(cv_scores) / len(cv_scores)

    return -accuracy_avg  # Optimize for average accuracy

# Optimize using Optuna
study = optuna.create_study(direction='minimize')
start_time = time.time()
study.optimize(objective, n_trials=100)
end_time = time.time()

# Get the best parameters and retrain the model
best_params = study.best_params
best_clf = MLPClassifier(
    hidden_layer_sizes=(best_params['hidden_layer_size'],),
    activation=best_params['activation'],
    alpha=best_params['alpha'],
    random_state=42
)
best_clf.fit(X_train, y_train)

# Predict on the test set using the best classifier
y_pred = best_clf.predict(X_test)

# Calculate accuracy, precision, recall, and f1-score on the test set
test_accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Print metrics and best parameters
print("Best Parameters:", best_params)
print("Test Accuracy:", test_accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)