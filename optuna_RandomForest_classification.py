
# Import necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
import time

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
    max_depth = trial.suggest_int('max_depth', 2, 32, log=True)
    n_estimators = trial.suggest_int('n_estimators', 10, 1000, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    
    # Create the classifier with the suggested parameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )

    # Perform cross-validation with 5 folds
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, val_index in kf.split(X_train):
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
best_clf = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    max_depth=best_params['max_depth'],
    min_samples_split=best_params['min_samples_split'],
    random_state=42,
    n_jobs=-1
)
best_clf.fit(X_train, y_train)

# Predict on the test set using the best classifier
y_pred = best_clf.predict(X_test)

# Calculate metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
accuracy = accuracy_score(y_test, y_pred)

# Print metrics and best parameters
print("Best Parameters:", best_params)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Accuracy:", accuracy)

# Print execution time
execution_time = end_time - start_time
print("Execution Time:", execution_time)