# 1-Multi-layer Perceptron (MLP):
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

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

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Convert the probabilities to binary predictions
y_pred = [1 if p >= 0.5 else 0 for p in y_pred]

# Calculate the performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='MLP - ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Print the ROC AUC score
print("Multi-layer Perceptron (MLP) ROC AUC score:", roc_auc)

# Print the results
print("Multi-layer Perceptron (MLP) metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
###############################################################################

# 2- Basic LSTM model:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

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

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the training and testing data to 3D for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2]), activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='LSTM - ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Print the ROC AUC score
print("Basic LSTM ROC AUC score:", roc_auc)

# Calculate other performance metrics
y_pred_class = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

# Print the other performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

###############################################################################

# 3 - Basic Stacked LSTM model:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

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

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the training and testing data to 3D for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the Stacked LSTM model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='Stacked LSTM - ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Print the ROC AUC score
print("Basic Stacked LSTM ROC AUC score:", roc_auc)

# Calculate other performance metrics
y_pred_class = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

# Print the other performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
################################################################################

# 4 -Basic CNN model:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

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

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the training and testing data to 3D for CNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the CNN model
model = Sequential()
model.add(Conv1D(64, 3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='CNN - ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Print the ROC AUC score
print("Basic CNN ROC AUC score:", roc_auc)

# Calculate other performance metrics
y_pred_class = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

# Print the other performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

####################################################################################

# 4 - Basic RNN model:
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

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

# Standardize the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the training and testing data to 3D for RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the RNN model
model = Sequential()
model.add(SimpleRNN(64, input_shape=(X_train.shape[1], 1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the ROC AUC score
roc_auc = roc_auc_score(y_test, y_pred)

# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='RNN - ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic (ROC) curve')
plt.legend(loc="lower right")
plt.show()

# Print the ROC AUC score
print("Basic RNN ROC AUC score:", roc_auc)

# Calculate other performance metrics
y_pred_class = np.round(y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

# Print the other performance metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)