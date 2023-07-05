#1-Multi-layer Perceptron (MLP):
import pandas as pd
import numpy as np
import time
import warnings

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Filter out warnings
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

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP model with two additional hidden layers
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
start_time= time.time()
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
end_time = time.time()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the performance metrics on the test set
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Multi-layer Perceptron (MLP) metrics on the test set:")
print("Mean Squared Error (MSE): {:.2f}".format(mse))
print("Mean Absolute Error (MAE): {:.2f}".format(mae))
print("R-squared (R2) score: {:.2f}".format(r2))
print("Time taken to train the model: {:.2f} seconds".format(end_time - start_time))

########################################################################################
#2-Long Short-Term Memory (LSTM):
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Reshape the training and testing data to 3D for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the LSTM model with two additional hidden layers
model = Sequential()
model.add(LSTM(64, input_shape=(1, X_train.shape[2]), activation='relu', return_sequences=True))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(LSTM(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='mse', optimizer='adam')

# Train the model
start_time = time.time()
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
end_time = time.time()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Reshape the predictions to 1D
y_pred = np.reshape(y_pred, (y_pred.shape[0],))

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Long Short-Term Memory (LSTM) metrics:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)
print("Time taken:", end_time - start_time)

#################################################################################
# 3- Convolutional Neural Network (CNN):
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Reshape the training and testing data to 3D for CNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the CNN model with two additional hidden layers
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
model.add(Conv1D(filters=8, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='mse', optimizer='adam')

# Train the model
start_time = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
end_time = time.time()


# Make predictions on the test set
y_pred = model.predict(X_test)

# Reshape the predictions to 1D
y_pred = np.reshape(y_pred, (y_pred.shape[0],))

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Convolutional Neural Network (CNN) metrics:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)
print("Time taken:", end_time - start_time)


##########################################################################################
# 4- Recurrent Neural Network (RNN):
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Reshape the training and testing data to 3D for RNN
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Define the RNN model with two additional hidden layers
model = Sequential()
model.add(SimpleRNN(64, input_shape=(1, X_train.shape[2]), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.compile(loss='mse', optimizer='adam')

# Train the model
start_time = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=0)
end_time = time.time()

# Make predictions on the test set
y_pred = model.predict(X_test)

# Reshape the predictions to 1D
y_pred = np.reshape(y_pred, (y_pred.shape[0],))

# Calculate the performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print("Recurrent Neural Network (RNN) metrics:")
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) score:", r2)
print("Time taken:", end_time - start_time)