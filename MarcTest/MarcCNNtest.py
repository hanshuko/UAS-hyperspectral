import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import keras
from keras import layers
import matplotlib.pyplot as plt

# Load Data
Bands = sio.loadmat('../ML_Data/Bands.mat')
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('../ML_Data/Moisture_Percentage.mat')
# Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

# Normalize
x_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)

#Split into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_train.max())
print(X_train.min())
print(y_train.max())
print(y_train.min())

## Build the model
# CNN Model
X_train_cnn = X_train[..., np.newaxis]
X_test_cnn = X_test[..., np.newaxis]

cnn_model = keras.Sequential([
    layers.Conv1D(64, 3, activation='relu', padding='valid', input_shape=(X_train_cnn.shape[1], 1)), # Conv1D layer
    layers.Conv1D(32, 3, activation='relu', padding='valid'),
    layers.Dropout(0.1), # add dropout after the first convolutional layer
    layers.Conv1D(16, 3, activation='relu', padding='valid'),
    layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'),
    layers.Flatten(), # flatten the output
    layers.Dense(128, activation='relu'), # Dense layer
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1), # add dropout after the dense layer
    layers.Dense(1, activation='linear')
])

cnn_model.summary()

## Train the model
batch_size = 8
epochs = 100
cnn_model.compile(optimizer="adam", loss='mse', metrics=['mae'])
hist = cnn_model.fit(X_train_cnn, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# Predict on training set
y_train_pred = cnn_model.predict(X_train_cnn)

# Evaluate CNN Model (testing set)
y_pred_cnn= cnn_model.predict(X_test_cnn)

# MSE and Mean Absolute Error on training
print("\n1D CNN Model Performance")
print("Train Results")
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)

## RMSE and R2 score
rmse_train_cnn = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)

# Results
print("Train R2 Score:", r2_train)
print("Train MSE:", mse_train)
print("Train Mean Absolute Error:", mae_train)
print("Train RMSE:", rmse_train_cnn)

# MSE and Mean Absolute Error
print("\n1D CNN Model Performance")
print("Test Results")
mae_cnn = mean_absolute_error(y_test, y_pred_cnn)
mse_cnn = mean_squared_error(y_test, y_pred_cnn)

## RMSE and R2 score
rmse_cnn = np.sqrt(mse_cnn)
r2_test = r2_score(y_test, y_pred_cnn)

# Results
print("Test R2 Score:", r2_test)
print("Test MSE:", mse_cnn)
print("Test Mean Absolute Error:", mae_cnn)
print("Test RMSE:", rmse_cnn)
# Plot training & validation MAE values
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(hist.history["mae"])
plt.plot(hist.history["val_mae"])
plt.title("Model Mean Absolute Error")
plt.ylabel("Mean Absolute Error")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.grid(True)

# Plot training & validation MSE values
plt.subplot(1, 2, 2)
plt.plot(hist.history["loss"])
plt.plot(hist.history["val_loss"])
plt.title("Model Loss (MSE)")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
plt.grid(True)

plt.tight_layout()
plt.show()

#Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(y_test, 'o-', label='Actual Moisture', markersize=6)
plt.plot(y_pred_cnn, 's--', label='Predicted Moisture', markersize=6)
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title('Predicted vs Actual Moisture on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting Predicted vs Actual field moisture content
plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_cnn)
min_val = min(y_test.min(), y_pred_cnn.min())
max_val = max(y_test.max(), y_pred_cnn.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
plt.xlabel('Actual FMC')
plt.ylabel('Predicted FMC')
plt.title('Predicted vs Actual FMC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()