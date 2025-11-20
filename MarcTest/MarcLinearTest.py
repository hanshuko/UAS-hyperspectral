import numpy as np
import scipy.io as sio
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load Data
Bands = sio.loadmat('../ML_Data/Bands.mat')
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('../ML_Data/Moisture_Percentage.mat')

#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

#Split into training and testing sets (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Predicting 
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)

# rmse
rmse_train = np.sqrt(mean_squared_error(Y_train_pred,y_train))
rmse_test = np.sqrt(mean_squared_error(Y_test_pred,y_test))
print("Train RMSE: ",rmse_train)
print("Test RMSE: ",rmse_test)

plt.figure(figsize=(10,6))
plt.plot(Y_test_pred,'s--', label='Predicted Moisture')
plt.plot(y_test,'o-', label='Actual Moisture')
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title('Predicted vs Actual Moisture on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,6))
plt.scatter(y_test, Y_test_pred)
min_val = min(y_test.min(), Y_test_pred.min())
max_val = max(y_test.max(), Y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
plt.xlabel('Actual FMC')
plt.ylabel('Predicted FMC')
plt.title('Predicted vs Actual FMC')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()