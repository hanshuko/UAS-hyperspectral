import scipy.io as sio
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


##Comment your name below
#-Grant Mirka
#-Ethan Royse
#-Hanshu Kotta




##

#Loading In Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "ML_Data")

Bands = sio.loadmat(os.path.join(DATA_DIR, "Bands.mat"))
Signals = sio.loadmat(os.path.join(DATA_DIR, "Signals.mat"))
Moisture_Percentage = sio.loadmat(os.path.join(DATA_DIR, "Moisture_Percentage.mat"))
#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

#Split into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#Create and fit regression model
reg = linear_model.LinearRegression()
reg.fit(X_train, Y_train)

#Predict on both train and test sets
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)
#Test Comment
#Evaluate model
r2_train = r2_score(Y_train, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(Y_train, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)

#Print Results
print("Training R^2:", r2_train)
print("Testing R^2:", r2_test)
print("Training MSE:", mse_train)
print("Testing MSE:", mse_test)

#Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(Y_test, 'o-', label='Actual Moisture', markersize=6)
plt.plot(Y_test_pred, 's--', label='Predicted Moisture', markersize=6)
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title('Predicted vs Actual Moisture on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
