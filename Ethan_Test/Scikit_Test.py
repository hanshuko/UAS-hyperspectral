import scipy.io as sio
from sklearn import linear_model, cross_decomposition, ensemble
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#Loading In Data
Bands = sio.loadmat('UAS-hyperspectral/ML_Data/Bands.mat')
Signals = sio.loadmat('UAS-hyperspectral/ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('UAS-hyperspectral/ML_Data/Moisture_Percentage.mat')

#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

#Split into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#Create convex combinations to produce new data


#Create and fit regression model
reg = cross_decomposition.PLSRegression()
reg.fit(X_train, Y_train)

#Predict on both train and test sets
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)

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
