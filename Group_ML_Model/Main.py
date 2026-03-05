
import scipy.io as sio
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

##Comment your name below
#-Grant Mirka
#-Ethan Royse
#-Hanshu Kotta
#-Marc Wannawitchate
#-Caue Faria
#-AHyeong Kim


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

#Establish function to create 50 new convex pairs
def ConvexMixWithLabels(X, y, nNew=50):
    n = X.shape[0]
    XNew = []
    YNew = []
    
    for q in range(nNew):
        i, j = np.random.choice(n, 2, replace=False)
        alpha = np.random.rand()
        
        x_new = alpha * X[i] + (1 - alpha) * X[j]
        y_new_sample = alpha * y[i] + (1 - alpha) * y[j]
        
        XNew.append(x_new)
        YNew.append(y_new_sample)
        
    return np.array(XNew), np.array(YNew)

#Create new pairs
XNew, YNew = ConvexMixWithLabels(X,Y)

#Append new data to training set
XTrainNew = np.append(X_train,XNew,axis=0)
YTrainNew = np.append(Y_train,YNew)

#Create and fit regression model
reg = LinearRegression()
reg.fit(XTrainNew, YTrainNew)

#Predict on both train and test sets
Y_train_pred = reg.predict(XTrainNew)
Y_test_pred = reg.predict(X_test)

#Evaluate model
r2_train = r2_score(YTrainNew, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(YTrainNew, Y_train_pred)
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


