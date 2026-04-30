
import scipy.io as sio
from scipy.signal import savgol_filter
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import ElasticNet

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

def spectral_preprocessing(X, use_savgol=True, window_length=11, polyorder=2, deriv=0):
  
    X_processed = X.copy()

    if use_savgol:
        X_processed = savgol_filter(
            X_processed,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            axis=1
        )

    return X_processed

X_processed = spectral_preprocessing(X)

#Split into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X_processed, Y, test_size=0.2, random_state=42
)

#PCA Implementation
use_pca = True
n_components = 5

if use_pca:
    pca = PCA(n_components=n_components)
    X_train_features = pca.fit_transform(X_train)
    X_test_features = pca.transform(X_test)
    print(f"PCA ON | n_components = {n_components}")
else:
    X_train_features = X_train
    X_test_features = X_test
    print("PCA OFF")

#Create and fit regression model
reg = ElasticNet(alpha=0.001, l1_ratio=0.1)
reg.fit(X_train_features, Y_train)

#Predict on both train and test sets
Y_train_pred = reg.predict(X_train_features)
Y_test_pred = reg.predict(X_test_features)

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
plt.scatter(Y_test, Y_test_pred, alpha=0.7)
min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.xlabel("Actual FMC")
plt.ylabel("Predicted FMC")
plt.show


