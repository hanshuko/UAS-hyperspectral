import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Loading In Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "ML_Data")

Bands = sio.loadmat(os.path.join(DATA_DIR, "Bands.mat"))
Signals = sio.loadmat(os.path.join(DATA_DIR, "Signals.mat"))
Moisture_Percentage = sio.loadmat(os.path.join(DATA_DIR, "Moisture_Percentage.mat"))

# Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

# Split into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Function to create new convex pairs
def ConvexMixWithLabels(X, y, nNew=50, random_state=42):
    n = X.shape[0]
    XNew = []
    YNew = []

    for _ in range(nNew):
        i, j = np.random.choice(n, 2, replace=False)
        alpha = np.random.rand()

        x_new = alpha * X[i] + (1 - alpha) * X[j]
        y_new_sample = alpha * y[i] + (1 - alpha) * y[j]

        XNew.append(x_new)
        YNew.append(y_new_sample)

    return np.array(XNew), np.array(YNew)

#Create new pairs
XNew, YNew = ConvexMixWithLabels(X_train, Y_train, random_state=42)

#Append new data to training set
XTrainNew = np.append(X_train,XNew,axis=0)
YTrainNew = np.append(Y_train,YNew)

use_pca = False
n_components = 5

if use_pca:
    pca = PCA(n_components=n_components)
    X_train_features = pca.fit_transform(XTrainNew)
    X_test_features = pca.transform(X_test)
    print(f"PCA ON | n_components = {n_components}")
else:
    X_train_features = XTrainNew
    X_test_features = X_test
    print("PCA OFF")


# Polynomial feature expansion
degree = 2
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train_features)
X_test_poly = poly.transform(X_test_features)



# Create and fit regression model
from sklearn.linear_model import Ridge

reg = Ridge(alpha=1.0)
reg.fit(X_train_poly, YTrainNew)

# Predict on both train and test sets
Y_train_pred = reg.predict(X_train_poly)
Y_test_pred = reg.predict(X_test_poly)

# Evaluate model
r2_train = r2_score(YTrainNew, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(YTrainNew, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
rmse_test = np.sqrt(mse_test)

# Print Results
print("Training R^2:", r2_train)
print("Testing R^2:", r2_test)
print("Training MSE:", mse_train)
print("Testing MSE:", mse_test)
print("Testing RMSE:", rmse_test)

# Plotting Results
plt.figure(figsize=(10, 6))
plt.plot(Y_test, 'o-', label='Actual Moisture', markersize=6)
plt.plot(Y_test_pred, 's--', label='Predicted Moisture', markersize=6)
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title(f'Predicted vs Actual Moisture on Test Set Regression (Degree {degree})')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_test_pred, alpha=0.7, label="Samples")
min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
plt.xlabel("Actual FMC")
plt.ylabel("Predicted FMC")
plt.title(f"Predicted vs Actual Regression (Degree {degree})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()