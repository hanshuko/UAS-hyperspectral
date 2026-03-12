import scipy.io as sio
import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# Loading In Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "ML_Data")

Bands = sio.loadmat(os.path.join(DATA_DIR, "Bands.mat"))
Signals = sio.loadmat(os.path.join(DATA_DIR, "Signals.mat"))
Moisture_Percentage = sio.loadmat(os.path.join(DATA_DIR, "Moisture_Percentage.mat"))

# Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T.ravel()

# Split into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Optional convex mixing on TRAINING ONLY
def ConvexMixWithLabels(X, y, nNew=50, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    XNew = []
    YNew = []

    for _ in range(nNew):
        i, j = rng.choice(n, 2, replace=False)
        alpha = rng.random()

        x_new = alpha * X[i] + (1 - alpha) * X[j]
        y_new = alpha * y[i] + (1 - alpha) * y[j]

        XNew.append(x_new)
        YNew.append(y_new)

    return np.array(XNew), np.array(YNew)

use_mix = False
if use_mix:
    XNew, YNew = ConvexMixWithLabels(X_train, Y_train, nNew=50, random_state=42)
    X_train_final = np.concatenate([X_train, XNew], axis=0)
    Y_train_final = np.concatenate([Y_train, YNew], axis=0)
else:
    X_train_final = X_train
    Y_train_final = Y_train

# Savitzky-Golay preprocessing
window_length = 11
polyorder = 2
deriv = 0

X_train_sg = savgol_filter(
    X_train_final, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1
)
X_test_sg = savgol_filter(
    X_test, window_length=window_length, polyorder=polyorder, deriv=deriv, axis=1
)

# Scale before PCA
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sg)
X_test_scaled = scaler.transform(X_test_sg)

# PCA
n_components = 5
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA ON | n_components = {n_components}")
print(f"SG ON | window_length = {window_length}, polyorder = {polyorder}, deriv = {deriv}")

# Ridge regression
alpha = 0.1
reg = Ridge(alpha=alpha)
reg.fit(X_train_pca, Y_train_final)

# Predict
Y_train_pred = reg.predict(X_train_pca)
Y_test_pred = reg.predict(X_test_pca)

# Metrics
r2_train = r2_score(Y_train_final, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(Y_train_final, Y_train_pred)
mse_test = mean_squared_error(Y_test, Y_test_pred)
rmse_test = np.sqrt(mse_test)

# Print results
print("Training R^2:", r2_train)
print("Testing R^2:", r2_test)
print("Training MSE:", mse_train)
print("Testing MSE:", mse_test)
print("Testing RMSE:", rmse_test)

# Plot predicted vs actual on test set
plt.figure(figsize=(10, 6))
plt.plot(Y_test, 'o-', label='Actual Moisture', markersize=6)
plt.plot(Y_test_pred, 's--', label='Predicted Moisture', markersize=6)
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title('SG + PCA + Ridge: Predicted vs Actual Moisture on Test Set')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_test_pred, alpha=0.7, label="Samples")
min_val = min(Y_test.min(), Y_test_pred.min())
max_val = max(Y_test.max(), Y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label="Ideal Fit")
plt.xlabel("Actual FMC")
plt.ylabel("Predicted FMC")
plt.title("SG + PCA + Ridge: Predicted vs Actual")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()