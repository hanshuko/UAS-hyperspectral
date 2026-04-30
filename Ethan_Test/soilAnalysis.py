import pandas as pd
import os
import scipy.io as sio
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

#Loading In Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "ML_Data")

Bands = sio.loadmat(os.path.join(DATA_DIR, "Bands.mat"))
Signals = sio.loadmat(os.path.join(DATA_DIR, "Signals.mat"))
Moisture_Percentage = sio.loadmat(os.path.join(DATA_DIR, "Moisture_Percentage.mat"))

df = pd.read_csv(os.path.join(DATA_DIR, 'soilmoisture_dataset.csv'))

#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T
Y = np.squeeze(Y)

#Split into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Target variable (moisture)
ySoil = df['soil_moisture'].values
ySoil = ySoil/100

# Spectral bands = all numeric column names (wavelengths)
band_columns = [col for col in df.columns if col.isdigit()]

# Feature matrix (reflectance signals)
XSoil = df[band_columns].values

# Soil wavelengths (from column names)
soil_wavelengths = np.array(list(map(float, band_columns)))

# Create interpolation function (vectorized across rows)
interp_func = interp1d(
    soil_wavelengths,
    XSoil,                      # shape: (n_samples, n_soil_bands)
    axis=1,
    kind='linear',          
    bounds_error=False,
    fill_value="extrapolate"
)

# Interpolated spectra
X_interp = interp_func(Bands)
X_interp = np.squeeze(X_interp)  # Remove singleton dimension if present
print(X_interp)

#Append interpolated spectra to original features
X_combined = np.vstack((X_interp, X_train))
y_combined = np.hstack((ySoil, Y_train))

#Create and fit regression model
reg = LinearRegression()
reg.fit(X_combined, y_combined)

#Predict on both train and test sets
Y_train_pred = reg.predict(X_combined)
Y_test_pred = reg.predict(X_test)

#Evaluate model
r2_train = r2_score(y_combined, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(y_combined, Y_train_pred)
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