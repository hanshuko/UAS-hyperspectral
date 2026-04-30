import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

## SMOOTHING FUNCTION
def spectral_preprocessing(X, use_savgol=False, window_length=11, polyorder=2, deriv=0):

    if use_savgol:
        X = savgol_filter(
            X,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            axis=1
        )
        print(f"Savitzky-Golay applied | window={window_length}, poly={polyorder}, deriv={deriv}")
    else:
        print("No spectral preprocessing applied")

    return X


# Load Data
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture = sio.loadmat('../ML_Data/Moisture_Percentage.mat')

X = Signals[list(Signals.keys())[-1]].T
y = Moisture[list(Moisture.keys())[-1]].T

## APPLY PREPROCESSING
X_sg = spectral_preprocessing(X, use_savgol=True, window_length=11, polyorder=2, deriv=1)

# scale 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sg)

# 3. PCA
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)
# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Function to evaluate polynomial regression
def test_poly_model(degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ## RMSE and R2 score
    rmse = np.sqrt(mse)
    
    print(f"\nPolynomial Degree {degree}")
    print("R²:", r2)
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

   #  Scatter plot vs perfect prediction 
    plt.figure(figsize=(10,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual FMC")
    plt.ylabel("Predicted FMC")
    plt.title(f"PCA + Polynomial Regression (Degree {degree})")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Test linear, quadratic, cubic
test_poly_model(1)
test_poly_model(2)
#test_poly_model(3)
# PCA Explained Variance Plot
#plt.figure(figsize=(6,4))
#plt.bar(range(1,4), pca.explained_variance_ratio_*100)
#plt.xlabel("Principal Component")
#plt.ylabel("Variance Explained (%)")
#plt.title("PCA Explained Variance")z
#plt.grid(True)
#plt.show()