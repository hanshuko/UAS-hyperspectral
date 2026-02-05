import scipy.io as sio
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

##Comment your name below
#-Grant Mirka
#-Ethan Royse
#-Hanshu Kotta
#-Marc Wannawitchate




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

## Marc's note
# Scale inputs (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# PCA to 3–5 components (start with 3)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance:", pca.explained_variance_ratio_.sum())

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=42)

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
    # ---- PLOTTING ----
    plt.figure(figsize=(12,5))

    # Scatter plot
    plt.subplot(1,2,1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual FMC")
    plt.ylabel("Predicted FMC")
    plt.title(f"PCA + Polynomial Regression (Degree {degree})")

    # Line plot
    plt.subplot(1,2,2)
    plt.plot(y_test, 'o-', label="Actual Moisutre")
    plt.plot(y_pred, 's--',label="Predicted Moisture")
    plt.title(f"Actual vs Predicted (Degree {degree})")
    plt.xlabel("Sample Index")
    plt.ylabel("Moisture (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Test linear, quadratic, cubic
test_poly_model(1)
test_poly_model(2)
test_poly_model(3)

####
