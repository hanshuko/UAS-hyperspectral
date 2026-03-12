import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Data
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture = sio.loadmat('../ML_Data/Moisture_Percentage.mat')

X = Signals[list(Signals.keys())[-1]].T
y = Moisture[list(Moisture.keys())[-1]].ravel()   # make 1D for sklearn

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def test_poly_model(degree):
    # Pipeline prevents leakage: scaling + PCA + poly are fit inside each CV fold
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.85)),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression())
    ])

    # Cross-validation
    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    print(f"\nPolynomial Degree {degree}")
    print("CV R² scores:", cv_scores)
    print("Mean CV R²:", cv_scores.mean())

    # Hold-out test (optional, for plots)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print("Hold-out R²:", r2)
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)

    # Plot
    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel("Actual FMC")
    plt.ylabel("Predicted FMC")
    plt.title(f"PCA + Polynomial Regression (Degree {degree})")

    plt.subplot(1,2,2)
    plt.plot(y_test, 'o-', label="Actual Moisture")
    plt.plot(y_pred, 's--', label="Predicted Moisture")
    plt.title(f"Actual vs Predicted (Degree {degree})")
    plt.xlabel("Sample Index")
    plt.ylabel("Moisture (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

test_poly_model(1)
test_poly_model(2)
test_poly_model(3)