import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Data
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture = sio.loadmat('../ML_Data/Moisture_Percentage.mat')

X = Signals[list(Signals.keys())[-1]].T
y = Moisture[list(Moisture.keys())[-1]].T.ravel()

# Savitzky-Golay derivative transformer
def sg_derivative(X):
    return savgol_filter(X, window_length=11, polyorder=2, deriv=1, axis=1)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def test_poly_model(degree):
    pipe = Pipeline([
        ("sg", FunctionTransformer(sg_derivative, validate=False)),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=4)),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("model", LinearRegression())
    ])

    # Cross-validation scores
    cv_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")

    # Cross-validated predictions
    y_cv_pred = cross_val_predict(pipe, X, y, cv=kf)

    # Metrics from CV predictions
    r2 = r2_score(y, y_cv_pred)
    mse = mean_squared_error(y, y_cv_pred)
    mae = mean_absolute_error(y, y_cv_pred)
    rmse = np.sqrt(mse)

    print(f"\nPolynomial Degree {degree}")
    print("CV R² scores:", cv_scores)
    print("Mean CV R²:", cv_scores.mean())
    print("Std CV R²:", cv_scores.std())
    print("CV Prediction R²:", r2)
    print("CV MSE:", mse)
    print("CV MAE:", mae)
    print("CV RMSE:", rmse)

    # Scatter plot vs perfect prediction 
    plt.figure(figsize=(10,6))
    plt.scatter(y, y_cv_pred, alpha=0.7)
    min_val = min(y.min(), y_cv_pred.min())
    max_val = max(y.max(), y_cv_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual FMC")
    plt.ylabel("CV Predicted FMC")
    plt.title(f"Cross-Validated Predictions (Degree {degree})")
    plt.grid(True)

    plt.tight_layout()
    plt.show()



     #fold score bar plot
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(cv_scores) + 1), cv_scores)
    plt.xlabel("Fold")
    plt.ylabel("R² Score")
    plt.title(f"CV Fold R² Scores (Degree {degree})")
    plt.grid(True)
    plt.show()

# Test linear, quadratic, cubic
test_poly_model(1)
test_poly_model(2)
#test_poly_model(3)