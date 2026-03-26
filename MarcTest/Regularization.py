import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Load Data
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture = sio.loadmat('../ML_Data/Moisture_Percentage.mat')

X = Signals[list(Signals.keys())[-1]].T
y = Moisture[list(Moisture.keys())[-1]].ravel()  # 1D for sklearn

kf = KFold(n_splits=5, shuffle=True, random_state=42)

degree = 3  # try 1 and 2
alphas = np.logspace(-6, 1, 40)  # good starting range for Lasso

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=4)),  # also try 0.95
    ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
    ("model", LassoCV(alphas=alphas, cv=kf, max_iter=200000))
])

# Fit on all data (CV happens inside LassoCV)
pipe.fit(X, y)

best_alpha = pipe.named_steps["model"].alpha_
print("Best alpha:", best_alpha)

# Optional: hold-out plot for visualization
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

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_test, y_pred, alpha=0.7)
mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
plt.plot([mn, mx], [mn, mx], 'r--')
plt.xlabel("Actual FMC")
plt.ylabel("Predicted FMC")
plt.title(f"LassoCV (Degree {degree})")

plt.subplot(1,2,2)
plt.plot(y_test, 'o-', label="Actual")
plt.plot(y_pred, 's--', label="Predicted")
plt.title("Actual vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Moisture (%)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()