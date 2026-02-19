import scipy.io as sio
import os
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

## LINEAR REGRESSION MODEL
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
#Test Comment
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
plt.show(block=False)



## RANDOM FOREST REGRESSION MODEL
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# Initial test of Random Forest Regressor with default parameters
rf = RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, Y_train)

Y_train_pred = rf.predict(X_train)
Y_test_pred = rf.predict(X_test)

print(" ")
print("Training R^2:", r2_score(Y_train, Y_train_pred))
print("Testing R^2:", r2_score(Y_test, Y_test_pred))
print("Training MSE:", mean_squared_error(Y_train, Y_train_pred))
print("Testing MSE:", mean_squared_error(Y_test, Y_test_pred))



## PARTIAL LEAST SQUARES REGRESSION MODEL
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_score

Y_train_1d = Y_train.ravel()
Y_test_1d = Y_test.ravel()

max_components = min(X_train.shape[0] - 1, X_train.shape[1], 50) # Limit to 25 

kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_n = None
best_cv_r2 = -np.inf
cv_r2_scores = []

# Loops through 25 components and performs 5-fold CV to find the best number of components based on R^2 score (takes avg of 5 folds each iteration)
for n_comp in range(1, max_components + 1):
    pls = PLSRegression(n_components=n_comp)

    scores = cross_val_score(pls, X_train, Y_train_1d, cv=kf, scoring="r2")
    mean_score = scores.mean()
    cv_r2_scores.append(mean_score)

    if mean_score > best_cv_r2:
        best_cv_r2 = mean_score
        best_n = n_comp

print("\nPLSR")
print(f"Best n_components (CV): {best_n}")
print(f"Best CV R^2: {best_cv_r2:.4f}")

# Fit best PLS on full training data
pls_best = PLSRegression(n_components=best_n)
pls_best.fit(X_train, Y_train_1d)

# Predict
Y_train_pred_pls = pls_best.predict(X_train).ravel()
Y_test_pred_pls = pls_best.predict(X_test).ravel()

# Evaluate
print("PLSR Training R^2:", r2_score(Y_train_1d, Y_train_pred_pls))
print("PLSR Testing R^2:", r2_score(Y_test_1d, Y_test_pred_pls))
print("PLSR Training MSE:", mean_squared_error(Y_train_1d, Y_train_pred_pls))
print("PLSR Testing MSE:", mean_squared_error(Y_test_1d, Y_test_pred_pls))

# Plot predicted vs actual moisture (test set)
plt.figure(figsize=(10, 6))
plt.plot(Y_test_1d, 'o-', label='Actual Moisture', markersize=6)
plt.plot(Y_test_pred_pls, 's--', label='PLSR Predicted Moisture', markersize=6)
plt.xlabel('Sample Index')
plt.ylabel('Moisture Percentage')
plt.title(f'PLSR Predicted vs Actual Moisture (Test Set) | n_components={best_n}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()