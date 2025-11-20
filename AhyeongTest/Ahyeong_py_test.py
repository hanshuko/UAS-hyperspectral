# ---------------------------------------------------------
# AHyeong Kim - Linear Regression Model (Hyperspectral)
# ---------------------------------------------------------

import numpy as np
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Load data files
# ---------------------------------------------------------
signals = loadmat("Signals.mat")["Sample_Signals"]
moisture = loadmat("Moisture_Percentage.mat")["Moisture_Percentage"]

# ---------------------------------------------------------
# 2) Extract X, Y:
# ---------------------------------------------------------
X = signals[3, :].reshape(-1, 1)  # using 4th band column of the signal matrix
Y = moisture.reshape(-1, 1)

# ---------------------------------------------------------
# 3) Split train & test
# ---------------------------------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# 4) Train Linear Regression
# ---------------------------------------------------------
model = LinearRegression()
model.fit(X_train, Y_train)

# ---------------------------------------------------------
# 5) Predict
# ---------------------------------------------------------
Y_pred = model.predict(X_test)

# ---------------------------------------------------------
# 6) Evaluate model
# ---------------------------------------------------------
mse = np.mean((Y_pred - Y_test) ** 2)
print("Test MSE:", mse)

# ---------------------------------------------------------
# 7) Plot results
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(Y_test, label="Actual Moisture", marker="o")
plt.plot(Y_pred, label="Predicted Moisture", marker="x")
plt.title("Moisture Prediction - Linear Regression (AHyeong)")
plt.xlabel("Sample Index")
plt.ylabel("Moisture (%)")
plt.legend()
plt.grid(True)
plt.show()
