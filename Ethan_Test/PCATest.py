from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import scipy.io as sio
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression

# Load Data
Signals = sio.loadmat('../ML_Data/Signals.mat')
Moisture = sio.loadmat('../ML_Data/Moisture_Percentage.mat')

X = Signals[list(Signals.keys())[-1]].T
Y = Moisture[list(Moisture.keys())[-1]].T

#Split into training and testing sets (80/20)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#X: Shape (84,300), y: Moisture
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X)

#Find which PCs correlate most with moisture
corrs = [pearsonr(X_pca[:,i].flatten(),Y.flatten())[0] for i in range(X_pca.shape[1])]
best_pcs = np.argsort(np.abs(corrs))[::-1]   #Descending order

def convex_mix_pca(X_pca, y, pca_model, n_new=500, top_pcs=[1,2,4], moisture_window=0.05):
    n = X_pca.shape[0]
    X_new = []
    y_new = []

    for _ in range(n_new):
        if moisture_window is None:
            i, j = np.random.choice(n, 2, replace=False)
        else:
            i = np.random.randint(n)

            valid = np.where(np.abs(y - y[i]) <= moisture_window)[0]
            valid = valid[valid != i]  #remove self

            if len(valid) == 0:
                continue

            j = np.random.choice(valid)
        
        alpha = np.random.rand()
        
        # Mix only top PCs
        mix_pcs = np.zeros(X_pca.shape[1])
        mix_pcs[top_pcs] = alpha * X_pca[i, top_pcs] + (1 - alpha) * X_pca[j, top_pcs]
        
        # Keep the other PCs as the mean (or zero if centered)
        other_pcs = np.setdiff1d(np.arange(X_pca.shape[1]), top_pcs)
        mix_pcs[other_pcs] = 0  # or np.mean(X_pca[:, other_pcs], axis=0)
        
        # Inverse transform to original spectrum
        x_new = pca_model.inverse_transform(mix_pcs)
        y_new_sample = alpha * y[i] + (1 - alpha) * y[j]

        X_new.append(x_new)
        y_new.append(y_new_sample)
        
    return np.array(X_new), np.array(y_new)

XNew, YNew = convex_mix_pca(X_pca,Y,pca)

#Append new data to training set
XTrainNew = np.append(X_train,XNew,axis=0)
YTrainNew = np.append(Y_train,YNew)

#Create and fit regression model
reg = PLSRegression(n_components=2)
reg.fit(XTrainNew, YTrainNew)

#Predict on both train and test sets
Y_train_pred = reg.predict(XTrainNew)
Y_test_pred = reg.predict(X_test)

#Evaluate model
r2_train = r2_score(YTrainNew, Y_train_pred)
r2_test = r2_score(Y_test, Y_test_pred)
mse_train = mean_squared_error(YTrainNew, Y_train_pred)
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