import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#Loading In Data
Bands = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Bands.mat')
Signals = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Moisture_Percentage.mat')

#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

### PLOT ALL BANDS & ORGANIZE BY MOISTURE
#Transpose bands to match dimensions
Bands = Bands.T

# Get the order of moisture values
sort_idx = np.argsort(Y)  # indices that would sort moisture

# Sort X and moisture
X_sorted = X[sort_idx, :]
moisture_sorted = Y[sort_idx]

#Create figure
fig, ax = plt.subplots()

# Normalize moisture values to 0–1
norm = colors.Normalize(vmin=np.min(Y), vmax=np.max(Y))

# Choose colormap
cmap = cm.viridis

#Plot signals one at a time
for i in range(X.shape[0]):
    ax.plot(Bands, X[i, :], color=cmap(norm(Y[i])), alpha=0.7)

# Add colorbar
sm = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
plt.colorbar(sm, ax=ax, label="Moisture Content")

#Create plot
plt.xlabel("Wavelength (nm)")
plt.ylabel("Reflectance")
plt.title('Moisture Content Organization')
plt.show()

### CORRELATION COEFFICIENT ANALYSIS
#Initialize array
n_samples, n_bands = X.shape
correlations = np.zeros(n_bands)

#Calculate correlation coefficient between each signal and moisture at each band
for i in range(n_bands):
    correlations[i] = np.corrcoef(X[:, i].flatten(), Y.flatten())[0, 1]

#Make smoothed version of correlations
window = 7
smooth_corr = np.convolve(correlations, np.ones(window)/window, mode='same')

#Plot relationship
plt.figure()
plt.plot(Bands,correlations, alpha=0.5)
plt.plot(Bands,smooth_corr, linewidth=2)
plt.xlabel('Wavelength')
plt.ylabel('Correlation with Moisture')
plt.title('Band-Wise Linear Correlation')
plt.legend(['Unsmoothed','Smoothed'])
plt.show()

### Run PLS and see which bands it keeps
#Create model - use pipeline to prevent data leakage
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("pls", PLSRegression(n_components=2))
])

#Cross-evaluate
cv = KFold(n_splits=5, shuffle=True, random_state=0)
r2Scores = cross_val_score(pipeline,X,Y, cv=cv, scoring='r2')
rmseScores = cross_val_score(pipeline,X,Y, cv=cv, scoring='neg_root_mean_squared_error')
score, perm_scores, pvalue = permutation_test_score(pipeline, X, Y,scoring="r2",cv=5,n_permutations=1000,n_jobs=-1)

#Print Results
print("R^2 scores: ", r2Scores)
print("Mean R^2: ", r2Scores.mean())
print("R^2 Std: ", r2Scores.std())
print("RMSE scores: ", rmseScores)
print("Mean RMSE: ", rmseScores.mean())
print("RMSE Std: ", rmseScores.std())
print("Permutation Test R^2: ", score)
print("Permutation Test P-Value: ", pvalue)
