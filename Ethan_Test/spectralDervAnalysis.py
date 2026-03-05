import scipy.io as sio
from scipy.signal import savgol_filter
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score, KFold, permutation_test_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


#Loading In Data
Bands = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Bands.mat')
Signals = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Moisture_Percentage.mat')

#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

#Perform filtering

X_deriv = savgol_filter(X,window_length=11,polyorder=2,deriv=1,axis=1)

### Run PLS and see which bands it keeps
#Create model - use pipeline to prevent data leakage
pipeline = Pipeline([
    ("scale", StandardScaler()),
    ("pls", PLSRegression(n_components=2))
])

#Cross-evaluate
cv = KFold(n_splits=5, shuffle=True, random_state=0)
r2Scores = cross_val_score(pipeline,X_deriv,Y, cv=cv, scoring='r2')
rmseScores = cross_val_score(pipeline,X_deriv,Y, cv=cv, scoring='neg_root_mean_squared_error')
score, perm_scores, pvalue = permutation_test_score(pipeline, X_deriv, Y,scoring="r2",cv=5,n_permutations=1000,n_jobs=-1)

#Print Results
print("R^2 scores: ", r2Scores)
print("Mean R^2: ", r2Scores.mean())
print("R^2 Std: ", r2Scores.std())
print("RMSE scores: ", rmseScores)
print("Mean RMSE: ", rmseScores.mean())
print("RMSE Std: ", rmseScores.std())
print("Permutation Test R^2: ", score)
print("Permutation Test P-Value: ", pvalue)