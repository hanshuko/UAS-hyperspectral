import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

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
plt.xlabel("Band Index")
plt.ylabel("Reflectance")
plt.title('Moisture Content Organization')
plt.show()

### CORRELATION COEFFICIENT ANALYSIS
#Initialize array
n_samples, n_bands = X.shape
correlations = np.zeros(n_bands)

#Calculate correlation coefficient between each signal and moisture at each band
for i in range(n_bands):
    correlations[i] = np.corrcoef(X[:, i], Y)[0, 1]

#Plot relationship
plt.figure()
plt.plot(range(n_bands), correlations)
plt.xlabel('Band Index')
plt.ylabel('Correlation with Moisture')
plt.title('Band-Wise Correlation')
plt.show()


