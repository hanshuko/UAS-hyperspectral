import scipy.io as sio
import numpy as np
from pysptools import eea, material_count

#Loading In Data
Bands = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Bands.mat')
Signals = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Signals.mat')
Moisture_Percentage = sio.loadmat('C:/Users/eroys/OneDrive/Documents/GitHub/UAS-hyperspectral/ML_Data/Moisture_Percentage.mat')

#Extracting Out Data
Bands = Bands[list(Bands.keys())[-1]].T
X = Signals[list(Signals.keys())[-1]].T
Y = Moisture_Percentage[list(Moisture_Percentage.keys())[-1]].T

#Reformat data into cube
X3d = X.reshape(7,12,300)

#Estimate dimensions
dims = material_count.vd.HfcVd(X)

# Employ n-findr
nfindr = eea.NFINDR()
mem = nfindr.extract(X3d, dims[0])