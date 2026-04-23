# Gaussian Process Regression Folder

## Purpose
This folder contains experimentation with implementing gaussian process regression in MATLAB as an ML method for predicting soil moisture from hyperspectral soil readings. Work in this folder was created before our data collection flight, so it uses publicly sourced data.

## Contents

  - **gpml-matlab-master** - Files for third-party MATLAB toolbox with expanded GP capabilities. Installation instructions included in folder. Partnered with GP textbook, which may be accessed at https://gaussianprocess.org/gpml/.

  - **soilmoisture_dataset.xlsx** - Soil hyperspectral signatures with corresponding soil temperature and moisture content. Measures reflectance for 120 bands.

  - **GPMLAttempt.m** - Experimental GP regression pipeline for predicting soil moisture from hyperspectral data. Trains Gaussian process models using GPML toolbox, and repeats training for different covariance setups and smoothing parameters to compare RMSE performance.

  - **kFoldCVVersion.m** - Similar workflow to `GPMLAttempt.m`, but implements 10-fold cross validation instead of a single random train/test split.

## Notes
- Explore work from this folder with caution, as data leakage was a major problem at this stage in our research.
