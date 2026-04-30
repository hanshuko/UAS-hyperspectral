# ML Data Folder

## Purpose
This folder contains all hyperspectral and moisture data collected from the prairie flight in November 2025 that has been used to train and test our ML models up to this point (4/23/2026). If any additional data is introduced, please add it to this file.

##  Contents

- **Signals.mat** - Hyperspectral reflectance data collected from the flight. Contains 84 samples, each with 300 reflectance values.
- **Bands.mat** - Single array with 300 wavelength values (in nm) that correspond to each reflectance value in `Signals.mat`.
- **Moisture_Percentage.mat** - Array with 84 fuel-moisture content values corresponding to each sample in `Signals.mat` respectively.

## Using this folder
- Import this data to any script performing data analysis or ML model training/testing
- Important Note: These files are all MATLAB data files, so they will need to be converted to other formats when using other coding languages. See `main.py` for performing this conversion in Python.
