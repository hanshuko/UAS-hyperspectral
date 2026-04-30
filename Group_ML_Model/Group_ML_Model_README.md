# Group_ML_Model

## Overview

This folder contains a collection of machine learning models and preprocessing techniques developed by the group to predict **moisture percentage** from **hyperspectral signals** acquired by UAS (Unmanned Aerial Systems). The models explore various regression approaches, dimensionality reduction strategies, and spectral preprocessing methods to optimize prediction accuracy.

---

## Purpose

The primary goal is to build and compare different machine learning regression models that can accurately estimate soil or vegetation moisture content from hyperspectral data. The folder contains both foundational implementations and experimental approaches to identify the best-performing model architecture.

---

## File Descriptions

### Core Implementation Files

#### **Main.py**
- **Purpose**: Primary baseline model implementation
- **Approach**: Linear regression with convex mixture data augmentation
- **Key Features**:
  - Loads spectral signals and moisture labels from the `ML_Data` folder
  - Implements `ConvexMixWithLabels()` function to synthetically generate 50 new training samples by interpolating between existing samples
  - Trains a linear regression model on augmented training data
  - Evaluates performance using R² and MSE metrics
  - Visualizes predicted vs. actual moisture values
  - **REMOVE DATA AUGMENTATION PROCEDURE**

#### **PCARegAndMain.py**
- **Purpose**: Extended version of Main.py with optional dimensionality reduction
- **Approach**: Ridge regression with optional PCA preprocessing
- **Key Features**:
  - Includes toggle to enable/disable PCA (currently set to `False`)
  - Applies polynomial feature expansion (degree 2) for non-linear relationships
  - Uses Ridge regression for regularization
  - Generates both line and scatter plot visualizations
  - Includes convex mixture data augmentation
  - **REMOVE DATA AUGMENTATION PROCEDURE**

#### **RF_PLSR.py**
- **Purpose**: Comprehensive multi-model comparison script
- **Approach**: Implements three different regression paradigms
- **Models Included**:
  1. **Linear Regression** with optional Savitzky-Golay spectral smoothing
  2. **Random Forest Regressor** (500 estimators) for non-linear relationships
  3. **Partial Least Squares Regression (PLSR)** with cross-validated component selection
- **Key Features**:
  - Spectral preprocessing function with derivative calculation options
  - Hyperparameter tuning for PLSR using 5-fold cross-validation
  - Comparative evaluation across all three models

#### **AhyeongModel.py**
- **Purpose**: Comparative study of regularized regression techniques
- **Approach**: Tests multiple linear and non-linear models
- **Models Included**:
  1. Ridge Regression
  2. Lasso Regression
  3. Elastic Net
  4. Gaussian Process Regression (GPR)
- **Key Features**:
  - Standardized input preprocessing for fair comparison
  - Modular evaluation function for consistency
  - Handles multi-model benchmarking
  - **CHANGE TITLE TO REFLECT CONTENT**

#### **ElasticNet_PCA_Tuning.py**
- **Purpose**: Hyperparameter optimization for ElasticNet + PCA pipeline
- **Approach**: Grid search with 5-fold cross-validation
- **Tuning Parameters**:
  - PCA components: [2, 3, 4, 5, 10]
  - ElasticNet alpha: [0.0001, 0.001, 0.01, 0.1]
  - L1 ratio: [0.1, 0.3, 0.5, 0.7, 0.9]
- **Key Features**:
  - Prevents data leakage through pipeline architecture
  - Reports best parameters and cross-validated RMSE
  - Calculates R² on full dataset

### Supplementary Analysis Files

#### **Marc.py**
- **Purpose**: Experimental polynomial regression with PCA preprocessing
- **Approach**: PCA feature extraction followed by polynomial regression
- **Key Features**:
  - Reduces high-dimensional spectral data to 4 principal components
  - Tests polynomial degrees 1, 2, and 3
  - Produces scatter and line plots for visual evaluation
  - Calculates R², MSE, MAE, and RMSE metrics
  - **CHANGE TITLE TO REFLECT CONTENT**

#### **SavgolAndLinRegression.py**
- **Purpose**: Demonstrates end-to-end pipeline with spectral smoothing
- **Approach**: Savitzky-Golay smoothing → Scaling → PCA → Polynomial Regression
- **Key Features**:
  - Applies 1st derivative Savitzky-Golay filter (window=11, poly order=2)
  - 4-component PCA reduction
  - Tests polynomial degrees 1 and 2
  - Comprehensive visualization

#### **SavGolPCARegCV.py**
- **Purpose**: Builds an end-to-end regression pipeline for predicting moisture percentage from spectral signals using cross-validation
- **Approach**: Savitzky–Golay first derivative → Scaling → PCA → Polynomial Regression → 5-fold cross-validation.
- **Key Features**:
  - Uses a Pipeline to avoid data leakage during cross-validation
  - Applies Savitzky–Golay derivative preprocessing to spectral data 
  - Reduces dimensionality using PCA before regression
  - Compare polynomial regression models
  - Applying k-fold cross validation to evaluate model performance


#### **Smoothing.py**
- **Purpose**: Reusable spectral preprocessing utility
- **Approach**: Modular Savitzky-Golay filtering function
- **Key Features**:
  - Configurable window length and polynomial order
  - Support for derivative calculation (0th, 1st, 2nd)
  - Flexible integration into other pipelines
  - Well-documented function signature

### Configuration Files

#### **requirements.txt**
- Complete list of Python dependencies

---

## Data Input

All scripts expect data files in the `../ML_Data/` directory relative to this folder:
- **Bands.mat**: Spectral band information
- **Signals.mat**: Hyperspectral signal data (samples × spectral bands)
- **Moisture_Percentage.mat**: Target labels (moisture content)

---

## Model Comparison Summary

| Model | File | Approach | Key Advantage |
|-------|------|----------|---------------|
| Linear Regression | Main.py | Baseline with data augmentation | Simplicity and interpretability |
| Ridge + Polynomial | PCARegAndMain.py | Regularized with feature expansion | Balances fit and generalization |
| Random Forest | RF_PLSR.py | Ensemble non-linear | Captures complex relationships |
| PLSR | RF_PLSR.py | Latent variable reduction | Handles multicollinearity well |
| ElasticNet | AhyeongModel.py | Dual-penalty regularization | Combines L1 and L2 benefits |
| Gaussian Process | AhyeongModel.py | Probabilistic regression | Uncertainty quantification |


---

## Usage

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
2. Please ensure that scripts are functional, pipelines are secure, and procedures are well-commented before introducing experimental code to this folder.

