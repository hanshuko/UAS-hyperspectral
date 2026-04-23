# Ethan_Test: ML Experimentation Folder

This folder contains experimental scripts exploring data augmentation and feature analysis techniques for hyperspectral moisture prediction. Each script is designed to experiment with a specific ML concept or technique that could be valuable for the team's ongoing work.

## Quick Overview

| Script | Main Focus | Key Takeaway |
|--------|-----------|--------------|
| `PCATest.py` | Smart data augmentation with PCA | Data augmentation requires larger data sets or stronger scientific foundation to be meaningful |
| `Scikit_Test.py` | Basic convex mixing augmentation | See above |
| `correlationAnalysis.py` | Band importance analysis | Identifying which wavelengths matter most for predicting moisture |
| `spectralDervAnalysis.py` | Preprocessing via derivatives | Testing if first-order spectral derivatives improve model performance |
| `endmemberExtraction.py` | Endmember extraction | Exploring hyperspectral unmixing to find pure spectral signatures |
| `train_Test_Split` | Stratified data splitting | Consider train/test splits maintaining target distribution for better validation |

---

## Detailed Script Descriptions

### 1. **PCATest.py** – Advanced Data Augmentation with PCA
**What it does:** Generates synthetic training samples by blending pairs of samples in PCA space.

**Key techniques:**
- Fits PCA on training data and selects top 3 principal components by correlation with target
- Uses "convex mixing" with a moisture window constraint (blends samples with similar moisture levels)
- Performs hyperparameter grid search over PCA components, moisture window, and mixing ratio (alpha)
- Uses K-Fold cross-validation with 5-fold splits
- Final model: PLS regression on augmented training data

**Takeaway:**
Makes an attempt at data augmentation by respecting the data distribution, only mixing samples with similar target values. The grid search results show which hyperparameters work best. **PROFESSORS ADVISED THAT DESPITE ATTEMPTS TO CAPITALIZE ON SIGNAL SIMILARITY, THIS MIXING METHOD IS NOT BASED ON REALITY. TRULY MEANINGFUL DATA AUGMENTATION WILL REQUIRE MANY MORE SAMPLES!**

**To run:** Update data paths, then execute. Outputs R² scores and heatmap visualization.

---

### 2. **Scikit_Test.py** – Basic Convex Mixing Augmentation
**What it does:** Simple baseline for data augmentation—randomly blends sample pairs with random mixing ratios.

**Key techniques:**
- Random convex combinations: `alpha * X[i] + (1-alpha) * X[j]` where alpha is uniform random
- Creates 50 new synthetic samples per run
- Endmember extraction using NFINDR (N-FINDR algorithm)
- PLS regression on augmented data

**Takeaway:**
This is the **simplest augmentation baseline**. Compare its performance against `PCATest.py` to understand the value of constraints and feature selection. **NAIVE AUGMENTATION MAY APPEAR TO HELP, BUT RESULTS ARE UNPREDICTABLE AND NOT BASED IN REALITY.**

**To run:** Update absolute file paths, execute. Outputs train/test R² and MSE metrics.

---

### 3. **correlationAnalysis.py** – Understanding Band Importance
**What it does:** Analyzes which spectral bands are most correlated with moisture content.

**Key techniques:**
- Plots all spectra colored by moisture level (viridis colormap)
- Calculates Pearson correlation between each band and moisture
- Smooths correlation curve with a 7-point moving window
- Validates model significance using permutation testing (1000 permutations)
- Pipeline-based approach to prevent data leakage

**Takeaway:**
Before augmenting or building models, **understand your data**. This script shows which wavelengths naturally predict moisture. The permutation test tells you if the model beats random chance. **Exploratory analysis saves time—you might find that only a few bands matter, enabling better feature selection.**

**To run:** Update paths, execute. Outputs R² statistics and visualizations.

---

### 4. **spectralDervAnalysis.py** – First-Order Derivative Preprocessing
**What it does:** Tests if taking the first derivative of spectral signatures improves prediction.

**Key techniques:**
- Savitzky-Golay filter with window length 11, polynomial order 2, first derivative (deriv=1)
- Cross-validation and permutation testing (identical to correlationAnalysis.py)
- Compares model performance on derivative-transformed data

**Takeaway:**
**Preprocessing matters!** Derivatives remove baseline effects and can enhance fine spectral features. This script tests one specific preprocessing approach. **Before jumping to complex models, try simple preprocessing. Compare R² scores against raw spectra to see if derivatives help your specific task.**

**To run:** Update relative paths (`../ML_Data/`), execute. Check if derivative preprocessing beats baseline.

---

### 5. **endmemberExtraction.py** – Hyperspectral Unmixing Basics
**What it does:** Extracts pure spectral signatures (endmembers) from mixed hyperspectral data using the NFINDR algorithm.

**Key techniques:**
- Reshapes data into 3D cube format (7 × 12 × 300 spatial × spectral)
- Estimates number of endmembers automatically using HfcVd method
- N-FINDR algorithm extracts pure spectra

**Takeaway:**
This is an introduction to hyperspectral unmixing. The extracted endmembers represent pure materials in your scene. **Understanding the physical composition (via endmembers) could lead to better features for moisture prediction or data augmentation strategies that respect material boundaries.**

**Note:** Script is incomplete—endmembers are extracted but not used downstream. This is a starting point for future exploration.

---

### 6. **train_Test_Split** – Stratified Splitting Best Practices
**What it does:** Explores stratified train/test splitting that maintains target distribution.

**Key techniques:**
- **Stratified split:** Uses `pd.qcut()` to divide target range into 5 quantiles
- Ensures both train and test sets have similar moisture distributions
- Visualizes distributions via histograms
- Quick test with Gaussian Process Regressor

**Takeaway:**
Consider using stratified splits for regression tasks with limited data. Random splitting can accidentally put all high/low values in one set, corrupting evaluation. This script demonstrates how to do this. **All sections of the machine learning pipeline require adequate forethought and reasoning for how they are conducted.**

---

## Recommendations for New Team Members

1. **Start with `correlationAnalysis.py`** – Understand your data before modeling
2. **Then try `train_Test_Split`** – Learn proper validation practices
4. **Experiment with `spectralDervAnalysis.py`** – Test preprocessing ideas
5. **Revisit `endmemberExtraction.py`** – Think about physical constraints in augmentation

## Common Issues & Fixes

- **File paths:** These scripts use absolute/relative paths. Update to your system.
- **Dependencies:** Requires `pysptools`, `sklearn`, `scipy`, `numpy`, `pandas`, `matplotlib`, `seaborn`
- **Data format:** All expect `.mat` files with specific keys—check data loader code

## Future Directions

- Combine endmember extraction with PCA analysis
- Test other preprocessing (e.g., continuum removal, second derivatives)
- Implement stratified K-Fold in established pipeline

---

**Questions?** Check inline comments in each script or reach out to Ethan.
