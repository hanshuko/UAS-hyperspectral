from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import scipy.io as sio
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
import itertools
import seaborn as sns

## LOOK INTO FINDING OUTLIERS FROM POORLY PERFORMING CV FOLDS AND REMOVING THEM TO SEE IF IT IMPROVES PERFORMANCE

# Load Data
Signals = sio.loadmat('UAS-hyperspectral/ML_Data/Signals.mat')
Moisture = sio.loadmat('UAS-hyperspectral/ML_Data/Moisture_Percentage.mat')

X = Signals[list(Signals.keys())[-1]].T
Y = Moisture[list(Moisture.keys())[-1]].T

#Split into training and testing sets (80/20)
X_dev, X_final_test, Y_dev, Y_final_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

#Convex Mixing Algorithm
def convex_mix_pca(X_pca, y, pca_model, n_new=500, top_pcs=[1,2,4],
                   moisture_window=0.05, alpha=0.5):
    
    n = X_pca.shape[0]
    X_new = []
    y_new = []

    for _ in range(n_new):
        if moisture_window is None:
            i, j = np.random.choice(n, 2, replace=False)
        else:
            i = np.random.randint(n)

            valid = np.where(np.abs(y - y[i]) <= moisture_window)[0]
            valid = valid[valid != i]

            if len(valid) == 0:
                continue

            j = np.random.choice(valid)

        # Use FIXED alpha instead of random
        a = alpha
        
        mix_pcs = np.zeros(X_pca.shape[1])
        mix_pcs[top_pcs] = a * X_pca[i, top_pcs] + (1 - a) * X_pca[j, top_pcs]

        other_pcs = np.setdiff1d(np.arange(X_pca.shape[1]), top_pcs)
        mix_pcs[other_pcs] = 0

        x_new = pca_model.inverse_transform(mix_pcs)
        y_new_sample = a * y[i] + (1 - a) * y[j]

        X_new.append(x_new)
        y_new.append(y_new_sample)

    return np.array(X_new), np.array(y_new)

#Wrap training pipeline into a function
def evaluate_pipeline(X, Y, n_components, moisture_window, alpha,
                      n_splits=5, n_new=500):

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    r2_scores = []

    for train_idx, test_idx in kf.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]

        #PCA (fit ONLY on training data)
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        #Find best PCs
        corrs = [pearsonr(X_train_pca[:,i].flatten(),Y_train.flatten())[0] 
                 for i in range(X_train_pca.shape[1])]
        top_pcs = np.argsort(np.abs(corrs))[::-1][:3]  # take top 3

        #Generate augmented data
        X_new, Y_new = convex_mix_pca(
            X_train_pca,
            Y_train,
            pca,
            n_new=n_new,
            top_pcs=top_pcs,
            moisture_window=moisture_window,
            alpha=alpha
        )

        #Combine
        X_train_aug = np.vstack([X_train, X_new])
        Y_train_aug = np.vstack([Y_train, Y_new])

        #Train model
        reg = PLSRegression(n_components=2)
        reg.fit(X_train_aug, Y_train_aug)

        #Evaluate
        Y_pred = reg.predict(X_test)
        r2_scores.append(r2_score(Y_test, Y_pred))

    return np.mean(r2_scores), np.std(r2_scores)

#Define the grid search
param_grid = {
    "n_components": [1, 2, 3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "moisture_window": [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2],
    "alpha": [0.2, 0.5, 0.8]
}

results = []

#Run the grid search
for n_comp, m_window, alpha in itertools.product(
        param_grid["n_components"],
        param_grid["moisture_window"],
        param_grid["alpha"]):

    mean_r2, std_r2 = evaluate_pipeline(
        X_dev, Y_dev,
        n_components=n_comp,
        moisture_window=m_window,
        alpha=alpha
    )

    results.append({
        "n_components": n_comp,
        "moisture_window": m_window,
        "alpha": alpha,
        "mean_r2": mean_r2,
        "std_r2": std_r2
    })

    print(f"Done: PCA={n_comp}, window={m_window}, alpha={alpha} → R²={mean_r2:.3f}")


#Convert results into table
df_results = pd.DataFrame(results)
best_params = df_results.sort_values("mean_r2", ascending=False).iloc[0]

print(df_results.sort_values(by="mean_r2", ascending=False))

#Create heat map
alpha_val = 0.5
subset = df_results[df_results["alpha"] == alpha_val]

pivot = subset.pivot(index="moisture_window",
                     columns="n_components",
                     values="mean_r2")

sns.heatmap(pivot, annot=False, cmap="viridis")
plt.title(f"R² Heatmap (alpha={alpha_val})")
plt.show()

#Final Evaluation
# Fit PCA on ALL dev data
pca = PCA(n_components=int(best_params["n_components"]))
X_dev_pca = pca.fit_transform(X_dev)
X_test_pca = pca.transform(X_final_test)

# Find top PCs
corrs = [pearsonr(X_dev_pca[:,i].flatten(), Y_dev.flatten())[0]
         for i in range(X_dev_pca.shape[1])]
top_pcs = np.argsort(np.abs(corrs))[::-1][:3]

# Generate augmented data
X_new, Y_new = convex_mix_pca(
    X_dev_pca,
    Y_dev,
    pca,
    moisture_window=best_params["moisture_window"],
    alpha=best_params["alpha"]
)

# Train final model
X_train_final = np.vstack([X_dev, X_new])
Y_train_final = np.vstack([Y_dev, Y_new])

reg = PLSRegression(n_components=2)
reg.fit(X_train_final, Y_train_final)

# Evaluate ONCE
Y_pred_final = reg.predict(X_final_test)

final_r2 = r2_score(Y_final_test, Y_pred_final)

print("Final Test R²:", final_r2)

