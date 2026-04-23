import os
import numpy as np
import scipy.io as sio

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error


def load_last_mat_variable(mat_dict):
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise ValueError("No valid data keys found in .mat file.")
    return mat_dict[keys[-1]]


def main():
    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "ML_Data")

    signals = sio.loadmat(os.path.join(data_dir, "Signals.mat"))
    moisture = sio.loadmat(os.path.join(data_dir, "Moisture_Percentage.mat"))

    X = load_last_mat_variable(signals).T
    y = load_last_mat_variable(moisture).T.ravel()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("-" * 40)

    # Pipeline (prevent leakage)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("model", ElasticNet(max_iter=20000, random_state=42))
    ])

    # Hyperparameter grid
    param_grid = {
        "pca__n_components": [2, 3, 4, 5, 10],
        "model__alpha": [0.0001, 0.001, 0.01, 0.1],
        "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
    }

    # Cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X, y)

    print("Best Parameters:")
    print(grid.best_params_)
    print("-" * 40)

    best_rmse = -grid.best_score_
    print(f"Best Cross-Validated RMSE: {best_rmse:.6f}")

    # 추가: R² 계산
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X)
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

    print(f"R^2 (on full data): {r2:.4f}")


if __name__ == "__main__":
    main()