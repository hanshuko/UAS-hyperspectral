import os
import numpy as np
import scipy.io as sio

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet


def load_last_mat_variable(mat_dict):
    keys = [k for k in mat_dict.keys() if not k.startswith("__")]
    if len(keys) == 0:
        raise ValueError("No valid data keys found in .mat file.")
    return mat_dict[keys[-1]]


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(name)
    print(f"  Testing R^2: {r2:.4f}")
    print(f"  Testing RMSE: {rmse:.6f}")
    print("-" * 40)

    return r2, rmse


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

    # Scale first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try several PCA component sizes
    pca_list = [3, 5, 10, 15, 20, 30]

    best_rmse = float("inf")
    best_n = None
    best_r2 = None

    for n in pca_list:
        print(f"PCA components: {n}")

        pca = PCA(n_components=n)
        X_pca = pca.fit_transform(X_scaled)

        X_train, X_test, y_train, y_test = train_test_split(
            X_pca, y, test_size=0.2, random_state=42
        )

        model = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)

        r2, rmse = evaluate_model(
            "ElasticNet + PCA",
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        if rmse < best_rmse:
            best_rmse = rmse
            best_n = n
            best_r2 = r2

    print("\nBEST PCA RESULT")
    print(f"Best PCA components: {best_n}")
    print(f"Best Testing R^2: {best_r2:.4f}")
    print(f"Best Testing RMSE: {best_rmse:.6f}")


if __name__ == "__main__":
    main()