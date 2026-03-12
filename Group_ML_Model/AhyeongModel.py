import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler

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

    print(f"{name}")
    print(f"  Testing R^2: {r2:.4f}")
    print(f"  Testing RMSE: {rmse:.6f}")
    print("-" * 40)


def main():
    # Load data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "ML_Data")

    signals_mat = sio.loadmat(os.path.join(data_dir, "Signals.mat"))
    moisture_mat = sio.loadmat(os.path.join(data_dir, "Moisture_Percentage.mat"))

    X = load_last_mat_variable(signals_mat).T
    y = load_last_mat_variable(moisture_mat).T.ravel()

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("-" * 40)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.001, max_iter=10000)
    elastic = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)

    # Evaluate
    evaluate_model("Ridge Regression", ridge, X_train, X_test, y_train, y_test)
    evaluate_model("Lasso Regression", lasso, X_train, X_test, y_train, y_test)
    evaluate_model("Elastic Net", elastic, X_train, X_test, y_train, y_test)

    # Gaussian Process Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kernel = C(1.0) * RBF(length_scale=1.0) + WhiteKernel()

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        normalize_y=True,
        random_state=42
    )

    evaluate_model(
        "Gaussian Process Regression",
        gpr,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()
