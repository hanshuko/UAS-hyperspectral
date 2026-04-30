## Marc's note
# Scale inputs (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# PCA to 3–5 components (start with 3)
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance:", pca.explained_variance_ratio_.sum())

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_pca, Y, test_size=0.3, random_state=42)

# Function to evaluate polynomial regression
def test_poly_model(degree):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred = model.predict(X_test_poly)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    ## RMSE and R2 score
    rmse = np.sqrt(mse)
    
    print(f"\nPolynomial Degree {degree}")
    print("R²:", r2)
    print("MSE:", mse)
    print("MAE:", mae)
    print("RMSE:", rmse)
    # ---- PLOTTING ----
    plt.figure(figsize=(12,5))

    # Scatter plot
    plt.subplot(1,2,1)
    plt.scatter(y_test, y_pred, alpha=0.7)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.xlabel("Actual FMC")
    plt.ylabel("Predicted FMC")
    plt.title(f"PCA + Polynomial Regression (Degree {degree})")

    # Line plot
    plt.subplot(1,2,2)
    plt.plot(y_test, 'o-', label="Actual Moisutre")
    plt.plot(y_pred, 's--',label="Predicted Moisture")
    plt.title(f"Actual vs Predicted (Degree {degree})")
    plt.xlabel("Sample Index")
    plt.ylabel("Moisture (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Test linear, quadratic, cubic
test_poly_model(1)
test_poly_model(2)
test_poly_model(3)

####