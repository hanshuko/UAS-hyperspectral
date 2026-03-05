from scipy.signal import savgol_filter

def spectral_preprocessing(X, use_savgol=False, window_length=11, polyorder=2, deriv=0):
    """
    Parameters
    X : numpy array
        Spectral data (samples × bands)

    use_savgol : bool
        Apply Savitzky-Golay filtering

    window_length : int
        Number of bands used in smoothing (must be odd)

    polyorder : int
        Polynomial order

    deriv : int
        Derivative order
        0 = smoothing only
        1 = first derivative
        2 = second derivative

    Returns
    X_processed : numpy array
    """

    X_processed = X.copy()

    if use_savgol:
        X_processed = savgol_filter(
            X_processed,
            window_length=window_length,
            polyorder=polyorder,
            deriv=deriv,
            axis=1
        )

    return X_processed