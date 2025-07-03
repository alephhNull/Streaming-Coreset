import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


def train_classifier(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> float:
    unique_classes = np.unique(y_train)
    
    if len(unique_classes) < 2:
        print(f"  Warning: Training data contains only one class (label: {unique_classes[0]}). Using constant prediction.")
        # Predict the same class for all validation samples
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        acc = accuracy_score(y_val, y_pred)
        return acc

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    return acc


def train_regressor(X_train: np.ndarray, X_val: np.ndarray, y_train: np.ndarray, y_val: np.ndarray) -> float:
    """
    Trains a Linear Regression model and evaluates its R-squared score.

    Args:
        X_train (np.ndarray): Training features.
        X_val (np.ndarray): Validation features.
        y_train (np.ndarray): Training targets.
        y_val (np.ndarray): Validation targets.

    Returns:
        float: The R-squared score on the validation set.
    """
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    return rmse