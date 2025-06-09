import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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