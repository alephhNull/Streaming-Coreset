from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, roc_auc_score
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time


def train_logistic_regression(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray = None
) -> dict:
    unique_classes = np.unique(y_train)

    if len(unique_classes) < 2:
        print(f"  Warning: Training data contains only one class (label: {unique_classes[0]}). Using constant prediction.")
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        acc = accuracy_score(y_val, y_pred)
        f1 = float('nan')
        auc = float('nan')
        return {"acc": acc, "auc": auc, "f1": f1}

    clf = LogisticRegression(max_iter=1000, random_state=42)

    if weights is not None:
        assert len(weights) == len(y_train), "Length of weights must match number of training samples"
        clf.fit(X_train, y_train, sample_weight=weights)
    else:
        clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if clf.classes_.shape[0] == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='binary' if clf.classes_.shape[0] == 2 else 'macro', zero_division=0)
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else float('nan')

    return {"acc": acc, "auc": auc, "f1": f1}

def train_nn_classifier(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    input_dim: int,
    num_classes: int,
    weights: torch.Tensor = None,
    epochs: int = 20,
    lr: float = 0.001
) -> dict:
    class SimpleNN(nn.Module):
        def __init__(self, input_dim: int, num_classes: int):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(256, 128)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(128, num_classes)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu2(x)
            x = self.fc3(x)
            return x

    model = SimpleNN(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training NN classifier on {device} with weights={weights is not None}...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if weights is not None:
                batch_indices = train_loader.dataset.indices if hasattr(train_loader.dataset, 'indices') else torch.arange(len(labels))
                sample_weights = weights[batch_indices[i * len(labels):(i + 1) * len(labels)]].to(device)
                loss = (loss * sample_weights).mean()
            else:
                loss = loss.mean()

            loss.backward()
            optimizer.step()

    model.eval()
    y_true, y_pred, y_score = [], [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs[:, 1].cpu().numpy() if num_classes == 2 else [None] * len(labels))

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary' if num_classes == 2 else 'macro', zero_division=0)
    auc = roc_auc_score(y_true, y_score) if num_classes == 2 else float('nan')

    return {"acc": acc, "auc": auc, "f1": f1}

def train_nn_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray = None,
    hidden_dim: int = 128,
    epochs: int = 100,
    lr: float = 0.001,
    device: torch.device = torch.device("cpu")
) -> dict:
    class SimpleRegressor(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        def forward(self, x):
            return self.model(x)

    model = SimpleRegressor(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='none')

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
    if weights is not None:
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    for _ in range(epochs):
        model.train()
        preds = model(X_train_tensor)
        loss = loss_fn(preds, y_train_tensor)
        if weights is not None:
            loss = (loss.view(-1) * weights_tensor).mean()
        else:
            loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

    with torch.no_grad():
        y_pred_tensor = model(X_val_tensor).squeeze()
        y_pred = y_pred_tensor.cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mae = np.mean(np.abs(y_val - y_pred))
    return {"rmse": rmse, "r2": r2, "mae": mae}

def train_linear_regression(X_train, X_val, y_train, y_val, weights=None) -> dict:
    reg = LinearRegression()
    reg.fit(X_train, y_train, sample_weight=weights)
    y_pred = reg.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mae = np.mean(np.abs(y_val - y_pred))
    return {"rmse": rmse, "r2": r2, "mae": mae}

def train_svm_classifier(X_train, X_val, y_train, y_val, weights=None) -> dict:
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        return {"acc": accuracy_score(y_val, y_pred), "auc": float('nan'), "f1": float('nan')}
    
    clf = SVC(probability=True, class_weight='balanced' if weights is not None else None)
    clf.fit(X_train, y_train, sample_weight=weights)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if len(clf.classes_) == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='binary' if len(unique_classes) == 2 else 'macro')
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else float('nan')
    return {"acc": acc, "auc": auc, "f1": f1}

def train_random_forest_classifier(X_train, X_val, y_train, y_val, weights=None) -> dict:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train, sample_weight=weights)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if clf.n_classes_ == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='binary' if clf.n_classes_ == 2 else 'macro')
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else float('nan')
    return {"acc": acc, "auc": auc, "f1": f1}

def train_xgboost_classifier(X_train, X_val, y_train, y_val, weights=None) -> dict:
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf.fit(X_train, y_train, sample_weight=weights)
    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if clf.n_classes_ == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='binary' if clf.n_classes_ == 2 else 'macro')
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else float('nan')
    return {"acc": acc, "auc": auc, "f1": f1}

def train_naive_bayes_classifier(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray = None
) -> dict:
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        return {"acc": accuracy_score(y_val, y_pred), "auc": float('nan'), "f1": float('nan')}

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if len(clf.classes_) == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='binary' if len(clf.classes_) == 2 else 'macro', zero_division=0)
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else float('nan')

    return {"acc": acc, "auc": auc, "f1": f1}

def train_knn_classifier(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray = None,
    n_neighbors: int = 5
) -> dict:
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        return {"acc": accuracy_score(y_val, y_pred), "auc": float('nan'), "f1": float('nan')}

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    y_proba = clf.predict_proba(X_val)[:, 1] if len(clf.classes_) == 2 else None

    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='binary' if len(clf.classes_) == 2 else 'macro', zero_division=0)
    auc = roc_auc_score(y_val, y_proba) if y_proba is not None else float('nan')

    return {"acc": acc, "auc": auc, "f1": f1}

def train_random_forest_regression(X_train, X_val, y_train, y_val, weights=None) -> dict:
    reg = RandomForestRegressor(random_state=42)
    reg.fit(X_train, y_train, sample_weight=weights)
    y_pred = reg.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mae = np.mean(np.abs(y_val - y_pred))
    return {"rmse": rmse, "r2": r2, "mae": mae}

def train_xgboost_regression(X_train, X_val, y_train, y_val, weights=None) -> dict:
    reg = XGBRegressor()
    reg.fit(X_train, y_train, sample_weight=weights)
    y_pred = reg.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    mae = np.mean(np.abs(y_val - y_pred))
    return {"rmse": rmse, "r2": r2, "mae": mae}

def evaluate_clustering(y_true, y_pred) -> dict:
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    return {"ari": ari, "nmi": nmi, "ami": ami}

def train_kmeans_clustering(X_train: np.ndarray, y_train: np.ndarray, n_clusters: int = None) -> dict:
    if n_clusters is None:
        n_clusters = len(np.unique(y_train))

    model = KMeans(n_clusters=n_clusters, random_state=42)
    y_pred = model.fit_predict(X_train)
    return evaluate_clustering(y_train, y_pred)

def train_dbscan_clustering(X_train: np.ndarray, y_train: np.ndarray, eps: float = 0.5, min_samples: int = 5) -> dict:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    y_pred = model.fit_predict(X_train)
    return evaluate_clustering(y_train, y_pred)

def train_agglomerative_clustering(X_train: np.ndarray, y_train: np.ndarray, n_clusters: int = None) -> dict:
    if n_clusters is None:
        n_clusters = len(np.unique(y_train))

    model = AgglomerativeClustering(n_clusters=n_clusters)
    y_pred = model.fit_predict(X_train)
    return evaluate_clustering(y_train, y_pred)
