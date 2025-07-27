import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, f1_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import time


def train_classifier(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    weights: np.ndarray = None  # Optional sample weights
) -> float:
    unique_classes = np.unique(y_train)

    if len(unique_classes) < 2:
        print(f"  Warning: Training data contains only one class (label: {unique_classes[0]}). Using constant prediction.")
        y_pred = np.full_like(y_val, fill_value=unique_classes[0])
        acc = accuracy_score(y_val, y_pred)
        f1 = float('nan')
        auc = float('nan')  # AUC is undefined with one class
        return acc, auc, f1

        return accuracy_score(y_val, y_pred)

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

    return acc, auc, f1

def train_nn_classifier(train_loader: DataLoader, val_loader: DataLoader, device: torch.device, input_dim: int, num_classes: int, epochs: int = 20, lr: float = 0.001) -> float:
    """
    Trains a simple neural network classifier on the provided data loaders.

    Args:
        train_loader: DataLoader for the training data (features and labels).
        val_loader: DataLoader for the validation data (features and labels).
        device: The device to train on ('cuda' for GPU or 'cpu').
        input_dim: The number of input features (e.g., 512 for ResNet18 features, 3072 for raw CIFAR10).
        num_classes: The number of unique classes in the dataset (e.g., 10 for CIFAR-10).
        epochs: The number of training epochs.
        lr: The learning rate for the optimizer.

    Returns:
        The accuracy of the classifier on the validation set.
    """

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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Training NN classifier on {device} with input_dim: {input_dim}...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = val_correct / val_total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    end_time = time.time()
    print(f"Training finished in {(end_time - start_time):.2f} seconds.")

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    final_accuracy = accuracy_score(y_true, y_pred)
    print(f"Final Validation Accuracy: {final_accuracy:.4f}")

    return final_accuracy


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