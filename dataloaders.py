import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml, fetch_kddcup99, fetch_covtype
import arff
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split, TensorDataset, Dataset
import os
from torchvision.models import resnet18, ResNet18_Weights
import joblib
import hashlib


class TransformDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transform(x), y

    def __len__(self):
        return len(self.base)

# ResNet18 expects 3-channel 224x224 inputs
class GrayscaleToRGB(torch.nn.Module):
    def forward(self, x):
        return x.expand(3, -1, -1)


def load_dataset(dataset_name, subset_size, batch_size, seed, embedding, embed_dim, device):
    print(f"Loading dataset: {dataset_name}")
    np.random.seed(seed)
    if dataset_name == 'adult':
        X_train, X_val, y_train, y_val = load_adult_data(subset_size)
    elif dataset_name == 'electricity':
        X_train, X_val, y_train, y_val = load_electricity_data(subset_size)
    elif dataset_name == 'cifar10':
        X_train, X_val, y_train, y_val = load_cifar10(subset_size, seed, embedding, embed_dim, device)
    elif dataset_name == 'mnist':
        X_train, X_val, y_train, y_val = load_mnist(subset_size, seed, embedding, embed_dim, device)
    elif dataset_name == 'fashion_mnist':
        X_train, X_val, y_train, y_val = load_fashion_mnist(subset_size, seed, embedding, embed_dim, device)
    elif dataset_name == 'kdd99':
        X_train, X_val, y_train, y_val = load_kdd99_data(subset_size, seed)
    elif dataset_name == 'covtype':
        X_train, X_val, y_train, y_val = load_covtype_data(subset_size, embed_dim, seed)
    elif dataset_name == 'svhn':
        X_train, X_val, y_train, y_val = load_svhn(subset_size, seed, embedding, embed_dim, device)
    elif dataset_name == 'gaussian_mixture':
        X_train, X_val, y_train, y_val = load_gaussian_mixture_2d(subset_size, seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    X_train_tensor = torch.from_numpy(X_train).float().to(device)
    y_train_tensor = torch.from_numpy(y_train).long().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                            batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, X_train, X_val, y_train, y_val


from sklearn.datasets import make_spd_matrix

def load_gaussian_mixture_2d(subset_size, seed, n_classes=1, n_components_per_class=5):
    np.random.seed(seed)
    X, y = [], []
    total_samples = subset_size if subset_size is not None else 5000

    # Highly imbalanced class distribution
    class_probs = np.array([0.7, 0.2, 0.1])[:n_classes]
    class_probs = class_probs / class_probs.sum()
    class_counts = (class_probs * total_samples).astype(int)

    for class_id in range(n_classes):
        n_samples_class = class_counts[class_id]
        weights = np.random.dirichlet(np.ones(n_components_per_class), size=1).flatten()

        for comp_id in range(n_components_per_class):
            n_samples_comp = max(1, int(weights[comp_id] * n_samples_class))

            # Random mean (spread far, but biased towards a quadrant)
            mean = np.random.randn(2) * 3 + np.array([class_id * 4, class_id * -2])

            # Skewed, correlated covariance matrix
            cov = make_spd_matrix(2)
            cov *= np.random.uniform(0.05, 2.0)  # scale differently per component
            if np.random.rand() < 0.5:
                cov[0, 1] *= np.random.uniform(0.8, 0.99)  # high correlation

            samples = np.random.multivariate_normal(mean, cov, size=n_samples_comp)
            X.append(samples)
            y.append(np.full(n_samples_comp, class_id))

    X = np.vstack(X)
    y = np.concatenate(y)

    # Shuffle
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]

    # Train/val split
    n_train = int(0.8 * len(X))
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    print(f"Generated Gaussian Mixture — X_train: {X_train.shape}, X_val: {X_val.shape}, classes: {np.unique(y)}")
    return X_train, X_val, y_train, y_val


def load_full_svhn():
    from torchvision import datasets
    from torch.utils.data import ConcatDataset

    train_ds = datasets.SVHN(root='./data', split='train', download=True)
    test_ds = datasets.SVHN(root='./data', split='test', download=True)

    # SVHN uses 10 to represent the digit '0'
    train_ds.labels[train_ds.labels == 10] = 0
    test_ds.labels[test_ds.labels == 10] = 0

    return ConcatDataset([train_ds, test_ds])


def load_svhn(subset_size, seed, embedding, embed_dim, device, cache_dir="feature_cache"):
    import os
    import numpy as np
    import joblib
    from sklearn.decomposition import PCA
    from torchvision import transforms
    from torch.utils.data import DataLoader

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = generate_cache_key('svhn', embedding)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = joblib.load(cache_path)
        X_full, y_full = data['X'], data['y']
    else:
        print(f"Extracting features for SVHN using embedding: {embedding}")
        full_ds = load_full_svhn()

        if embedding == 'resnet18':
            resnet_transform = transforms.Compose([
                transforms.Resize(224, InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            full_ds = TransformDataset(full_ds, resnet_transform)
            X_full, y_full = extract_resnet18_features(full_ds, device)

        elif embedding == 'pca':
            raw_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # flatten
            ])
            full_ds = TransformDataset(full_ds, raw_transform)
            all_feats, all_labels = [], []

            for img, label in full_ds:
                all_feats.append(img.numpy())
                all_labels.append(label)
            X_full = np.stack(all_feats)
            y_full = np.array(all_labels)

            if embed_dim is not None:
                pca = PCA(n_components=embed_dim)
                X_full = pca.fit_transform(X_full)
                print(f"PCA applied. Reduced to {embed_dim} dimensions.")
        else:
            raw_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            full_ds = TransformDataset(full_ds, raw_transform)
            all_feats, all_labels = [], []
            for img, label in full_ds:
                all_feats.append(img.numpy())
                all_labels.append(label)
            X_full = np.stack(all_feats)
            y_full = np.array(all_labels)
            print(f"Using raw flattened SVHN with dim: {X_full.shape[1]}")

        joblib.dump({'X': X_full, 'y': y_full}, cache_path)
        print(f"Saved features to cache: {cache_path}")

    # Subset and split
    np.random.seed(seed)
    indices = np.arange(len(X_full))
    if subset_size and subset_size < len(indices):
        indices = np.random.choice(indices, subset_size, replace=False)

    X = X_full[indices]
    y = y_full[indices]

    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]

    print(f"Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def load_covtype_data(subset_size, embed_dim, seed):
    cov = fetch_covtype(as_frame=True)
    data = cov.data
    target = cov.target - 1  # Convert from 1-7 to 0-6

    # Optional binary task: make class 2 vs others for imbalance
    # Uncomment below to do binary classification with imbalance
    # target = (target == 1).astype(int)  # Class 2 (index 1) vs others

    if subset_size is not None and subset_size < len(data):
        np.random.seed(seed)
        idx = np.random.choice(len(data), subset_size, replace=False)
        data = data.iloc[idx]
        target = target.iloc[idx]

    scaler = StandardScaler()
    X = scaler.fit_transform(data)
    y = target.to_numpy()

    if embed_dim is not None:
        pca = PCA(n_components=embed_dim)
        X = pca.fit_transform(X)
        print(f"PCA applied. Reduced to {embed_dim} dimensions.")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'Loaded CovType: {len(data)} samples, {X.shape[1]} features')
    return X_train, X_val, y_train, y_val


def load_kdd99_data(subset_size, seed):
    # Fetch the 10% KDD99 subset
    raw = fetch_kddcup99(percent10=True, as_frame=True)
    data = raw.data
    target = raw.target

    # Convert byte strings to regular strings for categorical processing
    data = data.map(lambda x: x.decode() if isinstance(x, bytes) else x)
    target = target.apply(lambda x: x.decode() if isinstance(x, bytes) else x)

    # Optionally subsample the dataset
    if subset_size is not None and subset_size < len(data):
        np.random.seed(seed)
        idx = np.random.choice(len(data), subset_size, replace=False)
        data = data.iloc[idx]
        target = target.iloc[idx]

    # Encode labels: normal -> 0, attack -> 1
    y = (target != 'normal.').astype(int).to_numpy()

    # Separate numerical and categorical features
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns

    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[num_cols])

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat = ohe.fit_transform(data[cat_cols])

    X = np.hstack((X_num, X_cat))

    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f'Loaded KDD99 subset: {len(data)} samples, feature dim: {X.shape[1]}')
    return X_train, X_val, y_train, y_val



def load_electricity_data(subset_size=500):
    electricity = fetch_openml('electricity', version=1, as_frame=True)
    data_df = electricity.data
    target = electricity.target

    # Random subset
    subset_indices = np.random.choice(len(data_df), subset_size, replace=False)
    data_df = data_df.iloc[subset_indices, :]
    target = target.iloc[subset_indices]

    # Encode target ('UP'/'DOWN' to 0/1)
    le_target = LabelEncoder()
    y = le_target.fit_transform(target)

    # Split into training and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(data_df, y, test_size=0.2, random_state=42, stratify=y)

    # Identify numerical and categorical columns
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
    categorical_cols = []

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[categorical_cols])
    X_train_cat = ohe.transform(X_train[categorical_cols])
    X_val_cat = ohe.transform(X_val[categorical_cols])

    scaler = StandardScaler()
    scaler.fit(X_train[numerical_cols])
    X_train_num = scaler.transform(X_train[numerical_cols])
    X_val_num = scaler.transform(X_val[numerical_cols])

    X_train_processed = np.hstack((X_train_num, X_train_cat))
    X_val_processed = np.hstack((X_val_num, X_val_cat))

    print(f'Shapes of X_train: {X_train_processed.shape}, X_val: {X_val_processed.shape}')
    return X_train_processed, X_val_processed, y_train, y_val



def generate_cache_key(dataset, embedding):
    key_str = f"{dataset}_{embedding}"
    return hashlib.md5(key_str.encode()).hexdigest()


def extract_resnet18_features(dataset, device, batch_size=256, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()

    all_feats = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            feats = model(imgs).view(imgs.size(0), -1)
            all_feats.append(feats.cpu())
            all_labels.append(labels)

    X = torch.cat(all_feats, dim=0).numpy()
    y = torch.cat(all_labels, dim=0).numpy()
    return X, y


def load_full_fashion_mnist():
    train_ds = datasets.FashionMNIST(root='./data', train=True, download=True)
    test_ds = datasets.FashionMNIST(root='./data', train=False, download=True)
    return ConcatDataset([train_ds, test_ds])

def load_fashion_mnist(subset_size, seed, embedding, embed_dim, device, cache_dir="feature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = generate_cache_key('fashion_mnist', embedding)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = joblib.load(cache_path)
        X_full, y_full = data['X'], data['y']
    else:
        print(f"Extracting features for Fashion-MNIST using embedding: {embedding}")
        full_ds = load_full_fashion_mnist()

        if embedding == 'resnet18':
            resnet_transform = transforms.Compose([
                transforms.Resize(224, InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                GrayscaleToRGB(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            full_ds = TransformDataset(full_ds, resnet_transform)
            X_full, y_full = extract_resnet18_features(full_ds, device)

        elif embedding == 'pca':
            raw_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # flatten
            ])
            full_ds = TransformDataset(full_ds, raw_transform)
            all_feats, all_labels = [], []

            for img, label in full_ds:
                all_feats.append(img.numpy())
                all_labels.append(label)
            X_full = np.stack(all_feats)
            y_full = np.array(all_labels)

            if embed_dim is not None:
                pca = PCA(n_components=embed_dim)
                X_full = pca.fit_transform(X_full)
                print(f"PCA applied. Reduced to {embed_dim} dimensions.")
        else:
            # No embedding: use raw flattened Fashion-MNIST images
            raw_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            full_ds = TransformDataset(full_ds, raw_transform)
            all_feats, all_labels = [], []
            for img, label in full_ds:
                all_feats.append(img.numpy())
                all_labels.append(label)
            X_full = np.stack(all_feats)
            y_full = np.array(all_labels)
            print(f"Using raw flattened Fashion-MNIST with dim: {X_full.shape[1]}")

        joblib.dump({'X': X_full, 'y': y_full}, cache_path)
        print(f"Saved features to cache: {cache_path}")

    # Subset and split
    np.random.seed(seed)
    indices = np.arange(len(X_full))
    if subset_size and subset_size < len(indices):
        indices = np.random.choice(indices, subset_size, replace=False)

    X = X_full[indices]
    y = y_full[indices]

    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]

    print(f"Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val



def load_full_mnist():
    train_ds = datasets.MNIST(root='./data', train=True, download=True)
    test_ds = datasets.MNIST(root='./data', train=False, download=True)
    return ConcatDataset([train_ds, test_ds])


def load_mnist(subset_size, seed, embedding, embed_dim, device, cache_dir="feature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = generate_cache_key('mnist', embedding)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        data = joblib.load(cache_path)
        X_full, y_full = data['X'], data['y']
    else:
        print(f"Extracting features for MNIST using embedding: {embedding}")
        full_ds = load_full_mnist()

        if embedding == 'resnet18':
            resnet_transform = transforms.Compose([
                transforms.Resize(224, InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                GrayscaleToRGB(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            full_ds = TransformDataset(full_ds, resnet_transform)
            X_full, y_full = extract_resnet18_features(full_ds, device)

        elif embedding == 'pca':
            raw_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))  # flatten
            ])
            full_ds = TransformDataset(full_ds, raw_transform)
            all_feats, all_labels = [], []

            for img, label in full_ds:
                all_feats.append(img.numpy())
                all_labels.append(label)
            X_full = np.stack(all_feats)
            y_full = np.array(all_labels)

            if embed_dim is not None:
                pca = PCA(n_components=embed_dim)
                X_full = pca.fit_transform(X_full)
                print(f"PCA applied. Reduced to {embed_dim} dimensions.")
        else:
            # No embedding: use raw flattened MNIST images
            raw_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.view(-1))
            ])
            full_ds = TransformDataset(full_ds, raw_transform)
            all_feats, all_labels = [], []
            for img, label in full_ds:
                all_feats.append(img.numpy())
                all_labels.append(label)
            X_full = np.stack(all_feats)
            y_full = np.array(all_labels)
            print(f"Using raw flattened MNIST with dim: {X_full.shape[1]}")

        joblib.dump({'X': X_full, 'y': y_full}, cache_path)
        print(f"Saved features to cache: {cache_path}")

    # Subset and split
    np.random.seed(seed)
    indices = np.arange(len(X_full))
    if subset_size and subset_size < len(indices):
        indices = np.random.choice(indices, subset_size, replace=False)

    X = X_full[indices]
    y = y_full[indices]

    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]

    print(f"Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def load_full_cifar():
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True)
    return ConcatDataset([train_ds, test_ds])


def load_cifar10(subset_size, seed, embedding, embed_dim, device, cache_dir="feature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = generate_cache_key('cifar10', embedding)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    # === Load full features from cache or compute ===
    if os.path.exists(cache_path):
        print(f"Loading full cached features from {cache_path}")
        data = joblib.load(cache_path)
        X_full, y_full = data['X'], data['y']
    else:
        print("Extracting features for full CIFAR using pre-trained ResNet18...")
        full_ds = load_full_cifar()
        resnet_transform = transforms.Compose([
            transforms.Resize(224, InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        dataset = TransformDataset(full_ds, resnet_transform)
        X_full, y_full = extract_resnet18_features(dataset, device)
        joblib.dump({'X': X_full, 'y': y_full}, cache_path)
        print(f"Saved full features to cache: {cache_path}")

    # === Subset and split ===
    np.random.seed(seed)
    indices = np.arange(len(X_full))
    if subset_size and subset_size < len(indices):
        indices = np.random.choice(indices, subset_size, replace=False)

    X = X_full[indices]
    y = y_full[indices]

    # Split 80/20
    n_train = int(0.8 * len(X))
    perm = np.random.permutation(len(X))  # shuffle before split
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val     = X[val_idx], y[val_idx]

    # === Apply PCA if requested ===
    if embed_dim is not None:
        pca = PCA(n_components=embed_dim)
        X_train = pca.fit_transform(X_train)
        X_val   = pca.transform(X_val)
        print(f"PCA applied. New feature dimension: {embed_dim}")
    else:
        print(f"No PCA. Original feature dimension: {X_train.shape[1]}")

    print(f"Shapes — X_train: {X_train.shape}, X_val: {X_val.shape}")
    return X_train, X_val, y_train, y_val


def load_adult_data(subset_size=500):
  adult = fetch_openml('adult', version=2, as_frame=True)
  data_df = adult.data
  target = adult.target

  # Handle missing values
  data_df = data_df.replace('?', np.nan)
  data_df = data_df.dropna()
  target = target.loc[data_df.index]

  subset_indices = np.random.choice(len(data_df), subset_size, replace=False)
  data_df = data_df.iloc[subset_indices, :]
  target = target.iloc[subset_indices]

  # Encode target ('<=50K' and '>50K' to 0 and 1)
  le_target = LabelEncoder()
  y = le_target.fit_transform(target)

  # Split into training and validation sets (80% train, 20% validation)
  X_train, X_val, y_train, y_val = train_test_split(data_df, y, test_size=0.2, random_state=42, stratify=y)

  # Identify numerical and categorical columns
  numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
  categorical_cols = X_train.select_dtypes(include=['category']).columns

  ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
  ohe.fit(X_train[categorical_cols])
  X_train_cat = ohe.transform(X_train[categorical_cols])
  X_val_cat = ohe.transform(X_val[categorical_cols])

  # Standardize numerical features
  scaler = StandardScaler()
  scaler.fit(X_train[numerical_cols])
  X_train_num = scaler.transform(X_train[numerical_cols])
  X_val_num = scaler.transform(X_val[numerical_cols])

  # Combine processed features
  X_train_processed = np.hstack((X_train_num, X_train_cat))
  X_val_processed = np.hstack((X_val_num, X_val_cat))

  print(f'Shapes of X_train: {X_train_processed.shape}, X_val: {X_val_processed.shape}')
  return X_train_processed, X_val_processed, y_train, y_val


def load_electricity_tiny(file_path='data/electricity_tiny.arff'):
    # Load the ARFF file
    with open(file_path, 'r') as f:
        dataset = arff.load(f)
    
    # Extract attributes (column names)
    columns = [attr[0] for attr in dataset['attributes']]
    
    # Convert data to a Pandas DataFrame
    data_df = pd.DataFrame(dataset['data'], columns=columns)
    
    # Convert 'class' column to integer (since it's {0,1})
    y = data_df['class'].astype(int).values
    
    # Split into training and validation sets (80% train, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(data_df, y, test_size=0.2, random_state=42, stratify=y)

    # Standardize numerical features
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    
    print(f'Shapes of X_train: {X_train.shape}, X_val: {X_val.shape}')
    return X_train, X_val, y_train, y_val


def load_boston():
  # Load the Boston Housing dataset
  boston = fetch_openml('boston', version=1, as_frame=True)
  data_df = boston.data
  target = boston.target

  # Handle missing values (Boston dataset typically has none, but confirm)
  data_df = data_df.replace('?', np.nan).dropna()
  target = target.loc[data_df.index].to_numpy()

  # Split into training and validation sets (80% train, 20% validation)
  X_train, X_val, y_train, y_val = train_test_split(data_df, target, test_size=0.2, random_state=42)

  # Standardize numerical features
  scaler = StandardScaler()
  scaler.fit(X_train)
  X_train_processed = scaler.transform(X_train)
  X_val_processed = scaler.transform(X_val)

  print(f'Shapes of X_train: {X_train_processed.shape}, X_val: {X_val_processed.shape}')
  return X_train_processed, X_val_processed, y_train, y_val
