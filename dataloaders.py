import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
import arff
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, random_split, TensorDataset, Dataset
import os
from utils import train_autoencoder
from models import Autoencoder
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


def load_dataset(dataset_name, subset_size, batch_size, seed, embedding, embed_dim, device):
    print(f"Loading dataset: {dataset_name}")
    np.random.seed(seed)
    if dataset_name == 'adult':
        X_train, X_val, y_train, y_val = load_adult_data(subset_size)
    elif dataset_name == 'cifar10':
        X_train, X_val, y_train, y_val = load_cifar10(subset_size, seed, embedding, embed_dim, device)
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


def generate_cache_key(embedding):
    return hashlib.md5(embedding.encode()).hexdigest()


def extract_resnet18_features(dataset, device, batch_size=256, num_workers=2):
    resnet_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = TransformDataset(dataset, resnet_transform)

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


def load_full_cifar():
    train_ds = datasets.CIFAR10(root='./data', train=True, download=True)
    test_ds = datasets.CIFAR10(root='./data', train=False, download=True)
    return ConcatDataset([train_ds, test_ds])


def load_cifar10(subset_size, seed, embedding, embed_dim, device, cache_dir="feature_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = generate_cache_key(embedding)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")

    # === Load full features from cache or compute ===
    if os.path.exists(cache_path):
        print(f"Loading full cached features from {cache_path}")
        data = joblib.load(cache_path)
        X_full, y_full = data['X'], data['y']
    else:
        print("Extracting features for full CIFAR using pre-trained ResNet18...")
        full_ds = load_full_cifar()
        X_full, y_full = extract_resnet18_features(full_ds, device)
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


def load_mnist_embedded(subset_size, embed_dim=50):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load MNIST training and test sets
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Select random subset for training
    indices_train = np.random.choice(len(train_dataset), subset_size, replace=False)
    X_train = train_dataset.data[indices_train].numpy().reshape(-1, 784) / 255.0
    y_train = train_dataset.targets[indices_train].numpy()
    
    # Select proportional test subset (maintaining ~6:1 train:test ratio)
    test_subset_size = int(np.ceil(subset_size / 6))
    indices_test = np.random.choice(len(test_dataset), test_subset_size, replace=False)
    X_val = test_dataset.data[indices_test].numpy().reshape(-1, 784) / 255.0
    y_val = test_dataset.targets[indices_test].numpy()
    
    # Apply PCA
    pca = PCA(n_components=embed_dim)
    X_train_embedded = pca.fit_transform(X_train)
    X_val_embedded = pca.transform(X_val)
    
    return X_train_embedded, X_val_embedded, y_train, y_val


def load_mnist_embedded_autoencoder(subset_size, embed_dim=50, autoencoder_path=None, train_autoencoder_if_not_found=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) # Normalizes to [-1, 1]
    ])
    
    # Load MNIST training and test sets
    # Ensure data directory exists
    if not os.path.exists('./data'):
        os.makedirs('./data')
        
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Select random subset for training
    indices_train = np.random.choice(len(train_dataset), subset_size, replace=False)
    
    # Create Subset datasets for efficient loading with DataLoader
    train_subset_dataset = Subset(train_dataset, indices_train)
    
    # Select proportional test subset (maintaining ~6:1 train:test ratio)
    test_subset_size = int(np.ceil(subset_size / 6))
    indices_test = np.random.choice(len(test_dataset), test_subset_size, replace=False)
    test_subset_dataset = Subset(test_dataset, indices_test)
    
    # Prepare DataLoaders
    # Using batch_size for training autoencoder and for embedding extraction
    batch_size = 256 
    train_loader = DataLoader(train_subset_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Autoencoder
    input_dim = 28 * 28 # MNIST image size
    autoencoder = Autoencoder(input_dim, embed_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autoencoder.to(device)

    # Load or train Autoencoder
    if autoencoder_path and os.path.exists(autoencoder_path):
        print(f"Loading pre-trained Autoencoder from {autoencoder_path}")
        autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device, weights_only=True))
    elif train_autoencoder_if_not_found:
        print("Pre-trained Autoencoder not found or path not provided. Training a new one.")
        # Train on the full training dataset for better generalization if subset_size is small
        # Or train on the train_subset_dataset if you want the autoencoder to learn from the same data
        full_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        train_autoencoder(autoencoder, full_train_loader, epochs=5, device=device) # Train on full MNIST train set
        if autoencoder_path:
            os.makedirs(os.path.dirname(autoencoder_path), exist_ok=True)
            torch.save(autoencoder.state_dict(), autoencoder_path)
            print(f"Saved trained Autoencoder to {autoencoder_path}")
    else:
        raise FileNotFoundError(f"Autoencoder path '{autoencoder_path}' not found and 'train_autoencoder_if_not_found' is False.")

    # Extract embeddings
    autoencoder.eval() # Set to evaluation mode
    
    X_train_embedded_list = []
    y_train_list = []
    with torch.no_grad():
        for data, targets in train_loader:
            img = data.view(data.size(0), -1).to(device)
            embeddings = autoencoder.encoder(img).cpu().numpy()
            X_train_embedded_list.append(embeddings)
            y_train_list.append(targets.numpy())
    
    X_val_embedded_list = []
    y_val_list = []
    with torch.no_grad():
        for data, targets in test_loader:
            img = data.view(data.size(0), -1).to(device)
            embeddings = autoencoder.encoder(img).cpu().numpy()
            X_val_embedded_list.append(embeddings)
            y_val_list.append(targets.numpy())

    X_train_embedded = np.vstack(X_train_embedded_list)
    y_train = np.hstack(y_train_list)
    X_val_embedded = np.vstack(X_val_embedded_list)
    y_val = np.hstack(y_val_list)

    return X_train_embedded, X_val_embedded, y_train, y_val


def load_cifar10_encoded(subset_size, embed_dim=512): # embed_dim is typically the output dimension of your chosen encoder layer
    """
    Loads CIFAR-10 data and extracts features using a pretrained ResNet-18 encoder.

    Args:
        subset_size (int): The number of samples to use for the training subset.
        embed_dim (int): The desired dimension of the embedded features.
                         This should match the output dimension of the chosen
                         encoder layer.

    Returns:
        tuple: X_train_encoded, X_val_encoded, y_train, y_val
               where X_train_encoded and X_val_encoded are the encoded features,
               and y_train and y_val are the corresponding labels.
    """

    # 1. Define Transforms for CIFAR-10
    # CIFAR-10 images are 32x32. Pretrained models like ResNet expect 224x224,
    # so we'll need to resize. Normalization also needs to match ImageNet's
    # mean and standard deviation.
    transform = transforms.Compose([
        transforms.Resize(224), # Resize for pretrained models like ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load CIFAR-10 Training and Test Sets
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # 3. Select Random Subsets (as in your original function)
    indices_train = np.random.choice(len(train_dataset), subset_size, replace=False)
    # Using Subset to get actual data based on indices
    train_subset = torch.utils.data.Subset(train_dataset, indices_train)
    y_train = np.array([train_dataset.targets[i] for i in indices_train])

    test_subset_size = int(np.ceil(subset_size / 6))
    indices_test = np.random.choice(len(test_dataset), test_subset_size, replace=False)
    test_subset = torch.utils.data.Subset(test_dataset, indices_test)
    y_val = np.array([test_dataset.targets[i] for i in indices_test])

    # Create DataLoaders for efficient batch processing
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=False, num_workers=2)
    val_loader = torch.utils.data.DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=2)


    # 4. Load a Pretrained Encoder (e.g., ResNet-18)
    # We'll use a ResNet-18 pretrained on ImageNet.
    # We remove the final classification layer to get the features.
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # Remove the final fully connected layer (classifier)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.eval() # Set the model to evaluation mode

    # Determine the device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 5. Extract Features
    def extract_features(data_loader, model, device):
        features_list = []
        with torch.no_grad(): # Disable gradient calculation for inference
            for images, _ in data_loader:
                images = images.to(device)
                features = model(images)
                # Flatten the features (e.g., from (batch_size, 512, 1, 1) to (batch_size, 512))
                features_list.append(features.view(features.size(0), -1).cpu().numpy())
        return np.vstack(features_list)

    print("Extracting training features...")
    X_train_encoded = extract_features(train_loader, model, device)
    print("Extracting validation features...")
    X_val_encoded = extract_features(val_loader, model, device)

    # 6. Optional: Apply PCA after feature extraction if embed_dim is smaller
    # than the encoder's output dimension.
    # ResNet-18's last feature layer (before the FC layer) outputs 512 features.
    # If you want a smaller embed_dim, you can apply PCA on the extracted features.
    if embed_dim < X_train_encoded.shape[1]:
        print(f"Applying PCA to reduce feature dimension from {X_train_encoded.shape[1]} to {embed_dim}")
        pca = PCA(n_components=embed_dim)
        X_train_encoded = pca.fit_transform(X_train_encoded)
        X_val_encoded = pca.transform(X_val_encoded)
    elif embed_dim > X_train_encoded.shape[1]:
        print(f"Warning: Desired embed_dim ({embed_dim}) is greater than the encoder's output dimension ({X_train_encoded.shape[1]}). No PCA applied for reduction.")
        # In this case, embed_dim effectively becomes X_train_encoded.shape[1]
        embed_dim = X_train_encoded.shape[1]


    return X_train_encoded, X_val_encoded, y_train, y_val