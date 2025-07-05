import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
import arff
import pandas as pd
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
from utils import train_autoencoder
from models import Autoencoder

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
#   categorical_cols = []

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