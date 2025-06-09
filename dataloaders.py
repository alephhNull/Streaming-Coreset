import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml


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