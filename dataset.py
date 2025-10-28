import torch
import numpy as np
import pandas as pd
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import scipy.io as spio
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Helper functions for loading affNIST .mat files


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


class MNISTDataset(Dataset):
    """ Handles loading, normalization, and preparation of the MNIST dataset using pytorch Dataset and DataLoader. """

    def __init__(self, filepath: str):
        """
        Loads data from CSV, normalizes, and stores as PyTorch tensors.

        Args:
            filepath (str): Path to the CSV file (e.g., data/raw/MNIST/mnist_train.csv).
        """
        print(f"Loading data from: {filepath}")
        df = pd.read_csv(filepath)

        # Labels are the first column (index 0)
        labels = df.iloc[:, 0].values
        # Features are all other columns (pixels)
        features = df.iloc[:, 1:].values.astype(np.float32)

        # Normalization: Scale pixel values to [0, 1]
        features /= 255.0

        # Convert to PyTorch tensors and store them
        self.X = torch.from_numpy(features)
        # Labels should be LongTensor for Cross-Entropy Loss in classification
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns one data sample (feature vector and corresponding label) at the given index. """
        return self.X[idx], self.y[idx]


class AffNISTDataset(Dataset):
    """ Handles loading and preparation of the affNIST dataset from .mat batch files. """

    def __init__(self, dirpath: str):
        """
        Loads all .mat batch files from a directory, normalizes, and stores as tensors.

        Args:
            dirpath (str): Path to the directory containing .mat batch files.
        """
        print(f"Loading affNIST data from: {dirpath}")

        all_features = []
        all_labels = []

        # Iterate through all .mat files in the directory
        for filename in sorted(os.listdir(dirpath)):
            if filename.endswith('.mat'):
                filepath = os.path.join(dirpath, filename)
                data = loadmat(filepath)

                # Images are (1600, num_samples), transpose to (num_samples, 1600)
                features = data['affNISTdata']['image'].T.astype(np.float32)
                labels = data['affNISTdata']['label_int']

                all_features.append(features)
                all_labels.append(labels)

        # Concatenate data from all batches
        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        # 1. Normalization: Scale pixel values to [0, 1]
        features /= 255.0

        # Convert to PyTorch tensors
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels).long()

        print(f"Loaded {len(self.y)} samples.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class ForestFiresDataset(Dataset):
    """ Handles loading and preparation of the Forest Fires dataset for regression. """

    def __init__(self, filepath: str, train: bool = True, test_size: float = 0.2, random_state: int = 42):
        print(f"Loading Forest Fires data from: {filepath}")
        df = pd.read_csv(filepath)

        # Convert categorical month and day to numerical
        df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)

        # Separate features (X) and target (y)
        X = df.drop('area', axis=1).values.astype(np.float32)
        y = df['area'].values.astype(np.float32)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        # Standardize numerical features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        if train:
            self.X = torch.from_numpy(X_train)
            # Apply log transform to the skewed target variable
            self.y = torch.from_numpy(np.log1p(y_train))
        else:
            self.X = torch.from_numpy(X_test)
            self.y = torch.from_numpy(np.log1p(y_test))

        # Reshape y to be a column vector
        self.y = self.y.view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
