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
    """
    Handles loading, splitting, and preparation of the MNIST dataset.
    The test set is kept separate, and the training data is split into train/validation.
    """

    # Label names for MNIST (digits 0-9)
    LABEL_NAMES = {i: str(i) for i in range(10)}

    def __init__(self, train_filepath: str, test_filepath: str, split: str = 'train', val_size: float = 0.2, random_state: int = 42):
        """
        Args:
            train_filepath (str): Path to the training CSV file.
            test_filepath (str): Path to the testing CSV file.
            split (str): The partition of data to use ('train', 'val', or 'test').
            val_size (float): The proportion of the training data to use for validation.
            random_state (int): Seed for the random split.
        """
        print(f"Loading MNIST data for '{split}' split...")

        if split == 'test':
            df = pd.read_csv(test_filepath)
            labels = df.iloc[:, 0].values
            features = df.iloc[:, 1:].values.astype(np.float32)
        else:  # 'train' or 'val'
            df = pd.read_csv(train_filepath)
            full_labels = df.iloc[:, 0].values
            full_features = df.iloc[:, 1:].values.astype(np.float32)

            # Split the full training data into a smaller training set and a validation set
            train_features, val_features, train_labels, val_labels = train_test_split(
                full_features, full_labels, test_size=val_size, random_state=random_state
            )

            if split == 'train':
                features, labels = train_features, train_labels
            else:  # 'val'
                features, labels = val_features, val_labels

        # Normalization: Scale pixel values to [0, 1]
        features /= 255.0

        # Convert to PyTorch tensors
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns one data sample (feature vector and corresponding label) at the given index. """
        return self.X[idx], self.y[idx]

    @staticmethod
    def get_label_name(label: int) -> str:
        """Convert numeric label to human-readable name."""
        return MNISTDataset.LABEL_NAMES.get(label, f'Unknown-{label}')


class AffNISTDataset(Dataset):
    """
    Handles loading and preparation of the affNIST dataset from .mat batch files.
    The test set is kept separate, and the training data is split into train/validation.
    """

    def __init__(self, train_dir: str, test_dir: str, split: str = 'train', val_size: float = 0.2, random_state: int = 42):
        """
        Args:
            train_dir (str): Path to the directory with training .mat files.
            test_dir (str): Path to the directory with testing .mat files.
            split (str): The partition of data to use ('train', 'val', or 'test').
            val_size (float): The proportion of the training data to use for validation.
            random_state (int): Seed for the random split.
        """
        print(f"Loading affNIST data for '{split}' split...")

        if split == 'test':
            source_dir = test_dir
        else:  # 'train' or 'val'
            source_dir = train_dir

        all_features = []
        all_labels = []
        for filename in sorted(os.listdir(source_dir)):
            if filename.endswith('.mat'):
                filepath = os.path.join(source_dir, filename)
                data = loadmat(filepath)
                all_features.append(
                    data['affNISTdata']['image'].T.astype(np.float32))
                all_labels.append(data['affNISTdata']['label_int'])

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)

        if split in ['train', 'val']:
            train_features, val_features, train_labels, val_labels = train_test_split(
                features, labels, test_size=val_size, random_state=random_state
            )
            if split == 'train':
                features, labels = train_features, train_labels
            else:  # 'val'
                features, labels = val_features, val_labels

        # Normalization: Scale pixel values to [0, 1]
        # turn raw pixel value (large and small numbers) to correct input type
        features /= 255.0

        # Convert to PyTorch tensors
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels).long()
        print(f"Loaded {len(self.y)} samples for '{split}' split.")

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class FashionMNISTDataset(Dataset):
    """
    Handles loading, splitting, and preparation of the Fashion MNIST dataset.
    Uses the same structure as MNIST dataset.
    """

    # Label names for Fashion MNIST
    LABEL_NAMES = {
        0: 'T-shirt/top',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle boot'
    }

    def __init__(self, train_filepath: str, test_filepath: str, split: str = 'train',
                 val_size: float = 0.2, random_state: int = 42):
        """
        Args:
            train_filepath (str): Path to the training CSV file.
            test_filepath (str): Path to the testing CSV file.
            split (str): The partition of data to use ('train', 'val', or 'test').
            val_size (float): The proportion of the training data to use for validation.
            random_state (int): Seed for the random split.
        """
        print(f"Loading Fashion MNIST data for '{split}' split...")

        if split == 'test':
            df = pd.read_csv(test_filepath)
            labels = df.iloc[:, 0].values
            features = df.iloc[:, 1:].values.astype(np.float32)
        else:  # 'train' or 'val'
            df = pd.read_csv(train_filepath)
            full_labels = df.iloc[:, 0].values
            full_features = df.iloc[:, 1:].values.astype(np.float32)

            # Split the full training data into a smaller training set and a validation set
            train_features, val_features, train_labels, val_labels = train_test_split(
                full_features, full_labels, test_size=val_size, random_state=random_state
            )

            if split == 'train':
                features, labels = train_features, train_labels
            else:  # 'val'
                features, labels = val_features, val_labels

        # Normalization: Scale pixel values to [0, 1]
        features /= 255.0

        # Convert to PyTorch tensors
        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Returns one data sample (feature vector and corresponding label) at the given index. """
        return self.X[idx], self.y[idx]

    @staticmethod
    def get_label_name(label: int) -> str:
        """Convert numeric label to human-readable name."""
        return FashionMNISTDataset.LABEL_NAMES.get(label, f'Unknown-{label}')


class ForestFiresDataset(Dataset):
    """
    Handles loading, splitting, and preparation of the Forest Fires dataset.
    The entire dataset is split into train, validation, and test sets.
    """
    # preprocessing
    _scaler = StandardScaler()  # creates an instance of standardscaler
    # standardizes features by removing the mean and scaling to unit variance z = (x - mean) / std
    _is_fitted = False
    # bolean flag to track if scaler has been fitted to the data

    def __init__(self, filepath: str, split: str = 'train', val_size: float = 0.15, test_size: float = 0.15, random_state: int = 42):
        """
        Args:
            filepath (str): Path to the forestfires.csv file.
            split (str): The partition of data to use ('train', 'val', or 'test').
            val_size (float): Proportion for the validation set.
            test_size (float): Proportion for the test set.
            random_state (int): Seed for the random splits.
        """
        print(f"Loading Forest Fires data for '{split}' split...")
        df = pd.read_csv(filepath)
        df = pd.get_dummies(df, columns=['month', 'day'], drop_first=True)

        # Separate features (X) and target (y)
        X = df.drop('area', axis=1).values.astype(np.float32)
        y = df['area'].values.astype(np.float32)

        # Split into initial training set and a temporary test set
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        # Split the initial training set into a final training set and a validation set
        # e.g. 1 - full data, test size = 0.2 (20%), then if val_size 0.16, val split = 0.2
        val_split_ratio = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            # e.g. 20% of the remaining 80% is validation
            X_train_full, y_train_full, test_size=val_split_ratio, random_state=random_state)

        if split == 'train':
            if not ForestFiresDataset._is_fitted:
                self.X_raw = X_train
                ForestFiresDataset._scaler.fit(self.X_raw)  # scaler run
                ForestFiresDataset._is_fitted = True  # change fitted
            self.y_raw = y_train
        elif split == 'val':
            self.X_raw = X_val
            self.y_raw = y_val
        else:  # 'test'
            self.X_raw = X_test
            self.y_raw = y_test

        # Standardize features
        self.X = torch.from_numpy(
            ForestFiresDataset._scaler.transform(self.X_raw))
        # Apply log transform to the skewed target variable and reshape
        self.y = torch.from_numpy(np.log1p(self.y_raw)).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
