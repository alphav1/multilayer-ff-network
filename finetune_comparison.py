import torch
from torch.utils.data import DataLoader
from dataset import FashionMNISTDataset
from autoencoder import Autoencoder
from models import MyFFNetworkForClassification
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.metrics import classification_report, accuracy_score


def extract_encoded_features(model, dataloader):
    """
    Extract encoded (bottleneck) representations from autoencoder.
    """
    all_encoded = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            encoded = model.encode(X_batch, training=False)
            all_encoded.append(encoded)
            all_labels.append(y_batch)

    return torch.cat(all_encoded), torch.cat(all_labels)


def train_classifier(model, train_loader, val_loader, optimizer, epochs, model_name):
    """
    Train a classifier and return training history.
    """
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n--- Training {model_name} ---")

    for epoch in range(epochs):
        # Training
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(X_batch, hidden_activation='relu',
                                   output_activation='softmax', training=True)
            loss = model.loss(y_pred, y_batch)
            total_train_loss += loss.item()
            model.backward(y_batch, hidden_activation='relu',
                           loss_mode='softmax_ce')
            optimizer.step()

        # Validation
        total_val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model.forward(X_batch, hidden_activation='relu',
                                       output_activation='softmax', training=False)
                loss = model.loss(y_pred, y_batch)
                total_val_loss += loss.item()

                _, predicted = torch.max(y_pred, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct / total

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    return history


def evaluate_classifier(model, test_loader, model_name):
    """
    Evaluate classifier on test set.
    """
    y_true_all = []
    y_pred_all = []
    total_test_loss = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model.forward(X_batch, hidden_activation='relu',
                                   output_activation='softmax', training=False)
            loss = model.loss(y_pred, y_batch)
            total_test_loss += loss.item()

            _, predicted = torch.max(y_pred, 1)
            y_pred_all.extend(predicted.numpy())
            y_true_all.extend(y_batch.numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    accuracy = accuracy_score(y_true_all, y_pred_all)

    print(f"\n=== {model_name} Test Results ===")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_true_all, y_pred_all))

    return accuracy, avg_test_loss


def main(args):
    """
    Part 2: Fine-tuning
    Compare classifier trained on:
    1. Raw pixel data (from scratch)
    2. Encoded representations from pre-trained autoencoder
    """
    print("=== Part 2: Fine-tuning Comparison ===\n")

    # Load data
    DATA_DIR = 'data/raw/MNIST/'
    train_filepath = DATA_DIR + 'fashion_mnist_train.csv'
    test_filepath = DATA_DIR + 'fashion_mnist_test.csv'

    if not os.path.exists(train_filepath):
        print("Warning: Fashion MNIST not found, using regular MNIST")
        train_filepath = DATA_DIR + 'mnist_train.csv'
        test_filepath = DATA_DIR + 'mnist_test.csv'

    # Raw data loaders
    train_dataset_raw = FashionMNISTDataset(
        train_filepath, test_filepath, split='train', val_size=args.val_size, random_state=42)
    val_dataset_raw = FashionMNISTDataset(
        train_filepath, test_filepath, split='val', val_size=args.val_size, random_state=42)
    test_dataset_raw = FashionMNISTDataset(
        train_filepath, test_filepath, split='test')

    train_loader_raw = DataLoader(
        train_dataset_raw, batch_size=args.batch_size, shuffle=True)
    val_loader_raw = DataLoader(
        val_dataset_raw, batch_size=args.batch_size, shuffle=False)
    test_loader_raw = DataLoader(
        test_dataset_raw, batch_size=args.batch_size, shuffle=False)

    print(
        f"Loaded datasets: {len(train_dataset_raw)} train, {len(val_dataset_raw)} val, {len(test_dataset_raw)} test")

    # ========== Model 1: Train from scratch on raw data ==========
    print("\n" + "="*60)
    print("MODEL 1: Training from scratch on RAW pixel data")
    print("="*60)

    model_raw = MyFFNetworkForClassification(
        input_dim=784,  # Raw 28x28 images
        hidden_dim=args.hidden_dim,
        output_dim=10,  # 10 classes
        num_hidden_layers=args.num_layers,
        initialization_method='xavier',
        use_batch_norm=args.bn
    )

    optimizer_raw = torch.optim.Adam(model_raw.params.values(), lr=args.lr)
    history_raw = train_classifier(model_raw, train_loader_raw, val_loader_raw,
                                   optimizer_raw, args.epochs, "Raw Classifier")

    acc_raw, loss_raw = evaluate_classifier(
        model_raw, test_loader_raw, "Raw Classifier")

    # ========== Model 2: Train on encoded representations ==========
    if args.autoencoder_path:
        print("\n" + "="*60)
        print("MODEL 2: Training on ENCODED representations")
        print("="*60)

        # Load pre-trained autoencoder
        print(f"\nLoading autoencoder from {args.autoencoder_path}...")
        autoencoder = Autoencoder.load_model(args.autoencoder_path)

        # Extract encoded features
        print("Extracting encoded features...")
        X_train_encoded, y_train = extract_encoded_features(
            autoencoder, train_loader_raw)
        X_val_encoded, y_val = extract_encoded_features(
            autoencoder, val_loader_raw)
        X_test_encoded, y_test = extract_encoded_features(
            autoencoder, test_loader_raw)

        # Create datasets from encoded features
        class EncodedDataset(torch.utils.data.Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_dataset_encoded = EncodedDataset(X_train_encoded, y_train)
        val_dataset_encoded = EncodedDataset(X_val_encoded, y_val)
        test_dataset_encoded = EncodedDataset(X_test_encoded, y_test)

        train_loader_encoded = DataLoader(
            train_dataset_encoded, batch_size=args.batch_size, shuffle=True)
        val_loader_encoded = DataLoader(
            val_dataset_encoded, batch_size=args.batch_size, shuffle=False)
        test_loader_encoded = DataLoader(
            test_dataset_encoded, batch_size=args.batch_size, shuffle=False)

        # Create classifier for encoded data
        encoded_dim = X_train_encoded.shape[1]  # Bottleneck dimension
        print(f"Encoded feature dimension: {encoded_dim}")

        model_encoded = MyFFNetworkForClassification(
            input_dim=encoded_dim,
            hidden_dim=args.hidden_dim,
            output_dim=10,
            num_hidden_layers=args.num_layers,
            initialization_method='xavier',
            use_batch_norm=args.bn
        )

        optimizer_encoded = torch.optim.Adam(
            model_encoded.params.values(), lr=args.lr)
        history_encoded = train_classifier(model_encoded, train_loader_encoded, val_loader_encoded,
                                           optimizer_encoded, args.epochs, "Encoded Classifier")

        acc_encoded, loss_encoded = evaluate_classifier(
            model_encoded, test_loader_encoded, "Encoded Classifier")

        # ========== Comparison ==========
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(
            f"Raw Classifier      - Test Acc: {acc_raw*100:.2f}% | Test Loss: {loss_raw:.4f}")
        print(
            f"Encoded Classifier  - Test Acc: {acc_encoded*100:.2f}% | Test Loss: {loss_encoded:.4f}")
        print(
            f"Improvement: {(acc_encoded - acc_raw)*100:.2f} percentage points")

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss comparison
        axes[0].plot(history_raw['val_loss'], label='Raw', linewidth=2)
        axes[0].plot(history_encoded['val_loss'], label='Encoded', linewidth=2)
        axes[0].set_title('Validation Loss Comparison')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy comparison
        axes[1].plot(history_raw['val_acc'], label='Raw', linewidth=2)
        axes[1].plot(history_encoded['val_acc'], label='Encoded', linewidth=2)
        axes[1].set_title('Validation Accuracy Comparison')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = 'data/processed/FashionMNIST/finetuning_comparison.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"\nComparison plot saved to {save_path}")

    else:
        print("\nNo autoencoder path provided. Only raw classifier trained.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Part 2: Fine-tuning comparison')

    parser.add_argument('--autoencoder_path', type=str, default=None,
                        help='Path to pre-trained autoencoder (optional)')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Validation set size. Default: 0.15')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Default: 128')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default: 0.001')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs. Default: 30')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden layer dimension. Default: 128')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of hidden layers. Default: 2')
    parser.add_argument('--bn', action='store_true',
                        help='Use batch normalization')

    args = parser.parse_args()
    main(args)
