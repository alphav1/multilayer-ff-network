import torch
from torch.utils.data import DataLoader
from dataset import FashionMNISTDataset
from autoencoder import Autoencoder
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")


def visualize_reconstructions(model, test_loader, num_samples=10, save_dir='data/processed/FashionMNIST/'):
    """
    Visualize original images and their reconstructions.
    """
    model_training = False

    # Get a batch of test data
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch[:num_samples]
    y_batch = y_batch[:num_samples]

    with torch.no_grad():
        X_reconstructed = model.forward(X_batch, activation='relu',
                                        output_activation='sigmoid', training=model_training)

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(X_batch[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        label_name = FashionMNISTDataset.get_label_name(y_batch[i].item())
        axes[0, i].set_title(f'{label_name}')

        # Reconstructed
        axes[1, i].imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Reconstructed', fontsize=12)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'reconstructions.png'), dpi=150)
    print(
        f"Reconstruction visualization saved to {save_dir}reconstructions.png")
    plt.close()


def visualize_hidden_features(model, test_loader, layer_idx=2, num_features=64, save_dir='data/processed/FashionMNIST/'):
    """
    Visualize the learned features in hidden layers.
    For autoencoder, we visualize the weights as images.
    """
    # Get weights from specified layer
    W = model.params[f'W{layer_idx}'].data.numpy()

    # Determine grid size
    n_features = min(num_features, W.shape[1])
    grid_size = int(np.ceil(np.sqrt(n_features)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        if i < n_features:
            # Get the weights for this feature
            feature = W[:, i].reshape(28, 28) if W.shape[0] == 784 else W[:, i]

            # Normalize for visualization
            feature = (feature - feature.min()) / \
                (feature.max() - feature.min() + 1e-8)

            if W.shape[0] == 784:
                axes[i].imshow(feature, cmap='viridis')
            else:
                # For non-input layers, show as bar chart
                axes[i].bar(range(len(feature)), feature)
            axes[i].axis('off')
            axes[i].set_title(f'F{i}', fontsize=8)
        else:
            axes[i].axis('off')

    plt.suptitle(f'Learned Features in Layer {layer_idx}', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, f'hidden_features_layer{layer_idx}.png'), dpi=150)
    print(
        f"Hidden features visualization saved to {save_dir}hidden_features_layer{layer_idx}.png")
    plt.close()


def visualize_bottleneck_representations(model, test_loader, save_dir='data/processed/FashionMNIST/'):
    """
    Visualize bottleneck (encoded) representations as images.
    """
    model_training = False
    num_samples = 10

    # Get a batch of test data
    X_batch, y_batch = next(iter(test_loader))
    X_batch = X_batch[:num_samples]
    y_batch = y_batch[:num_samples]

    # Get bottleneck representations
    with torch.no_grad():
        bottleneck = model.encode(X_batch, training=model_training)

    # Plot
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))

    for i in range(num_samples):
        # Original
        axes[0, i].imshow(X_batch[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=12)
        label_name = FashionMNISTDataset.get_label_name(y_batch[i].item())
        axes[0, i].set_title(f'{label_name}')

        # Bottleneck representation
        bottleneck_img = bottleneck[i].numpy()
        # Reshape for visualization
        side_len = int(np.ceil(np.sqrt(len(bottleneck_img))))
        padded = np.zeros(side_len * side_len)
        padded[:len(bottleneck_img)] = bottleneck_img
        axes[1, i].imshow(padded.reshape(side_len, side_len), cmap='viridis')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Bottleneck', fontsize=12)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(
        save_dir, 'bottleneck_representations.png'), dpi=150)
    print(
        f"Bottleneck visualization saved to {save_dir}bottleneck_representations.png")
    plt.close()


def visualize_umap(model, test_loader, save_dir='data/processed/FashionMNIST/', max_samples=2000):
    """
    Use UMAP to visualize the encoded representations in 2D.
    """
    if not UMAP_AVAILABLE:
        print("Skipping UMAP visualization (umap-learn not installed)")
        return

    model_training = False

    # Collect encoded representations and labels
    all_encoded = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            encoded = model.encode(X_batch, training=model_training)
            all_encoded.append(encoded.numpy())
            all_labels.append(y_batch.numpy())

            if len(all_encoded) * X_batch.shape[0] >= max_samples:
                break

    all_encoded = np.vstack(all_encoded)[:max_samples]
    all_labels = np.concatenate(all_labels)[:max_samples]

    print(f"Running UMAP on {len(all_encoded)} samples...")

    # Apply UMAP
    reducer = umap.UMAP(n_components=2, random_state=42,
                        n_neighbors=15, min_dist=0.1)
    embedding = reducer.fit_transform(all_encoded)

    # Plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=all_labels, cmap='tab10', s=5, alpha=0.6)

    # Create custom legend with class names
    handles = []
    for label_idx in range(10):
        label_name = FashionMNISTDataset.get_label_name(label_idx)
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=plt.cm.tab10(label_idx/9),
                                  markersize=8, label=label_name))
    plt.legend(handles=handles, title='Classes', loc='best')

    plt.title('UMAP Projection of Encoded Representations')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid(True, alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'umap_visualization.png'),
                dpi=150, bbox_inches='tight')
    print(
        f"UMAP visualization saved to {os.path.join(save_dir, 'umap_visualization.png')}")
    plt.close()


def main(args):
    """
    Main function for Part 1: Initial Exploration
    """
    print("=== Part 1: Autoencoder Exploration ===\n")

    # Load test data
    FASHION_DIR = 'data/raw/FashionMNIST/'
    MNIST_DIR = 'data/raw/MNIST/'

    # Try Fashion MNIST first (with hyphens)
    train_filepath = FASHION_DIR + 'fashion-mnist_train.csv'
    test_filepath = FASHION_DIR + 'fashion-mnist_test.csv'

    # Check if files exist, fallback to regular MNIST if not
    if not os.path.exists(train_filepath):
        print("Warning: Fashion MNIST not found, using regular MNIST")
        train_filepath = MNIST_DIR + 'mnist_train.csv'
        test_filepath = MNIST_DIR + 'mnist_test.csv'

    test_dataset = FashionMNISTDataset(
        train_filepath, test_filepath, split='test')
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {len(test_dataset)} test samples.")

    # Load pre-trained model
    print(f"\nLoading model from {args.model_path}...")
    model = Autoencoder.load_model(args.model_path)

    # Extract directory structure from model path if not explicitly provided
    if args.save_dir == 'data/processed/FashionMNIST/':
        # Parse the model path to extract the directory structure
        # Expected format: models/reg_type/lr_value/e_value/model_name.pth
        model_dir = os.path.dirname(args.model_path)
        # Normalize path to use consistent separators
        model_dir = os.path.normpath(model_dir)
        if 'models' in model_dir:
            # Extract everything after 'models/'
            path_parts = model_dir.split(os.sep)
            if 'models' in path_parts:
                models_idx = path_parts.index('models')
                if len(path_parts) > models_idx + 1:
                    # Reconstruct path: data/processed/FashionMNIST/reg_type/lr/e/
                    relative_path = os.path.join(*path_parts[models_idx + 1:])
                    save_dir = os.path.join(
                        'data/processed/FashionMNIST/', relative_path, '')
                    print(f"Auto-detected save directory: {save_dir}")
                else:
                    save_dir = args.save_dir
            else:
                save_dir = args.save_dir
        else:
            save_dir = args.save_dir
    else:
        save_dir = args.save_dir

    # 1. Visualize reconstructions
    print("\n1. Visualizing reconstructions...")
    visualize_reconstructions(
        model, test_loader, num_samples=args.num_samples, save_dir=save_dir)

    # 2. Visualize hidden layer features
    print("\n2. Visualizing hidden layer features...")
    visualize_hidden_features(
        model, test_loader, layer_idx=1, num_features=64, save_dir=save_dir)

    # 3. Visualize bottleneck representations
    print("\n3. Visualizing bottleneck representations...")
    visualize_bottleneck_representations(model, test_loader, save_dir=save_dir)

    # 4. UMAP visualization
    print("\n4. Creating UMAP visualization...")
    visualize_umap(model, test_loader, save_dir=save_dir,
                   max_samples=args.max_samples)

    print("\n=== Exploration Complete ===")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize and explore pre-trained autoencoder (Part 1)')

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained autoencoder model')
    parser.add_argument('--save_dir', type=str, default='data/processed/FashionMNIST/',
                        help='Directory to save visualizations')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for processing')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--max_samples', type=int, default=2000,
                        help='Maximum samples for UMAP')

    args = parser.parse_args()
    main(args)
