import torch
from torch.utils.data import DataLoader
from dataset import FashionMNISTDataset
from autoencoder import Autoencoder
import matplotlib.pyplot as plt
import argparse
import os


def train_autoencoder(args):
    """
    Pre-train autoencoder on Fashion MNIST dataset.
    """
    # 1. Hyperparameters
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_DIMS = args.hidden_dims  # [encoder1, bottleneck, decoder1]

    # 2. Load Fashion MNIST Data
    print("--- Loading Fashion MNIST Dataset ---")
    DATA_DIR = 'data/raw/MNIST/'  # Assuming Fashion MNIST is in same location

    # Create organized directory structure based on configuration
    base_save_dir = 'data/processed/FashionMNIST/'
    base_model_dir = 'models/'

    # Build directory path: regularization/lr/epochs
    regularization_flags = []
    if args.denoise:
        regularization_flags.append('denoise')
    if args.early_stop:
        regularization_flags.append('earlystop')

    reg_str = '_'.join(
        regularization_flags) if regularization_flags else 'noreg'
    lr_dir = f"lr{args.lr}"
    epochs_dir = f"e{args.epochs}"

    DATA_SAVE = os.path.join(base_save_dir, reg_str, lr_dir, epochs_dir, '')
    MODEL_DIR = os.path.join(base_model_dir, reg_str, lr_dir, epochs_dir, '')

    # Create directories if they don't exist
    os.makedirs(DATA_SAVE, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_filepath = DATA_DIR + 'fashion_mnist_train.csv'
    test_filepath = DATA_DIR + 'fashion_mnist_test.csv'

    # Check if files exist, otherwise use regular MNIST for testing
    if not os.path.exists(train_filepath):
        print("Warning: Fashion MNIST not found, using regular MNIST for testing")
        train_filepath = DATA_DIR + 'mnist_train.csv'
        test_filepath = DATA_DIR + 'mnist_test.csv'

    train_dataset = FashionMNISTDataset(
        train_filepath, test_filepath, split='train', val_size=args.val_size, random_state=42)
    val_dataset = FashionMNISTDataset(
        train_filepath, test_filepath, split='val', val_size=args.val_size, random_state=42)
    test_dataset = FashionMNISTDataset(
        train_filepath, test_filepath, split='test')

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(
        f"Data split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    # 4. Initialize Autoencoder
    input_dim = train_dataset.X.shape[1]  # 784 for 28x28 images

    model = Autoencoder(
        input_dim=input_dim,
        hidden_dims=HIDDEN_DIMS,
        initialization_method=args.init,
        use_batch_norm=args.bn,
        add_noise=args.denoise,
        noise_std=args.noise_std
    )

    # 5. Initialize Optimizer
    # Collect all parameters (weights, biases, and batch norm params if used)
    all_params = list(model.params.values())
    if model.use_batch_norm:
        all_params += list(model.bn_params.values())

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(all_params, lr=LEARNING_RATE)
        print("--- Using SGD Optimizer ---")
    elif args.optimizer == 'sgd_momentum':
        optimizer = torch.optim.SGD(all_params, lr=LEARNING_RATE, momentum=0.9)
        print("--- Using SGD with Momentum Optimizer ---")
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(all_params, lr=LEARNING_RATE)
        print("--- Using Adam Optimizer ---")
    else:
        raise ValueError("Invalid optimizer specified.")

    # 6. Training & History Tracking
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None  # Store best model parameters

    print("\n--- Starting Autoencoder Pre-training ---")

    for epoch in range(EPOCHS):
        # Training phase
        model_training = True
        total_train_loss = 0

        for X_batch, _ in train_loader:  # We don't need labels for autoencoder
            optimizer.zero_grad()

            # Forward pass
            X_reconstructed = model.forward(
                X_batch, activation=args.activation,
                output_activation='sigmoid', training=model_training)

            # Compute loss
            loss = model.loss(X_reconstructed, X_batch)
            total_train_loss += loss.item()

            # Backward pass
            model.backward(X_batch, activation=args.activation)

            # Update weights
            optimizer.step()

        # Validation phase
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, _ in val_loader:
                X_reconstructed = model.forward(
                    X_batch, activation=args.activation,
                    output_activation='sigmoid', training=False)
                loss = model.loss(X_reconstructed, X_batch)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Early stopping check and display
        improvement_msg = ""
        if args.early_stop:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                improvement_msg = " (Updated best model)"
                # Save best model state in memory
                best_model_state = {
                    'params': {k: v.data.clone() for k, v in model.params.items()},
                    'bn_params': {k: v.data.clone() for k, v in model.bn_params.items()} if model.use_batch_norm else {},
                }
                # Also save running stats if using batch norm
                if model.use_batch_norm:
                    best_model_state['running_stats'] = {}
                    for i in range(1, len(model.layer_dims) - 1):
                        best_model_state['running_stats'][f'running_mean{i}'] = getattr(
                            model, f'running_mean{i}').clone()
                        best_model_state['running_stats'][f'running_var{i}'] = getattr(
                            model, f'running_var{i}').clone()
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(
                        f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
                    print(
                        f"Early stopping triggered after {epoch + 1} epochs (patience={args.patience})")
                    # Restore best model
                    if best_model_state is not None:
                        print(
                            f"Restoring best model with validation loss: {best_val_loss:.6f}")
                        for k, v in best_model_state['params'].items():
                            model.params[k].data = v.clone()
                        if model.use_batch_norm:
                            for k, v in best_model_state['bn_params'].items():
                                model.bn_params[k].data = v.clone()
                            for k, v in best_model_state['running_stats'].items():
                                setattr(model, k, v.clone())
                    break

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}{improvement_msg}")

    print("--- Pre-training Finished ---\n")

    # Build filename with configuration flags
    model_filename_parts = [
        f"autoencoder",
        f"{args.init}",
        f"{args.activation}",
        f"dims{'_'.join(map(str, HIDDEN_DIMS))}"
    ]

    if args.denoise:
        model_filename_parts.append(f'denoise{args.noise_std}')
    if args.early_stop:
        model_filename_parts.append(f'earlystop_p{args.patience}')
    if args.bn:
        model_filename_parts.append('bn')

    model_filename = '_'.join(model_filename_parts) + '.pth'

    # Save the model (best if early stopping was used, otherwise final)
    model_save_path = os.path.join(MODEL_DIR, model_filename)
    model.save_model(model_save_path)

    if args.early_stop and best_model_state is not None:
        print(f"Best model saved to: {model_save_path}")
        print(f"Best validation loss: {best_val_loss:.6f}")
    else:
        print(f"Final model saved to: {model_save_path}")

    # 7. Plot and Save Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')

    title_parts = [f'dims={HIDDEN_DIMS}',
                   f'LR={LEARNING_RATE}', f'BS={BATCH_SIZE}']
    if args.denoise:
        title_parts.append(f'Denoise({args.noise_std})')
    if args.early_stop:
        title_parts.append(f'EarlyStop(p={args.patience})')
    if args.bn:
        title_parts.append('BN')

    plt.title(f'Autoencoder Training Loss ({" | ".join(title_parts)})')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plot_filename = os.path.join(DATA_SAVE, 'learning_curve.png')
    plt.savefig(plot_filename)
    print(f"Learning curve saved to '{plot_filename}'")

    # 8. Evaluate on test set
    print("\n--- Evaluating on Test Set ---")
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_reconstructed = model.forward(
                X_batch, activation=args.activation,
                output_activation='sigmoid', training=False)
            loss = model.loss(X_reconstructed, X_batch)
            total_test_loss += loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.6f}\n")

    return model, history, DATA_SAVE, MODEL_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-train Autoencoder on Fashion MNIST.')

    # Dataset parameters
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Validation set size (proportion). Default: 0.15')

    # Model architecture
    parser.add_argument('--hidden_dims', type=int, nargs=3, default=[256, 64, 256],
                        help='Hidden layer dimensions [encoder1, bottleneck, decoder1]. Default: 256 64 256')
    parser.add_argument('--init', type=str, default='xavier', choices=['xavier', 'he', 'constant'],
                        help='Weight initialization method. Default: xavier')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'sigmoid'],
                        help='Activation function for hidden layers. Default: relu')

    # Training parameters
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate. Default: 0.001')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs. Default: 50')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size. Default: 128')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'sgd_momentum', 'adam'],
                        help='Optimizer to use. Default: adam')

    # Regularization
    parser.add_argument('--bn', action='store_true',
                        help='Use Batch Normalization')
    parser.add_argument('--denoise', action='store_true',
                        help='Use denoising autoencoder (add noise to input)')
    parser.add_argument('--noise_std', type=float, default=0.1,
                        help='Standard deviation of noise for denoising. Default: 0.1')
    parser.add_argument('--early_stop', action='store_true',
                        help='Use early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping. Default: 10')

    args = parser.parse_args()

    # Train the autoencoder
    trained_model, training_history, data_save_dir, model_save_dir = train_autoencoder(
        args)

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Models saved to: {model_save_dir}")
    print(f"Plots saved to: {data_save_dir}")
    print(f"\nTo visualize this model, use the saved paths above.")
