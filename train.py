import torch
from torch.utils.data import DataLoader
from dataset import MNISTDataset, AffNISTDataset, ForestFiresDataset
from models import MyFFNetworkForClassification, MyFFNetworkForRegression
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import argparse


def main(args):
    # 1. Hyperparameters
    LEARNING_RATE = args.lr
    EPOCHS = args.e
    BATCH_SIZE = 64
    HIDDEN_DIM = 128
    NUM_HIDDEN_LAYERS = 2

    # 2. Load Data based on dataset argument
    if args.ds == 'mnist':
        print("--- Using MNIST Dataset ---")
        DATA_DIR = 'data/raw/MNIST/'
        DATA_SAVE = 'data/processed/MNIST/'
        train_filepath = DATA_DIR + 'mnist_train.csv'
        test_filepath = DATA_DIR + 'mnist_test.csv'
        train_dataset = MNISTDataset(train_filepath, test_filepath,
                                     split='train', val_size=args.vs, random_state=42)
        val_dataset = MNISTDataset(train_filepath, test_filepath,
                                   split='val', val_size=args.vs, random_state=42)
        test_dataset = MNISTDataset(
            train_filepath, test_filepath, split='test')
        output_dim = 10
    elif args.ds == 'affnist':
        print("--- Using affNIST Dataset ---")
        DATA_DIR = 'data/raw/affNIST/transformed/'
        DATA_SAVE = 'data/processed/affNIST/'
        train_dir = DATA_DIR + 'training_and_validation_batches/'
        test_dir = DATA_DIR + 'test_batches/'
        train_dataset = AffNISTDataset(train_dir, test_dir,
                                       split='train', val_size=args.vs, random_state=42)
        val_dataset = AffNISTDataset(train_dir, test_dir,
                                     split='val', val_size=args.vs, random_state=42)
        test_dataset = AffNISTDataset(train_dir, test_dir, split='test')
        output_dim = 10
    elif args.ds == 'forestfires':
        print("--- Using Forest Fires Dataset (Regression) ---")
        DATA_DIR = 'data/raw/forestfires/'
        DATA_SAVE = 'data/processed/forestfires/'
        filepath = DATA_DIR + 'forestfires.csv'
        # Note: ForestFiresDataset already has test_size parameter
        train_dataset = ForestFiresDataset(filepath, split='train',
                                           val_size=args.vs, test_size=1-args.ts-args.vs)
        val_dataset = ForestFiresDataset(filepath, split='val',
                                         val_size=args.vs, test_size=1-args.ts-args.vs)
        test_dataset = ForestFiresDataset(filepath, split='test',
                                          val_size=args.vs, test_size=1-args.ts-args.vs)
        output_dim = 1
    else:
        raise ValueError(
            "Invalid dataset specified. Choose 'mnist', 'affnist', or 'forestfires'.")

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(
        f"Data split: {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

    # 4. Initialize Model
    if args.ds == 'forestfires':
        model = MyFFNetworkForRegression(
            input_dim=train_dataset.X.shape[1],
            hidden_dim=HIDDEN_DIM,
            output_dim=output_dim,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            initialization_method=args.init,
            use_batch_norm=args.bn
        )
        output_activation = 'identity'
        loss_mode = 'mse'
    else:  # Classification
        model = MyFFNetworkForClassification(
            input_dim=train_dataset.X.shape[1],
            hidden_dim=HIDDEN_DIM,
            output_dim=output_dim,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            initialization_method=args.init,
            use_batch_norm=args.bn
        )
        output_activation = 'softmax'  # softmax here - this is for the forward pass,
        # it specifies the activation function to apply to the final layer to produce the prediction
        loss_mode = 'softmax_ce'  # softmax here - this is for the backward pass,
        # it specifies how to calculate the initial gradient, since we can use a shortcut knowing the loss was calculated with CE
        # and the forward pass was the softmax, so we can use the simplified gradient formula

    # 5. Initialize Optimizer
    optimizer = torch.optim.SGD(model.params.values(), lr=LEARNING_RATE)

    # 6. Training & History Tracking
    history = {'train_loss': [], 'val_loss': []}
    print("\n--- Starting Training ---")
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(
                X_batch, hidden_activation=args.activ, output_activation=output_activation, training=True)
            loss = model.loss(y_pred, y_batch)
            total_train_loss += loss.item()
            model.backward(y_batch, hidden_activation=args.activ,
                           loss_mode=loss_mode)
            optimizer.step()

        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:  # Using val_loader for validation
                y_pred = model.forward(
                    X_batch, hidden_activation=args.activ, output_activation=output_activation, training=False)
                loss = model.loss(y_pred, y_batch)
                total_val_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print("--- Training Finished ---\n")

    # 7. Plot and Save Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(
        f'Training & Validation Loss Curve ({args.ds.upper()}, {args.init}, {args.activ}, LR={args.lr}, Epochs={args.e}, BN={args.bn}), split train={args.ts}, val={args.vs}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    # Create a unique filename for the plot
    if args.bn:
        DATA_SAVE += 'BN-ON/'
    else:
        DATA_SAVE += 'BN-OFF/'
    plot_filename = f'{DATA_SAVE}{args.init}_{args.activ}_{args.ds}_lr{args.lr}_e{args.e}_bn{args.bn}.png'
    plt.savefig(plot_filename)
    print(
        f"Learning curve plot saved to '{plot_filename}'\n")

    # 8. Final Evaluation on Test Set
    print(f"--- Evaluating on {args.ds.upper()} Test Set ---")
    y_true_all, y_pred_all = [], []
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model.forward(
                X_batch, hidden_activation=args.activ, output_activation=output_activation, training=False)
            loss = model.loss(y_pred, y_batch)
            total_test_loss += loss.item()

            # Correctly handle predictions for each task type
            if args.ds == 'forestfires':
                # For regression, we might need to inverse transform if we scaled the target
                # The current ForestFiresDataset log-transforms the target, so predictions are also on a log scale.
                y_pred_all.extend(y_pred.numpy())
            else:  # Classification
                _, predicted_labels = torch.max(y_pred, 1)
                y_pred_all.extend(predicted_labels.numpy())

            y_true_all.extend(y_batch.numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}\n")

    if args.ds == 'forestfires':
        mse = mean_squared_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        print(f"Mean Squared Error (on log-transformed data): {mse:.4f}")
        print(f"R-squared: {r2:.4f}\n")
    else:  # Classification
        accuracy = accuracy_score(y_true_all, y_pred_all)
        print(f"Final Test Accuracy: {accuracy * 100:.2f}%\n")
        print("--- Classification Report ---")
        # The y_pred_all list now contains correct integer labels
        print(classification_report(y_true_all, y_pred_all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a Feed-Forward Neural Network.')
    parser.add_argument('--ds', type=str, default='mnist', choices=['mnist', 'affnist', 'forestfires'],
                        help='The dataset to train on.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--e', type=int, default=20,
                        help='Number of training epochs.')
    parser.add_argument('--bn', action='store_true',
                        help='Use Batch Normalization.')
    # Add new arguments for split sizes
    parser.add_argument('--ts', type=float, default=0.7,
                        help='Training set size (proportion). Default: 0.7')
    parser.add_argument('--vs', type=float, default=0.15,
                        help='Validation set size (proportion). Default: 0.15')
    parser.add_argument('--init', type=str, default='xavier', choices=['xavier', 'he'],
                        help='Weight initialization method. Default: xavier')
    parser.add_argument('--activ', type=str, default='relu', choices=['relu', 'sigmoid', 'tanh', 'softmax', 'identity'],
                        help='Activation function for hidden layers. Default: relu')

    args = parser.parse_args()

    # Validate split proportions
    if args.ts + args.vs >= 1.0:
        raise ValueError(
            "Training + validation split must be less than 1.0 to leave room for test set")
    main(args)
