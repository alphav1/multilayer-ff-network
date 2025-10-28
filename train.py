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
    EPOCHS = args.epochs
    BATCH_SIZE = 64
    HIDDEN_DIM = 128
    NUM_HIDDEN_LAYERS = 2

    # 2. Load Data based on dataset argument
    if args.dataset == 'mnist':
        print("--- Using MNIST Dataset ---")
        DATA_DIR = 'data/raw/MNIST/'
        DATA_SAVE = 'data/processed/MNIST/'
        train_filepath = DATA_DIR + 'mnist_train.csv'
        test_filepath = DATA_DIR + 'mnist_test.csv'
        train_dataset = MNISTDataset(filepath=train_filepath)
        test_dataset = MNISTDataset(filepath=test_filepath)
        output_dim = 10
    elif args.dataset == 'affnist':
        print("--- Using affNIST Dataset ---")
        DATA_DIR = 'data/raw/affNIST/transformed/'
        DATA_SAVE = 'data/processed/affNIST/'
        train_dir = DATA_DIR + 'training_and_validation_batches/'
        test_dir = DATA_DIR + 'test_batches/'
        train_dataset = AffNISTDataset(dirpath=train_dir)
        test_dataset = AffNISTDataset(dirpath=test_dir)
        output_dim = 10
    elif args.dataset == 'forestfires':
        print("--- Using Forest Fires Dataset (Regression) ---")
        DATA_DIR = 'data/raw/forestfires/'
        DATA_SAVE = 'data/processed/forestfires/'
        filepath = DATA_DIR + 'forestfires.csv'
        train_dataset = ForestFiresDataset(filepath=filepath, train=True)
        test_dataset = ForestFiresDataset(filepath=filepath, train=False)
        output_dim = 1
    else:
        raise ValueError(
            "Invalid dataset specified. Choose 'mnist', 'affnist', or 'forestfires'.")

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    if args.dataset == 'forestfires':
        model = MyFFNetworkForRegression(
            input_dim=train_dataset.X.shape[1],
            hidden_dim=HIDDEN_DIM,
            output_dim=output_dim,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            initialization_method='xavier',
            hidden_activation='relu'
        )
        output_activation = 'identity'
        loss_mode = 'mse'
    else:  # Classification
        model = MyFFNetworkForClassification(
            input_dim=train_dataset.X.shape[1],
            hidden_dim=HIDDEN_DIM,
            output_dim=output_dim,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            initialization_method='xavier',
        )
        output_activation = 'softmax'
        loss_mode = 'softmax_ce'

    # 5. Initialize Optimizer
    optimizer = torch.optim.SGD(model.params.values(), lr=LEARNING_RATE)

    # 6. Training & History Tracking
    history = {'train_loss': [], 'val_loss': []}
    print("--- Starting Training ---")
    for epoch in range(EPOCHS):
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model.forward(
                X_batch, hidden_activation='relu', output_activation=output_activation)
            loss = model.loss(y_pred, y_batch)
            total_train_loss += loss.item()
            model.backward(y_batch, hidden_activation='relu',
                           loss_mode=loss_mode)
            optimizer.step()

        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:  # Using test_loader as validation
                y_pred = model.forward(
                    X_batch, hidden_activation='relu', output_activation=output_activation)
                loss = model.loss(y_pred, y_batch)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'] = history.get('val_loss', []) + [avg_val_loss]
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    print("--- Training Finished ---\n")

    # 7. Plot and Save Learning Curve
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title(
        f'Training Loss Curve ({args.dataset.upper()}) with LR={LEARNING_RATE} across {EPOCHS} epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{DATA_SAVE}learning_curve_{args.dataset}.png')
    print(
        f"Learning curve plot saved to '{DATA_SAVE}learning_curve_{args.dataset}.png'\n")

    # 8. Final Evaluation on Test Set
    print(f"--- Evaluating on {args.dataset.upper()} Test Set ---")
    y_true_all, y_pred_all = [], []
    total_test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model.forward(
                X_batch, hidden_activation='relu', output_activation=output_activation)
            loss = model.loss(y_pred, y_batch)
            total_test_loss += loss.item()

            # Correctly handle predictions for each task type
            if args.dataset == 'forestfires':
                y_pred_all.extend(y_pred.numpy())
            else:  # Classification
                _, predicted_labels = torch.max(y_pred, 1)
                y_pred_all.extend(predicted_labels.numpy())

            y_true_all.extend(y_batch.numpy())

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss:.4f}\n")

    if args.dataset == 'forestfires':
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
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'affnist', 'forestfires'],
                        help='The dataset to train on.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs.')

    args = parser.parse_args()
    main(args)
