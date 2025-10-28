import torch
import matplotlib.pyplot as plt
from dataset import MNISTDataset


def visualize_sample(dataset: MNISTDataset, idx: int = 0):
    """Visualizes a single sample from the dataset."""
    # Get a sample
    image_tensor, label_tensor = dataset[idx]

    # Reshape the 1D tensor (784 pixels) to a 2D image (28x28)
    image = image_tensor.reshape(28, 28)

    # Display the image
    plt.imshow(image, cmap='gray')
    plt.title(f"Label: {label_tensor.item()}")
    plt.axis('off')  # Hide axes
    plt.show()


if __name__ == '__main__':
    # Path to the training data
    DATA_DIR = 'data/raw/MNIST/'
    train_filepath = DATA_DIR + 'mnist_train.csv'
    test_filepath = DATA_DIR + 'mnist_test.csv'

    # Create the dataset
    train_dataset = MNISTDataset(filepath=train_filepath)

    # Visualize the first sample (index 0)
    print("Displaying the first sample from the training data...")
    visualize_sample(train_dataset, idx=0)

    # You can uncomment the lines below to see other samples
    # print("Displaying another sample (index 100)...")
    # visualize_sample(train_dataset, idx=100)
