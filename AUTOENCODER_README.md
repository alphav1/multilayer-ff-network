# Autoencoder Assignment - Implementation Guide

This implementation provides a complete autoencoder neural network system for Fashion MNIST, including pre-training, visualization, regularization, and fine-tuning capabilities.

## Project Structure

```
multilayer-ff-network/
├── autoencoder.py              # Autoencoder model implementation
├── dataset.py                  # Dataset loaders (including FashionMNISTDataset)
├── models.py                   # Feed-forward network for classification
├── train_autoencoder.py        # Pre-training script
├── visualize_autoencoder.py    # Part 1: Visualization and exploration
├── finetune_comparison.py      # Part 2: Fine-tuning comparison
├── models/                     # Saved model directory
└── data/
    ├── raw/MNIST/              # Place Fashion MNIST CSV files here
    └── processed/FashionMNIST/ # Generated visualizations and plots
```

## Setup

### 1. Install Dependencies

```bash
pip install torch torchvision numpy pandas scikit-learn matplotlib umap-learn
```

### 2. Download Fashion MNIST Data

Download the Fashion MNIST dataset in CSV format and place it in:

- `data/raw/MNIST/fashion_mnist_train.csv`
- `data/raw/MNIST/fashion_mnist_test.csv`

(Or the code will fall back to regular MNIST for testing purposes)

## Pre-training (5 points)

### Basic Training

Train the autoencoder with different hyperparameters:

```bash
# Example 1: Basic configuration
python train_autoencoder.py --hidden_dims 256 64 256 --lr 0.001 --epochs 50 --batch_size 128 --optimizer adam

# Example 2: Smaller bottleneck
python train_autoencoder.py --hidden_dims 128 32 128 --lr 0.001 --epochs 50 --batch_size 64

# Example 3: Larger architecture
python train_autoencoder.py --hidden_dims 512 128 512 --lr 0.0005 --epochs 100 --batch_size 256
```

### Key Hyperparameters

- `--hidden_dims`: Three integers [encoder1, bottleneck, decoder1]
  - Example: `256 64 256` creates: 784 → 256 → 64 → 256 → 784
- `--lr`: Learning rate (try 0.0001, 0.001, 0.01)
- `--batch_size`: Batch size (try 64, 128, 256)
- `--epochs`: Number of training epochs (30-100)
- `--optimizer`: Choice of `sgd`, `sgd_momentum`, or `adam`
- `--init`: Weight initialization (`xavier`, `he`, `constant`)
- `--activation`: Hidden activation function (`relu`, `sigmoid`)

### Regularization Options

```bash
# With batch normalization
python train_autoencoder.py --hidden_dims 256 64 256 --bn

# With denoising (noise injection)
python train_autoencoder.py --hidden_dims 256 64 256 --denoise --noise_std 0.1

# With early stopping
python train_autoencoder.py --hidden_dims 256 64 256 --early_stop --patience 10

# Combined regularization
python train_autoencoder.py --hidden_dims 256 64 256 --bn --denoise --early_stop
```

### Saved Models

Models are automatically saved to `models/` directory:

- `autoencoder_xavier_relu_bnTrue_dims256_64_256.pth` (final model)
- `autoencoder_best.pth` (best validation model if using early stopping)

## Part 1: Initial Exploration

### Visualize Pre-trained Autoencoder

```bash
python visualize_autoencoder.py --model_path models/autoencoder_xavier_relu_bnFalse_dims256_64_256.pth
```

This script generates:

1. **Reconstructions** (`reconstructions.png`)

   - Shows original images vs. reconstructed images
   - Demonstrates how well the autoencoder learned to compress and decompress

2. **Hidden Features** (`hidden_features_layer1.png`)

   - Visualizes the learned filters/features in the first layer
   - Shows what patterns the encoder detects

3. **Bottleneck Representations** (`bottleneck_representations.png`)

   - Shows how images are represented in the compressed bottleneck space
   - Each image is reduced to a small vector

4. **UMAP Visualization** (`umap_visualization.png`)
   - 2D projection of the bottleneck representations
   - Colors indicate different classes
   - Shows cluster separation and structure

### Options

```bash
# Customize visualization
python visualize_autoencoder.py \
    --model_path models/autoencoder_best.pth \
    --num_samples 15 \
    --max_samples 5000 \
    --save_dir data/processed/FashionMNIST/exploration/
```

## Part 1: Regularized Model

### Train Regularized Models

```bash
# 1. Baseline (no regularization)
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 50

# 2. With Parameter Norm Penalties (implicit in Adam optimizer with weight decay)
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 50 --optimizer adam

# 3. With Noise Injection (Denoising Autoencoder)
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 50 --denoise --noise_std 0.15

# 4. With Early Stopping
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 100 --early_stop --patience 10

# 5. Combined: Batch Norm + Denoising + Early Stopping
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 100 --bn --denoise --noise_std 0.1 --early_stop
```

### Compare Regularized vs Non-Regularized

```bash
# Visualize non-regularized model
python visualize_autoencoder.py --model_path models/autoencoder_baseline.pth --save_dir data/processed/FashionMNIST/baseline/

# Visualize regularized model
python visualize_autoencoder.py --model_path models/autoencoder_regularized.pth --save_dir data/processed/FashionMNIST/regularized/
```

Compare the UMAP plots and reconstruction quality to see the effect of regularization.

## Part 2: Fine-tuning

### Run Comparison

```bash
# Train both raw and encoded classifiers
python finetune_comparison.py --autoencoder_path models/autoencoder_xavier_relu_bnFalse_dims256_64_256.pth --epochs 30 --lr 0.001
```

This will:

1. Train a classifier from scratch on raw pixel data (784 dimensions)
2. Extract encoded features from the autoencoder (e.g., 64 dimensions)
3. Train a classifier on the encoded features
4. Compare performance and generate plots

### Expected Results

The encoded classifier typically:

- Trains **faster** (fewer parameters to learn)
- May achieve **better generalization** (learned features are more robust)
- Uses **less memory** (smaller input dimension)

### Options

```bash
# Customize classifier architecture
python finetune_comparison.py \
    --autoencoder_path models/autoencoder_best.pth \
    --epochs 50 \
    --lr 0.001 \
    --hidden_dim 128 \
    --num_layers 2 \
    --batch_size 128 \
    --bn
```

### Without Autoencoder (Baseline Only)

```bash
# Train only the raw classifier for baseline comparison
python finetune_comparison.py --epochs 30
```

## Implementation Details

### Autoencoder Architecture

The autoencoder has 4 hidden layers total:

- **Encoder Layer 1**: input_dim (784) → hidden_dims[0] (e.g., 256)
- **Encoder Layer 2 (Bottleneck)**: hidden_dims[0] → hidden_dims[1] (e.g., 64)
- **Decoder Layer 1**: hidden_dims[1] → hidden_dims[2] (e.g., 256)
- **Decoder Layer 2 (Output)**: hidden_dims[2] → input_dim (784)

### Regularization Techniques Implemented

1. **Parameter Norm Penalties**: Implicit through optimizer (Adam with weight decay)
2. **Data Augmentation**: Can be extended in dataset.py
3. **Injecting Noise**: Denoising autoencoder (`--denoise` flag)
4. **Early Stopping**: Stop training when validation loss stops improving (`--early_stop`)
5. **Batch Normalization**: Normalize activations between layers (`--bn`)

### Loss Function

- **Reconstruction Loss**: Mean Squared Error (MSE)
  - Measures pixel-wise difference between input and reconstruction
  - Formula: `(1/m) * Σ(x - x̂)²`

### Activation Functions

- **Hidden Layers**: ReLU (default) or Sigmoid
- **Output Layer**: Sigmoid (to match [0, 1] pixel range)

## Results Interpretation

### Good Pre-training Indicators

1. **Learning Curves**

   - Training and validation loss should decrease
   - Should converge and not diverge

2. **Reconstructions**

   - Clear, recognizable images
   - Minimal blur or artifacts

3. **UMAP Plot**
   - Clear cluster separation for different classes
   - Similar items grouped together

### Regularization Effects

- **Without Regularization**: May overfit, sharp features, scattered UMAP clusters
- **With Regularization**: Better generalization, smoother features, tighter clusters

### Fine-tuning Comparison

- **Raw Classifier**: Baseline performance
- **Encoded Classifier**: Should show improvement in:
  - Convergence speed
  - Final accuracy
  - Generalization (gap between train/val)

## Troubleshooting

### Issue: Poor Reconstructions

**Solutions:**

- Increase bottleneck size
- Train for more epochs
- Adjust learning rate
- Try different activation functions

### Issue: UMAP Shows No Structure

**Solutions:**

- Train for more epochs
- Increase model capacity (larger hidden_dims)
- Add regularization
- Check if data is properly normalized

### Issue: Encoded Classifier Performs Worse

**Solutions:**

- Bottleneck might be too small (information loss)
- Autoencoder might not be well-trained
- Try different bottleneck dimensions
- Ensure autoencoder is trained on same dataset

## Deliverables Checklist

### Pre-training (5 points)

- [ ] Train autoencoder with multiple hyperparameter configurations
- [ ] Save final model to `models/` directory
- [ ] Document hyperparameters used in report

### Part 1: Initial Exploration

- [ ] Generate reconstruction visualizations
- [ ] Plot hidden layer features
- [ ] Create UMAP visualization
- [ ] Analyze what features the autoencoder learned

### Part 1: Regularized Model

- [ ] Implement at least 2 regularization techniques
- [ ] Train regularized model
- [ ] Generate UMAP plots for comparison
- [ ] Compare with non-regularized version

### Part 2: Fine-tuning

- [ ] Train classifier from scratch (raw data)
- [ ] Train classifier on encoded representations
- [ ] Compare performance metrics
- [ ] Generate comparison plots
- [ ] Analyze which approach works better and why

## Notes

- All visualization outputs are saved to `data/processed/FashionMNIST/`
- Models are saved to `models/` directory
- Learning curves are automatically generated during training
- The code reuses the feed-forward network from Assignment 1

## Example Complete Workflow

```bash
# 1. Pre-train autoencoder
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 50 --lr 0.001 --optimizer adam --bn --early_stop

# 2. Visualize (Part 1 - Initial)
python visualize_autoencoder.py --model_path models/autoencoder_best.pth

# 3. Train regularized version (Part 1 - Regularized)
python train_autoencoder.py --hidden_dims 256 64 256 --epochs 50 --lr 0.001 --optimizer adam --bn --denoise --early_stop

# 4. Visualize regularized model
python visualize_autoencoder.py --model_path models/autoencoder_xavier_relu_bnTrue_dims256_64_256.pth --save_dir data/processed/FashionMNIST/regularized/

# 5. Fine-tune and compare (Part 2)
python finetune_comparison.py --autoencoder_path models/autoencoder_best.pth --epochs 30 --lr 0.001 --bn
```
