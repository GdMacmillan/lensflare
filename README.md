![LensFlare Banner](https://raw.githubusercontent.com/GdMacmillan/lensflare/master/readme_files/lensflare_banner.png)

# LensFlare

LensFlare is an educational deep learning library for understanding neural networks. The code is based on work from the [Coursera deeplearning.ai course](https://www.coursera.org/specializations/deep-learning).

## Features

- **TensorFlow 2 with GradientTape** - Low-level API for explicit gradient computation
- **Metal GPU acceleration** on Apple Silicon Macs
- **sklearn-style interface** with `fit()`, `predict()`, `transform()`
- **Pure NumPy implementation** for maximum educational transparency

## Installation

```bash
pip install -e .

# For Apple Silicon GPU acceleration
pip install -e ".[metal]"
```

## Quick Start

```python
from lensflare import TfNNClassifier, load_moons_dataset, check_gpu_available, plot_decision_boundary

# Check for Metal GPU (Apple Silicon)
check_gpu_available()

# Load the moons dataset
X_train, y_train = load_moons_dataset(n_samples=300, noise=0.2, seed=42, plot=True)
```

![png](https://raw.githubusercontent.com/GdMacmillan/lensflare/master/readme_files/plot_data.png)

```python
# Define network architecture: 2 inputs -> 64 -> 32 -> 16 -> 1 output
layers_dims = [X_train.shape[0], 64, 32, 16, 1]

# Create classifier with Adam optimizer
clf = TfNNClassifier(
    layers_dims=layers_dims,
    optimizer="adam",
    alpha=0.01,
    lambd=0.01,
    keep_prob=0.9,
    num_epochs=2000,
    print_cost=True
)

# Train the model
clf.fit(X_train, y_train, seed=1)

# Get predictions
y_pred_train = clf.transform(X_train, y_train)
```

    Cost after epoch 0: 0.693147
    Cost after epoch 100: 0.341821
    ...
    Training Accuracy: 98.00%

```python
# Plot decision boundary
plot_decision_boundary(clf, X_train, y_train)
```

![png](https://raw.githubusercontent.com/GdMacmillan/lensflare/master/readme_files/decision_boundary.png)

## NumPy Implementation

For even more educational transparency, use the pure NumPy classifier:

```python
from lensflare import NpNNClassifier

np_clf = NpNNClassifier(
    layers_dims=[X_train.shape[0], 32, 16, 1],
    optimizer="adam",
    alpha=0.01,
    lambd=0.01,
    num_epochs=2000,
    print_cost=True
)

np_clf.fit(X_train, y_train, seed=1)
```

## Requirements

- Python >= 3.12
- TensorFlow >= 2.15, <= 2.18.1
- NumPy, Matplotlib, scikit-learn

## License

MIT
