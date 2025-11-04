# Capsule Neural Network for Dorsal Hand Veins Classification

This directory contains a complete implementation of a Capsule Neural Network (CapsNet) for classifying dorsal hand vein images.

## Overview

Capsule Networks are an advanced deep learning architecture that addresses some limitations of traditional CNNs:
- **Better spatial relationships**: Capsules preserve hierarchical pose relationships between features
- **Equivariance**: Naturally handles rotations and translations
- **Less data needed**: More efficient learning with limited training samples
- **Viewpoint invariance**: Better generalization across different viewpoints

## Dataset Structure

The code expects your dataset in the following structure:

```
DorsalHandVeins_DB1_png/
  train/
    person_001_db1_L1.png
    person_001_db1_L2.png
    person_001_db1_L3.png
    person_001_db1_L4.png
    ...
    person_276_db1_L1.png
    person_276_db1_L2.png
    person_276_db1_L3.png
    person_276_db1_L4.png
```

Where:
- Images are **grayscale**
- Each person has **4 images** (L1, L2, L3, L4)
- Person IDs range from **001 to 276**

## Requirements

```bash
pip install tensorflow>=2.8.0
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install pillow
```

Or use the provided requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Simply run the script with default parameters:

```bash
python capsnet_dorsal_vein_classification.py
```

### Customizing Parameters

Edit the configuration section at the top of the file:

```python
# Image and training parameters
IMG_SIZE = 224              # Input image size
BATCH_SIZE = 16             # Batch size for training
NUM_CLASSES = 276           # Number of persons
EPOCHS = 200                # Maximum epochs
LEARNING_RATE = 0.001       # Initial learning rate

# CapsNet specific parameters
PRIMARY_CAPS_DIM = 8        # Dimension of primary capsules
DIGIT_CAPS_DIM = 16         # Dimension of digit capsules
NUM_ROUTING = 3             # Routing iterations

# Dataset path
DATASET_DIR = "DorsalHandVeins_DB1_png/train/"
```

### Training Output

The script will:
1. Load and split your dataset (70% train, 30% validation)
2. Build the CapsNet architecture
3. Train the model with progress updates
4. Save two models:
   - `best_capsnet_model.h5` - Best model based on validation loss
   - `capsnet_final_model.h5` - Final model after all epochs
5. Generate training metrics plot: `capsnet_training_metrics.png`

### Loading a Trained Model

```python
import tensorflow as tf
from tensorflow import keras

# Load the best model
model = keras.models.load_model('best_capsnet_model.h5', 
                                 custom_objects={'margin_loss': margin_loss})

# Make predictions
import numpy as np
from PIL import Image

# Load and preprocess an image
img = Image.open('path/to/test/image.png').convert('L')
img = img.resize((224, 224))
img_array = np.array(img, dtype=np.float32) / 255.0
img_array = np.expand_dims(img_array, axis=[0, -1])  # Add batch and channel dims

# Predict
predictions = model.predict(img_array)
predicted_person = np.argmax(predictions) + 1  # Add 1 to get person ID
print(f"Predicted person: {predicted_person:03d}")
```

## Architecture Details

### Layer Structure

1. **Initial Convolution** (Conv2D)
   - 256 filters, 9x9 kernel
   - Extracts low-level features (edges, textures)

2. **Primary Capsules** (PrimaryCapsule)
   - 32 capsule types
   - 8D vectors per capsule
   - Converts features to capsule representation

3. **Digit Capsules** (DigitCapsule)
   - 276 capsules (one per person)
   - 16D vectors per capsule
   - Dynamic routing with 3 iterations

4. **Output** (Capsule Length)
   - Length of capsule = probability of class presence

### Key Components

#### Squash Activation
Non-linear activation for capsules that scales vectors to length [0, 1]:
```
squash(s) = ||s||² / (1 + ||s||²) * s / ||s||
```

#### Dynamic Routing
Iterative process to route information between capsule layers based on agreement:
- Lower capsules vote for higher capsules
- Routing coefficients determined by agreement
- More similar predictions → stronger routing

#### Margin Loss
Custom loss function for CapsNet:
```
L_k = T_k * max(0, 0.9 - ||v_k||)² + 0.5 * (1 - T_k) * max(0, ||v_k|| - 0.1)²
```
- Encourages correct class capsule length > 0.9
- Encourages wrong class capsule length < 0.1

## Monitoring Training

### TensorBoard

The script logs training metrics to TensorBoard:

```bash
tensorboard --logdir=./logs_capsnet
```

Then open http://localhost:6006 in your browser.

### Training Metrics Plot

After training, check `capsnet_training_metrics.png` for:
- Training and validation loss curves
- Training and validation accuracy curves

## Performance Tips

### GPU Acceleration
The script automatically detects and uses GPU if available:
```
Using device: GPU
Number of GPUs available: 1
```

### Memory Optimization
If you run out of GPU memory:
- Reduce `BATCH_SIZE` (e.g., 8 or 4)
- Reduce `IMG_SIZE` (e.g., 128 or 160)

### Training Time
CapsNet is more computationally intensive than CNN:
- Expect longer training time per epoch
- Use GPU for reasonable training times
- Consider reducing `NUM_ROUTING` to 2 for faster training

## Comparing with CNN

| Aspect | CNN | CapsNet |
|--------|-----|---------|
| Training speed | Faster | Slower |
| Data requirements | More data needed | Works with less data |
| Spatial relationships | Lost in pooling | Preserved in capsules |
| Rotation invariance | Needs augmentation | Natural equivariance |
| Model complexity | Simpler | More complex |
| Interpretability | Black box | Capsule activations interpretable |

## Troubleshooting

### "No image files found"
- Check `DATASET_DIR` path is correct
- Ensure images have supported extensions (.png, .jpg, etc.)

### "Cannot extract label from filename"
- Verify filename format: `person_XXX_db1_LY.png`
- Check underscores in filenames

### Out of Memory Error
- Reduce `BATCH_SIZE`
- Reduce `IMG_SIZE`
- Close other GPU-intensive applications

### Poor Accuracy
- Ensure dataset is balanced (same images per person)
- Try different learning rates
- Increase `NUM_ROUTING` iterations
- Enable data augmentation
- Check for data quality issues

## References

1. Sabour, S., Frosst, N., & Hinton, G. E. (2017). "Dynamic Routing Between Capsules". NeurIPS.
2. Hinton, G. E., Sabour, S., & Frosst, N. (2018). "Matrix Capsules with EM Routing". ICLR.

## License

This code follows the same license as the bob.bio.vein repository.

## Contact

For issues or questions, please open an issue in the repository.
