# CNN+ViT Model for Dorsal Hand Vein Recognition

This module provides a hybrid CNN+Vision Transformer model for person recognition using dorsal hand vein biometric patterns.

## Dataset Structure

The DorsalHandVeins database consists of grayscale images from 138 subjects, with 4 images per subject:

```
DorsalHandVeins_DB1_png/
    train/
        person_001_db1_L1.png
        person_001_db1_L2.png
        person_001_db1_L3.png
        person_001_db1_L4.png
        ...
        person_138_db1_L1.png
        person_138_db1_L2.png
        person_138_db1_L3.png
        person_138_db1_L4.png
```

## Installation

### Prerequisites

1. Install the base bob.bio.vein package
2. Install PyTorch (required for CNN+ViT model):

```bash
pip install torch torchvision
```

### Configuration

Set the path to your dataset:

```bash
bob config set bob.bio.vein.dorsalhandveins.directory /path/to/DorsalHandVeins_DB1_png
```

## Usage

### Training the Model

You can train the CNN+ViT model using the provided training script:

```bash
bob_bio_vein_train_cnn_vit.py \
    --data-dir /path/to/DorsalHandVeins_DB1_png \
    --output-dir models \
    --img-size 224 \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 0.0001
```

#### Training Parameters

- `--data-dir`: Path to the dataset directory (required)
- `--output-dir`: Directory to save trained models (default: models)
- `--img-size`: Input image size (default: 224)
- `--batch-size`: Batch size for training (default: 16)
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Learning rate (default: 0.0001)
- `--num-workers`: Number of data loader workers (default: 4)

### Using the Model Programmatically

```python
from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase

# Initialize database
database = DorsalHandVeinsDatabase(protocol='train-test')

# Initialize model
model = VeinCNNViTModel(
    num_classes=138,
    img_size=224,
    patch_size=16,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    dropout=0.1,
)

# Load trained weights
model.load_model('models/cnn_vit_dorsalhandveins.pth')

# Extract features from an image
import bob.io.base
image = bob.io.base.load('path/to/image.png')
features = model.extract_features(image)

# Predict class
class_id, confidence = model.predict(image)
print(f"Predicted class: {class_id}, Confidence: {confidence:.4f}")
```

## Model Architecture

The CNN+ViT hybrid model combines:

1. **CNN Backbone**: 
   - 3 convolutional blocks with batch normalization and max pooling
   - Extracts local features from vein patterns

2. **Vision Transformer**:
   - Patch embedding with positional encoding
   - Multi-head self-attention mechanism
   - Captures global context and spatial relationships

3. **Classification Head**:
   - Fully connected layers
   - Outputs person identification

### Model Parameters

- **num_classes**: Number of people/classes (default: 138)
- **img_size**: Input image size (default: 224)
- **patch_size**: Patch size for ViT (default: 16)
- **embed_dim**: Embedding dimension (default: 256)
- **num_heads**: Number of attention heads (default: 8)
- **num_layers**: Number of transformer layers (default: 6)
- **dropout**: Dropout rate (default: 0.1)

## Database Protocols

The DorsalHandVeins database supports two protocols:

### 1. train-test
- Uses first 3 images per subject for enrollment
- Last image for probing
- Split: 70% train, 15% dev, 15% eval

### 2. cross-validation
- All images can be used for both enrollment and probing
- Suitable for cross-validation experiments

## Features

- **Data Augmentation**: Random rotation and translation during training
- **Adaptive Learning Rate**: Cosine annealing scheduler
- **Feature Extraction**: Extract feature embeddings for similarity matching
- **Model Checkpointing**: Saves best model based on validation accuracy

## Example Results

After training, the model will:
- Save trained weights to `models/cnn_vit_dorsalhandveins.pth`
- Save label mapping to `models/label_map.npy`
- Report training and validation accuracy per epoch
- Evaluate on test set and report final accuracy

## Notes

- Images are automatically resized to the specified input size
- Grayscale images are expected (single channel)
- The model uses GPU acceleration if available
- Training requires PyTorch and torchvision

## Troubleshooting

### Import Error: PyTorch not available
Install PyTorch: `pip install torch torchvision`

### Dataset not found
Set the dataset path: `bob config set bob.bio.vein.dorsalhandveins.directory /path/to/dataset`

### Out of memory during training
- Reduce batch size: `--batch-size 8`
- Reduce image size: `--img-size 128`
- Reduce model size: modify embed_dim or num_layers in the code

## Citation

If you use this code, please cite:

```
@software{bob_bio_vein_cnn_vit,
  title={CNN+ViT Model for Dorsal Hand Vein Recognition},
  author={bob.bio.vein contributors},
  year={2025},
  url={https://gitlab.idiap.ch/bob/bob.bio.vein}
}
```
