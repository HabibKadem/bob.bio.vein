# Implementation Summary: CNN+ViT Model for Dorsal Hand Vein Recognition

## Overview

A complete CNN+Vision Transformer (ViT) hybrid model has been implemented for person recognition using dorsal hand vein biometric patterns. The implementation follows the existing bob.bio.vein architecture patterns and integrates seamlessly with the framework.

## What Was Implemented

### 1. Database Interface (`bob/bio/vein/database/dorsalhandveins.py`)

**Purpose**: Manages the DorsalHandVeins dataset with 138 people, 4 images per person.

**Features**:
- Automatic CSV protocol generation
- Two protocols: `train-test` and `cross-validation`
- 70/15/15 train/dev/eval split
- Compatible with bob.bio.base infrastructure
- Configuration via: `bob config set bob.bio.vein.dorsalhandveins.directory [PATH]`

**Expected Dataset Structure**:
```
DorsalHandVeins_DB1_png/
    train/
        person_001_db1_L1.png
        person_001_db1_L2.png
        person_001_db1_L3.png
        person_001_db1_L4.png
        ...
        person_138_db1_L4.png
```

### 2. CNN+ViT Model (`bob/bio/vein/extractor/CNNViT.py`)

**Purpose**: Deep learning model combining CNN and Vision Transformer architectures.

**Architecture Components**:

#### a. CNN Backbone
- 3 convolutional blocks (64, 128, 256 filters)
- Batch normalization and ReLU activation
- Max pooling for spatial downsampling
- Extracts local vein patterns

#### b. Vision Transformer
- Patch embedding (default: 16x16 patches)
- Positional encoding
- Multi-head self-attention (8 heads)
- 6 transformer encoder layers
- Captures global context and relationships

#### c. Classification Head
- Layer normalization
- Fully connected layers with dropout
- Outputs: person identification

**Model Classes**:
- `VeinCNNViTModel`: High-level API for training and inference
- `CNNViTNetwork`: PyTorch neural network architecture
- `VeinDataset`: Dataset class for vein images
- `get_transforms()`: Data augmentation and preprocessing

**Key Methods**:
- `train()`: Train the model with validation
- `evaluate()`: Evaluate on test data
- `extract_features()`: Extract embeddings for matching
- `predict()`: Predict person ID from image
- `save_model()` / `load_model()`: Persistence

### 3. Training Script (`bob/bio/vein/script/train_cnn_vit.py`)

**Purpose**: Command-line tool for training the CNN+ViT model.

**Usage**:
```bash
bob_bio_vein_train_cnn_vit.py \
    --data-dir /path/to/DorsalHandVeins_DB1_png \
    --output-dir models \
    --img-size 224 \
    --batch-size 16 \
    --epochs 50 \
    --learning-rate 0.0001 \
    --num-workers 4
```

**Features**:
- Automatic data loading and splitting
- Data augmentation (rotation, translation)
- Training with validation
- Model checkpointing (saves best model)
- Test set evaluation
- Label mapping persistence

### 4. Configuration Files

#### a. `bob/bio/vein/config/database/dorsalhandveins.py`
- Database configuration
- Registers database with bob.bio framework

#### b. `bob/bio/vein/config/cnn_vit.py`
- Model configuration
- Pipeline setup
- Default hyperparameters

### 5. Documentation

#### a. `doc/cnn_vit_guide.md`
- Comprehensive user guide
- Installation instructions
- Usage examples
- API reference
- Troubleshooting tips

### 6. Tests (`bob/bio/vein/tests/test_cnn_vit.py`)

**Test Coverage**:
- Model import and instantiation
- Forward pass validation
- Feature extraction
- Dataset class functionality
- Database protocols
- Configuration loading
- Mock data testing

### 7. Examples (`bob/bio/vein/script/cnn_vit_examples.py`)

**Purpose**: Interactive examples demonstrating all features.

**Examples Included**:
1. Basic model usage
2. Database setup
3. Training commands
4. Inference code
5. Dataset structure
6. Custom training scripts

### 8. Integration with bob.bio.vein

**setup.py Updates**:
- Added entry points for new database
- Added entry points for CNN+ViT pipeline
- Registered console scripts:
  - `bob_bio_vein_train_cnn_vit.py`
  - `bob_bio_vein_cnn_vit_examples.py`

**__init__.py Updates**:
- Added CNN+ViT imports to extractor module
- Graceful fallback if PyTorch not installed

## Dependencies

### Required
- bob.bio.base
- bob.io.base
- bob.extension
- numpy
- scipy
- scikit-learn

### Optional (for CNN+ViT model)
- PyTorch >= 1.9
- torchvision >= 0.10

**Installation**:
```bash
pip install torch torchvision
```

## Quick Start Guide

### Step 1: Install Dependencies
```bash
pip install torch torchvision
```

### Step 2: Configure Dataset Path
```bash
bob config set bob.bio.vein.dorsalhandveins.directory /path/to/DorsalHandVeins_DB1_png
```

### Step 3: Train the Model
```bash
bob_bio_vein_train_cnn_vit.py \
    --data-dir /path/to/DorsalHandVeins_DB1_png \
    --output-dir models \
    --epochs 50
```

### Step 4: Use Trained Model
```python
from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel
import bob.io.base

# Load model
model = VeinCNNViTModel(num_classes=138)
model.load_model('models/cnn_vit_dorsalhandveins.pth')

# Predict
image = bob.io.base.load('path/to/image.png')
class_id, confidence = model.predict(image)
print(f"Person: {class_id}, Confidence: {confidence:.4f}")
```

## Model Hyperparameters

Default values (can be customized):
- **num_classes**: 138 (number of people)
- **img_size**: 224 (input image size)
- **patch_size**: 16 (ViT patch size)
- **embed_dim**: 256 (embedding dimension)
- **num_heads**: 8 (attention heads)
- **num_layers**: 6 (transformer layers)
- **dropout**: 0.1 (dropout rate)
- **batch_size**: 16
- **learning_rate**: 1e-4
- **epochs**: 50

## Training Features

1. **Data Augmentation**:
   - Random rotation (±10 degrees)
   - Random translation (±10%)
   - Normalization

2. **Optimization**:
   - AdamW optimizer
   - Cosine annealing learning rate scheduler
   - Weight decay regularization

3. **Model Checkpointing**:
   - Saves best model based on validation accuracy
   - Automatic model persistence

4. **Multi-GPU Support**:
   - Automatically uses GPU if available
   - Falls back to CPU if needed

## File Structure

```
bob/bio/vein/
├── database/
│   └── dorsalhandveins.py          # Database interface
├── extractor/
│   ├── CNNViT.py                   # CNN+ViT model
│   └── __init__.py                 # Updated imports
├── config/
│   ├── cnn_vit.py                  # Model config
│   └── database/
│       └── dorsalhandveins.py      # Database config
├── script/
│   ├── train_cnn_vit.py           # Training script
│   └── cnn_vit_examples.py        # Examples
├── tests/
│   └── test_cnn_vit.py            # Unit tests
└── doc/
    └── cnn_vit_guide.md           # User guide
```

## Key Design Decisions

1. **Hybrid Architecture**: Combines CNN (local features) with ViT (global context) for optimal vein pattern recognition

2. **Grayscale Processing**: Single-channel input for vein images

3. **Modular Design**: Separates model, training, and database components

4. **Bob.bio Integration**: Follows existing patterns in bob.bio.vein

5. **Optional Dependencies**: PyTorch is optional; framework works without it

6. **Flexible Training**: Support for custom training loops and data loaders

## Testing and Validation

All files have been validated for:
- ✓ Python syntax correctness
- ✓ Module imports
- ✓ Code compilation
- ✓ Basic functionality (where dependencies available)

## Known Limitations

1. **PyTorch Required**: The CNN+ViT model requires PyTorch to be installed
2. **GPU Recommended**: Training on CPU will be slow
3. **Memory Requirements**: Model requires ~2GB GPU memory with default settings
4. **Dataset Format**: Expects specific naming convention and structure

## Future Enhancements (Suggestions)

1. Add pre-trained weights for transfer learning
2. Support for different image resolutions
3. Model quantization for mobile deployment
4. Additional data augmentation techniques
5. Ensemble methods with multiple models
6. Real-time inference optimization
7. Web interface for model training and testing

## Troubleshooting

### "PyTorch not available"
```bash
pip install torch torchvision
```

### "Database directory not found"
```bash
bob config set bob.bio.vein.dorsalhandveins.directory /correct/path
```

### "Out of memory"
Reduce batch size or image size:
```bash
--batch-size 8 --img-size 128
```

### "Import Error"
Install bob.bio.vein from repository:
```bash
pip install -e .
```

## References

- Bob Toolkit: https://www.idiap.ch/software/bob
- Vision Transformer Paper: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- CNN Architectures: ResNet, VGG patterns

## Contact and Support

For questions or issues:
- Repository: https://gitlab.idiap.ch/bob/bob.bio.vein
- Documentation: doc/cnn_vit_guide.md
- Examples: bob/bio/vein/script/cnn_vit_examples.py

## License

This implementation follows the bob.bio.vein license (GPLv3).
