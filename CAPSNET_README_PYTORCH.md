# Capsule Neural Network for Dorsal Hand Veins Classification (PyTorch)

This directory contains a complete PyTorch implementation of a Capsule Neural Network (CapsNet) for classifying dorsal hand vein images.

## Overview

This is the **PyTorch version** of the CapsNet implementation. If you prefer TensorFlow/Keras, see `capsnet_dorsal_vein_classification.py`.

### Why PyTorch?

- **More control**: Explicit tensor operations and training loops
- **Research-friendly**: Easy to modify and experiment with
- **Popular**: Large community and ecosystem
- **Dynamic computation**: More flexible for custom architectures

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
pip install torch torchvision
pip install numpy scikit-learn matplotlib pillow
```

Or use the provided requirements:
```bash
pip install -r capsnet_requirements_pytorch.txt
```

### GPU Support

For CUDA support, install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for more options.

## Usage

### Basic Training

Simply run the script with default parameters:

```bash
python capsnet_dorsal_vein_classification_pytorch.py
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
   - `best_capsnet_pytorch.pth` - Best model based on validation loss
   - `capsnet_pytorch_final.pth` - Final model after all epochs
5. Generate training metrics plot: `capsnet_training_metrics_pytorch.png`

### Loading a Trained Model

```python
import torch
from capsnet_dorsal_vein_classification_pytorch import CapsNet

# Create model
model = CapsNet(num_classes=276)

# Load weights
model.load_state_dict(torch.load('best_capsnet_pytorch.pth'))
model.eval()

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Make predictions
with torch.no_grad():
    outputs = model(img_tensor)
    predicted_person = outputs.argmax(dim=1).item() + 1
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
```python
def squash(vectors, dim=-1):
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    return scale * vectors
```

#### Dynamic Routing
Iterative process to route information between capsule layers based on agreement:
- Lower capsules vote for higher capsules
- Routing coefficients determined by agreement
- More similar predictions → stronger routing

#### Margin Loss
Custom loss function for CapsNet:
```python
L_k = T_k * max(0, 0.9 - ||v_k||)² + 0.5 * (1 - T_k) * max(0, ||v_k|| - 0.1)²
```
- Encourages correct class capsule length > 0.9
- Encourages wrong class capsule length < 0.1

## PyTorch vs TensorFlow/Keras

| Aspect | PyTorch | TensorFlow/Keras |
|--------|---------|------------------|
| Style | Imperative | Declarative |
| Training Loop | Manual (more control) | Built-in (simpler) |
| Debugging | Easier (Python debugger) | Harder (graph execution) |
| Research | Preferred | Good |
| Production | Good | Preferred |
| Community | Growing | Larger |

## Performance Tips

### GPU Acceleration
The script automatically detects and uses GPU if available:
```
Using device: cuda
GPU: NVIDIA GeForce RTX 3090
```

### Memory Optimization
If you run out of GPU memory:
- Reduce `BATCH_SIZE` (e.g., 8 or 4)
- Reduce `IMG_SIZE` (e.g., 128 or 160)
- Use `num_workers=0` in DataLoader

### Training Time
CapsNet is more computationally intensive than CNN:
- Expect longer training time per epoch
- Use GPU for reasonable training times
- Consider reducing `NUM_ROUTING` to 2 for faster training

## Monitoring Training

### Console Output
The training progress is displayed in the console:
```
Epoch [1/200] Train Loss: 0.5234 Acc: 0.7821 | Val Loss: 0.4123 Acc: 0.8234
  → Best model saved (val_loss: 0.4123)
```

### Training Metrics Plot
After training, check `capsnet_training_metrics_pytorch.png` for:
- Training and validation loss curves
- Training and validation accuracy curves

### TensorBoard (Optional)
Add TensorBoard logging to monitor training:

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/capsnet')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
writer.add_scalar('Accuracy/train', train_acc, epoch)
writer.add_scalar('Accuracy/val', val_acc, epoch)
```

Then run:
```bash
tensorboard --logdir=runs
```

## Troubleshooting

### "No image files found"
- Check `DATASET_DIR` path is correct
- Ensure images have supported extensions (.png, .jpg, etc.)

### "Cannot extract label from filename"
- Verify filename format: `person_XXX_db1_LY.png`
- Check underscores in filenames

### Out of Memory Error (CUDA)
- Reduce `BATCH_SIZE`
- Reduce `IMG_SIZE`
- Set `num_workers=0` in DataLoader
- Close other GPU-intensive applications

### Slow Training
- Ensure GPU is being used (`Using device: cuda`)
- Increase `num_workers` in DataLoader (default: 2)
- Consider using mixed precision training (AMP)

### Poor Accuracy
- Ensure dataset is balanced (same images per person)
- Try different learning rates
- Increase `NUM_ROUTING` iterations
- Enable data augmentation
- Check for data quality issues

## Advanced Usage

### Custom Dataset
```python
from torch.utils.data import Dataset

class CustomVeinDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform
    
    def __getitem__(self, idx):
        # Your custom loading logic
        image = load_image(self.file_paths[idx])
        label = extract_label(self.file_paths[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
```

### Transfer Learning
```python
# Load pretrained model
model = CapsNet(num_classes=276)
model.load_state_dict(torch.load('best_capsnet_pytorch.pth'))

# Freeze early layers
for param in model.conv1.parameters():
    param.requires_grad = False

# Fine-tune on new dataset
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001
)
```

### Model Export
```python
# Export to ONNX
dummy_input = torch.randn(1, 1, 224, 224).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "capsnet.onnx",
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)
```

## References

1. Sabour, S., Frosst, N., & Hinton, G. E. (2017). "Dynamic Routing Between Capsules". NeurIPS.
2. Hinton, G. E., Sabour, S., & Frosst, N. (2018). "Matrix Capsules with EM Routing". ICLR.
3. PyTorch Documentation: https://pytorch.org/docs/

## License

This code follows the same license as the bob.bio.vein repository.

## Contact

For issues or questions, please open an issue in the repository.
