# CNN+ViT Hybrid Model Architecture - Text Description

## Model Overview
Hybrid CNN+Vision Transformer model for dorsal hand vein biometric recognition
- **Purpose**: Person identification using vein patterns
- **Dataset**: 138 subjects with 4 grayscale images each
- **Input**: 224×224 grayscale images
- **Output**: 138 person classes

---

## Architecture Components

### 1. Input Layer
```
Input Image: 224×224×1 (grayscale)
```

### 2. CNN Backbone (Local Feature Extraction)

#### Conv Block 1
```
- Conv2D: 64 filters, kernel 3×3, padding 1
- BatchNorm2D: 64 channels
- ReLU activation
- Conv2D: 64 filters, kernel 3×3, padding 1
- BatchNorm2D: 64 channels
- ReLU activation
- MaxPool2D: 2×2, stride 2
Output: 112×112×64
```

#### Conv Block 2
```
- Conv2D: 128 filters, kernel 3×3, padding 1
- BatchNorm2D: 128 channels
- ReLU activation
- Conv2D: 128 filters, kernel 3×3, padding 1
- BatchNorm2D: 128 channels
- ReLU activation
- MaxPool2D: 2×2, stride 2
Output: 56×56×128
```

#### Conv Block 3
```
- Conv2D: 256 filters, kernel 3×3, padding 1
- BatchNorm2D: 256 channels
- ReLU activation
- Conv2D: 256 filters, kernel 3×3, padding 1
- BatchNorm2D: 256 channels
- ReLU activation
Output: 56×56×256
```

### 3. Vision Transformer (Global Context & Attention)

#### Patch Embedding
```
- Conv2D: 256→256 channels, kernel 16×16, stride 16
  (Converts feature maps to patches)
- Flatten patches: (56/16)² = 3.5² ≈ 12.25 patches
- Output: N_patches × 256-dim embeddings
```

#### Positional Encoding
```
- Learnable positional embeddings: (N_patches + 1) × 256
- Learnable CLS token: 1 × 256
- Add positional embeddings to patch embeddings
Output: (N_patches + 1) × 256
```

#### Transformer Encoder
```
For each of 6 layers:
  - Multi-Head Self-Attention:
    * Number of heads: 8
    * Embedding dimension: 256
    * Head dimension: 256/8 = 32
    * Attention: Q, K, V matrices
    * Scaled dot-product attention
  
  - Feed-Forward Network:
    * Linear: 256 → 1024 (4× expansion)
    * GELU activation
    * Dropout: 0.1
    * Linear: 1024 → 256
  
  - Residual connections and Layer Normalization
  
Output: (N_patches + 1) × 256
```

### 4. Classification Head

#### Layer Normalization
```
- LayerNorm: 256 dimensions
- Applied to CLS token output
Output: 256-dim vector
```

#### Fully Connected Layers
```
- Linear: 256 → 128
- ReLU activation
- Dropout: 0.1
- Linear: 128 → 138 (num_classes)
Output: 138 class logits
```

---

## Model Parameters

### Architecture Hyperparameters
```
num_classes:     138
img_size:        224
patch_size:      16
embed_dim:       256
num_heads:       8
num_layers:      6
dropout:         0.1
in_channels:     1 (grayscale)
```

### CNN Parameters
```
Conv Block 1: ~37K parameters
  - Conv1: 1×64×3×3 = 576
  - Conv2: 64×64×3×3 = 36,864
  
Conv Block 2: ~148K parameters
  - Conv1: 64×128×3×3 = 73,728
  - Conv2: 128×128×3×3 = 147,456
  
Conv Block 3: ~590K parameters
  - Conv1: 128×256×3×3 = 294,912
  - Conv2: 256×256×3×3 = 589,824
```

### ViT Parameters
```
Patch Embedding: ~1M parameters
Positional Embeddings: ~3K parameters
Transformer Encoder: ~4.7M parameters per layer × 6 layers
Classification Head: ~66K parameters
```

### Total Parameters
```
Approximate total: ~30-35M parameters
```

---

## Training Configuration

### Optimizer
```
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
Beta1: 0.9
Beta2: 0.999
```

### Learning Rate Schedule
```
Scheduler: CosineAnnealingLR
T_max: num_epochs
Eta_min: 0
```

### Loss Function
```
CrossEntropyLoss
```

### Training Parameters
```
Batch size: 16
Epochs: 50
Image size: 224×224
Number of workers: 4
```

---

## Data Augmentation

### Training Augmentation
```
1. Resize to 224×224
2. Random rotation: ±10 degrees
3. Random affine translation: ±10%
4. Normalization: mean=0.5, std=0.5
```

### Validation/Test Augmentation
```
1. Resize to 224×224
2. Normalization: mean=0.5, std=0.5
```

---

## Model Flow (Forward Pass)

### Step-by-Step Execution
```
1. Input Image (1×224×224)
   ↓
2. CNN Block 1 (64×112×112)
   ↓
3. CNN Block 2 (128×56×56)
   ↓
4. CNN Block 3 (256×56×56)
   ↓
5. Patch Embedding (N_patches×256)
   ↓
6. Add CLS Token (N_patches+1×256)
   ↓
7. Add Positional Encoding (N_patches+1×256)
   ↓
8. Transformer Encoder Layer 1 (N_patches+1×256)
   ↓
9. Transformer Encoder Layer 2 (N_patches+1×256)
   ↓
10. Transformer Encoder Layer 3 (N_patches+1×256)
    ↓
11. Transformer Encoder Layer 4 (N_patches+1×256)
    ↓
12. Transformer Encoder Layer 5 (N_patches+1×256)
    ↓
13. Transformer Encoder Layer 6 (N_patches+1×256)
    ↓
14. Extract CLS Token (256)
    ↓
15. Layer Normalization (256)
    ↓
16. FC Layer 1 + ReLU + Dropout (128)
    ↓
17. FC Layer 2 (138)
    ↓
18. Output Logits (138 classes)
```

---

## Key Features

### CNN Advantages
- Local feature extraction
- Translation invariance
- Parameter efficiency
- Hierarchical features

### Vision Transformer Advantages
- Global context awareness
- Long-range dependencies
- Self-attention mechanism
- Position-aware processing

### Hybrid Benefits
- Best of both worlds
- Local + Global features
- Robust representations
- Superior performance

---

## Implementation Details

### Weight Initialization
```
- Positional embeddings: TruncatedNormal (std=0.02)
- CLS token: TruncatedNormal (std=0.02)
- Conv layers: Kaiming Normal (mode='fan_out')
- BatchNorm: weight=1, bias=0
- Linear layers: TruncatedNormal (std=0.02)
```

### Dropout Locations
```
- Transformer FFN: 0.1
- Classification head: 0.1
```

### Activation Functions
```
- CNN: ReLU
- Transformer FFN: GELU
- Classification: ReLU
```

---

## Usage Example

### Model Instantiation
```python
from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel

model = VeinCNNViTModel(
    num_classes=138,
    img_size=224,
    patch_size=16,
    embed_dim=256,
    num_heads=8,
    num_layers=6,
    dropout=0.1
)
```

### Training
```python
model.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,
    learning_rate=1e-4,
    weight_decay=1e-4,
    save_path='model.pth'
)
```

### Inference
```python
# Load model
model.load_model('model.pth')

# Extract features
features = model.extract_features(image)  # 256-dim

# Predict class
class_id, confidence = model.predict(image)
```

---

## Performance Considerations

### Memory Requirements
```
- Model parameters: ~140 MB (FP32)
- Batch of 16: ~4-6 GB GPU memory
- Recommended GPU: 8GB+ VRAM
```

### Inference Speed
```
- Single image (GPU): ~10-20 ms
- Batch of 16 (GPU): ~50-100 ms
- CPU inference: ~200-500 ms per image
```

### Training Time
```
- 50 epochs on 138×4=552 images
- GPU (RTX 3090): ~30-60 minutes
- CPU: Several hours
```

---

## File Locations

### Model Implementation
```
bob/bio/vein/extractor/CNNViT.py
- VeinCNNViTModel: High-level API
- CNNViTNetwork: PyTorch model
- VeinDataset: Dataset class
- get_transforms: Data augmentation
```

### Training Script
```
bob/bio/vein/script/train_cnn_vit.py
- Command-line training interface
- Data loading and splitting
- Model checkpointing
- Evaluation
```

### Configuration
```
bob/bio/vein/config/cnn_vit.py
- Model configuration
- Default hyperparameters
```

---

## References

### Architecture Inspiration
- CNN: ResNet, VGG
- Vision Transformer: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
- Hybrid approach: Custom design for vein recognition

### Framework
- PyTorch 1.9+
- torchvision 0.10+
- bob.bio.vein framework
