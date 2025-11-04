"""
Capsule Neural Network for Dorsal Hand Veins Classification (PyTorch)
======================================================================

This module implements a Capsule Neural Network (CapsNet) using PyTorch for
classifying dorsal hand vein images. The architecture is designed to be
efficient and easy to understand, with detailed comments throughout.

Dataset Structure Expected:
    DorsalHandVeins_DB1_png/
        train/
            person_001_db1_L1.png
            person_001_db1_L2.png
            ...
            person_276_db1_L4.png

Where:
    - Images are grayscale
    - Y ranges from 1 to 4 (4 images per person)
    - XXX ranges from 1 to 276 (276 different persons)

Architecture Overview:
    1. Initial Convolution: Extract low-level features
    2. Primary Capsules: Convert features to capsule vectors
    3. Digit Capsules: One capsule per class (person) for classification
    4. Dynamic Routing: Route information between capsule layers

Key Benefits of CapsNet for Vein Recognition:
    - Better handling of spatial relationships
    - Equivariance to transformations (rotation, translation)
    - Requires less data augmentation
    - Better generalization with limited training samples

Memory-Efficient Design:
    - Uses weight sharing across spatial locations in DigitCapsule layer
    - This reduces parameters from ~12B to ~1.4M for the routing layer
    - Enables training on standard GPUs with 4-8GB VRAM
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

# Image and training parameters
IMG_SIZE = 224              # Input image size (224x224)
BATCH_SIZE = 16             # Batch size for training
NUM_CLASSES = 276           # Number of persons in the dataset
EPOCHS = 200                # Maximum number of training epochs
LEARNING_RATE = 0.001       # Initial learning rate

# CapsNet specific parameters
PRIMARY_CAPS_DIM = 8        # Dimension of primary capsule vectors
DIGIT_CAPS_DIM = 16         # Dimension of digit (class) capsule vectors
NUM_ROUTING = 3             # Number of dynamic routing iterations

# Dataset path
DATASET_DIR = "DorsalHandVeins_DB1_png/train/"

# Check GPU availability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================================
# 2. DATASET CLASS
# ============================================================================

class VeinDataset(Dataset):
    """
    PyTorch Dataset for loading dorsal hand vein images.
    
    Args:
        file_paths: List of image file paths
        transform: Optional torchvision transforms
        augment: Whether to apply data augmentation
    """
    
    def __init__(self, file_paths, transform=None, augment=False):
        self.file_paths = file_paths
        self.transform = transform
        self.augment = augment
        
        # Parse labels from filenames
        self.labels = [self._parse_label(path) for path in file_paths]
    
    def _parse_label(self, path):
        """
        Extract person ID from filename pattern: person_XXX_db1_LY.png
        
        Returns:
            Zero-based integer label (person_id - 1)
        """
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        if len(parts) < 2:
            raise ValueError(f"Cannot extract label from filename: {filename}")
        
        # Extract person number (second part after split)
        person_id_str = os.path.splitext(parts[1])[0]
        person_id = int(person_id_str)
        
        # Convert to zero-based label
        return person_id - 1
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Load and return a single image-label pair.
        
        Returns:
            image: Tensor of shape [1, IMG_SIZE, IMG_SIZE]
            label: Integer class label
        """
        # Load image
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('L')  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.labels[idx]
        
        return image, label


def get_transforms(augment=False):
    """
    Create transform pipeline for images.
    
    Args:
        augment: Whether to include data augmentation
        
    Returns:
        torchvision.transforms.Compose object
    """
    if augment:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),  # Converts to [0, 1] and [C, H, W]
        ])
    else:
        # Validation/test transforms without augmentation
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])


def list_image_files(root_dir, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    """
    Recursively list all image files in the directory.
    
    Args:
        root_dir: Root directory to search
        exts: Tuple of valid image extensions
        
    Returns:
        List of sorted file paths
    """
    files = []
    for root, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    
    files = sorted(files)
    
    if not files:
        raise RuntimeError(f"No image files found in {root_dir}")
    
    return files


# ============================================================================
# 3. CAPSULE NETWORK LAYERS (PyTorch)
# ============================================================================

def squash(vectors, dim=-1):
    """
    Squash function: non-linear activation for capsules.
    Ensures capsule output has length between 0 and 1.
    
    Formula:
        squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||
    
    Args:
        vectors: Capsule vectors to squash
        dim: Dimension along which to compute the norm
        
    Returns:
        Squashed vectors with same shape as input
    """
    # Compute squared norm
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    
    # Compute scale factor
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    
    # Apply squashing
    return scale * vectors


class PrimaryCapsule(nn.Module):
    """
    Primary Capsule Layer: Converts conv features to capsule vectors.
    
    This layer:
    1. Applies convolution to extract features
    2. Reshapes output into capsule format
    3. Applies squash activation
    
    Args:
        in_channels: Number of input channels from previous layer
        num_capsules: Number of primary capsule types
        capsule_dim: Dimension of each capsule vector
        kernel_size: Size of convolutional kernel
        stride: Stride for convolution
        padding: Padding for convolution
    """
    
    def __init__(self, in_channels, num_capsules, capsule_dim, 
                 kernel_size=9, stride=2, padding=0):
        super(PrimaryCapsule, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Convolution layer to generate capsule components
        # Output channels = num_capsules * capsule_dim
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        """
        Forward pass through primary capsule layer.
        
        Args:
            x: Feature maps from previous layer [batch, in_channels, h, w]
            
        Returns:
            Primary capsules [batch, num_primary_caps, capsule_dim]
        """
        # Apply convolution
        # Output: [batch, num_capsules * capsule_dim, h, w]
        out = self.conv(x)
        
        batch_size = out.size(0)
        
        # Reshape to capsule format
        # [batch, num_capsules * capsule_dim, h, w] -> [batch, num_capsules, capsule_dim, h, w]
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        
        # Flatten spatial dimensions
        # [batch, num_capsules, capsule_dim, h*w] -> [batch, num_capsules * h * w, capsule_dim]
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, -1, self.capsule_dim)
        
        # Apply squash activation
        return squash(out)


class DigitCapsule(nn.Module):
    """
    Digit Capsule Layer: High-level capsules for classification.
    
    Implements dynamic routing between capsules:
    1. Transform lower capsules via learned weight matrices
    2. Route information through iterative agreement process
    3. Output one capsule per class
    
    Args:
        num_capsules: Number of digit capsules (= num_classes)
        num_routes: Number of input capsules (primary capsule types, not spatial locations)
        in_capsule_dim: Dimension of input capsule vectors
        out_capsule_dim: Dimension of output capsule vectors
        num_routing: Number of routing iterations
    """
    
    def __init__(self, num_capsules, num_routes, in_capsule_dim, 
                 out_capsule_dim, num_routing=3):
        super(DigitCapsule, self).__init__()
        
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_capsule_dim = in_capsule_dim
        self.out_capsule_dim = out_capsule_dim
        self.num_routing = num_routing
        
        # Weight matrix W: transforms primary capsules to digit capsule space
        # Shape: [1, num_routes, num_capsules, in_capsule_dim, out_capsule_dim]
        # We share weights across all spatial locations of primary capsules
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, in_capsule_dim, out_capsule_dim) * 0.01
        )
    
    def forward(self, x):
        """
        Forward pass with dynamic routing.
        
        Args:
            x: Primary capsules [batch, num_primary_caps_total, in_capsule_dim]
                where num_primary_caps_total = spatial_locations * num_capsule_types
            
        Returns:
            Digit capsules [batch, num_capsules, out_capsule_dim]
        """
        batch_size = x.size(0)
        num_primary_caps_total = x.size(1)
        
        # Reshape x to separate capsule types from spatial locations
        # We have num_routes capsule types, each repeated across spatial locations
        # [batch, num_primary_caps_total, in_capsule_dim]
        # -> [batch, num_primary_caps_total, 1, in_capsule_dim, 1]
        x_expanded = x.unsqueeze(2).unsqueeze(4)
        
        # Expand W for all spatial locations
        # W shape: [1, num_routes, num_capsules, in_capsule_dim, out_capsule_dim]
        # We need to tile it for each spatial location
        # First, determine spatial size
        num_spatial = num_primary_caps_total // self.num_routes
        
        # Tile W across spatial locations
        # [1, num_routes, num_capsules, in_capsule_dim, out_capsule_dim]
        # -> [1, num_primary_caps_total, num_capsules, in_capsule_dim, out_capsule_dim]
        W_tiled = self.W.repeat(1, num_spatial, 1, 1, 1)
        W_tiled = W_tiled.view(1, num_primary_caps_total, self.num_capsules, 
                                self.in_capsule_dim, self.out_capsule_dim)
        
        # Compute prediction vectors u_hat
        # u_hat_j|i = W_ij * u_i
        # [batch, num_primary_caps_total, num_capsules, out_capsule_dim]
        u_hat = torch.matmul(x_expanded, W_tiled).squeeze(4)
        
        # Dynamic Routing Algorithm
        # Initialize routing logits b to zero
        # [batch, num_primary_caps_total, num_capsules]
        b = torch.zeros(batch_size, num_primary_caps_total, self.num_capsules).to(x.device)
        
        # Routing iterations
        for iteration in range(self.num_routing):
            # Compute coupling coefficients via softmax
            # c_ij represents how much capsule i should send to capsule j
            c = F.softmax(b, dim=2)  # [batch, num_primary_caps_total, num_capsules]
            
            # Expand for broadcasting
            c = c.unsqueeze(3)  # [batch, num_primary_caps_total, num_capsules, 1]
            
            # Weighted sum of predictions: s = sum(c_ij * u_hat_j|i)
            s = (c * u_hat).sum(dim=1)  # [batch, num_capsules, out_capsule_dim]
            
            # Apply squash activation to get output capsules
            v = squash(s, dim=2)  # [batch, num_capsules, out_capsule_dim]
            
            # Update routing logits (except on last iteration)
            if iteration < self.num_routing - 1:
                # Expand v for agreement calculation
                v_expand = v.unsqueeze(1)  # [batch, 1, num_capsules, out_capsule_dim]
                
                # Agreement: dot product between u_hat and v
                # Higher agreement increases routing coefficient
                agreement = (u_hat * v_expand).sum(dim=3)  # [batch, num_primary_caps_total, num_capsules]
                
                # Update routing logits
                b = b + agreement
        
        return v


class CapsNet(nn.Module):
    """
    Complete Capsule Network for vein classification.
    
    Architecture:
        1. Initial Conv Layer: Extract low-level features (edges, textures)
        2. Primary Capsules: Convert features to capsule vectors
        3. Digit Capsules: One capsule per person class
        4. Length Layer: Convert capsule lengths to probabilities
    
    Args:
        num_classes: Number of person classes
        input_channels: Number of input channels (1 for grayscale)
        primary_caps_dim: Dimension of primary capsule vectors
        digit_caps_dim: Dimension of digit capsule vectors
        num_routing: Number of routing iterations
    """
    
    def __init__(self, num_classes=NUM_CLASSES, input_channels=1,
                 primary_caps_dim=PRIMARY_CAPS_DIM, 
                 digit_caps_dim=DIGIT_CAPS_DIM,
                 num_routing=NUM_ROUTING):
        super(CapsNet, self).__init__()
        
        self.num_classes = num_classes
        
        # ====================================================================
        # Layer 1: Initial Convolution
        # Extract basic features from the vein image
        # ====================================================================
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=256,       # Number of feature maps
            kernel_size=9,          # Large kernel to capture vein patterns
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        
        # After conv1: [batch, 256, 216, 216] for 224x224 input
        
        # ====================================================================
        # Layer 2: Primary Capsules
        # Convert conv features to capsule representation
        # ====================================================================
        self.primary_capsules = PrimaryCapsule(
            in_channels=256,
            num_capsules=32,
            capsule_dim=primary_caps_dim,
            kernel_size=9,
            stride=2,
            padding=0
        )
        
        # After primary_capsules: spatial size is (216-9)/2 + 1 = 104
        # We have 32 capsule types, each at 104x104 spatial locations
        # Total primary capsules: 104 * 104 * 32 = 345,088
        # But for routing, we only need to know there are 32 capsule types
        num_capsule_types = 32
        
        # ====================================================================
        # Layer 3: Digit Capsules
        # One capsule per person class with dynamic routing
        # ====================================================================
        self.digit_capsules = DigitCapsule(
            num_capsules=num_classes,
            num_routes=num_capsule_types,  # Number of capsule types, not total capsules
            in_capsule_dim=primary_caps_dim,
            out_capsule_dim=digit_caps_dim,
            num_routing=num_routing
        )
    
    def forward(self, x):
        """
        Forward pass through CapsNet.
        
        Args:
            x: Input images [batch, 1, IMG_SIZE, IMG_SIZE]
            
        Returns:
            Capsule lengths [batch, num_classes]
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.relu(x)
        
        # Primary capsules
        x = self.primary_capsules(x)
        
        # Digit capsules
        x = self.digit_capsules(x)
        
        # Compute capsule lengths (class probabilities)
        # Length of capsule vector represents existence probability
        lengths = torch.sqrt((x ** 2).sum(dim=2))
        
        return lengths


# ============================================================================
# 4. MARGIN LOSS FOR CAPSNET
# ============================================================================

class MarginLoss(nn.Module):
    """
    Margin loss for Capsule Networks.
    
    For each class k:
        L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2
    
    Where:
        - T_k = 1 if class k is present, 0 otherwise
        - ||v_k|| = length of capsule k (predicted probability)
        - m+ = 0.9 (target for present class)
        - m- = 0.1 (target for absent class)
        - lambda = 0.5 (down-weighting factor for absent classes)
    
    Args:
        m_plus: Margin for present class (default: 0.9)
        m_minus: Margin for absent class (default: 0.1)
        lambda_: Down-weighting factor (default: 0.5)
    """
    
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
    
    def forward(self, lengths, labels):
        """
        Compute margin loss.
        
        Args:
            lengths: Predicted capsule lengths [batch, num_classes]
            labels: True labels [batch]
            
        Returns:
            Margin loss (scalar)
        """
        batch_size = lengths.size(0)
        num_classes = lengths.size(1)
        
        # Convert labels to one-hot encoding
        labels_one_hot = torch.zeros(batch_size, num_classes).to(lengths.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # Loss for present classes
        # Penalize if capsule length < m_plus
        present_loss = labels_one_hot * torch.clamp(self.m_plus - lengths, min=0) ** 2
        
        # Loss for absent classes
        # Penalize if capsule length > m_minus
        absent_loss = self.lambda_ * (1 - labels_one_hot) * torch.clamp(lengths - self.m_minus, min=0) ** 2
        
        # Total loss: sum over all classes, mean over batch
        loss = (present_loss + absent_loss).sum(dim=1).mean()
        
        return loss


# ============================================================================
# 5. TRAINING AND EVALUATION
# ============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: CapsNet model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run on
        
    Returns:
        avg_loss: Average loss for the epoch
        accuracy: Training accuracy
    """
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        # Move to device
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: CapsNet model
        dataloader: Validation/test data loader
        criterion: Loss function
        device: Device to run on
        
    Returns:
        avg_loss: Average loss
        accuracy: Accuracy
    """
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            # Move to device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    
    return avg_loss, accuracy


# ============================================================================
# 6. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Main function to train the CapsNet model.
    """
    print("=" * 80)
    print("CAPSULE NETWORK FOR DORSAL HAND VEINS CLASSIFICATION (PyTorch)")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Step 1: Load and prepare dataset
    # ========================================================================
    print("Step 1: Loading dataset...")
    
    # Get all image files
    all_files = list_image_files(DATASET_DIR)
    print(f"Found {len(all_files)} images")
    
    # Infer number of classes
    def infer_num_classes(files):
        dataset_temp = VeinDataset(files)
        labels = dataset_temp.labels
        return max(labels) + 1
    
    actual_num_classes = infer_num_classes(all_files)
    print(f"Detected {actual_num_classes} unique persons")
    
    if actual_num_classes != NUM_CLASSES:
        print(f"WARNING: NUM_CLASSES ({NUM_CLASSES}) doesn't match detected classes ({actual_num_classes})")
        print("Update NUM_CLASSES variable if needed")
    
    # Split into train and validation sets (70-30 split with stratification)
    temp_dataset = VeinDataset(all_files)
    labels = temp_dataset.labels
    
    train_files, val_files = train_test_split(
        all_files,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    
    print(f"Training set: {len(train_files)} images")
    print(f"Validation set: {len(val_files)} images")
    print()
    
    # ========================================================================
    # Step 2: Create datasets and data loaders
    # ========================================================================
    print("Step 2: Creating data loaders...")
    
    # Create datasets
    train_dataset = VeinDataset(
        train_files,
        transform=get_transforms(augment=True),
        augment=True
    )
    
    val_dataset = VeinDataset(
        val_files,
        transform=get_transforms(augment=False),
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print("Data loaders created successfully")
    print()
    
    # ========================================================================
    # Step 3: Build CapsNet model
    # ========================================================================
    print("Step 3: Building Capsule Network...")
    
    model = CapsNet(
        num_classes=NUM_CLASSES,
        input_channels=1,
        primary_caps_dim=PRIMARY_CAPS_DIM,
        digit_caps_dim=DIGIT_CAPS_DIM,
        num_routing=NUM_ROUTING
    )
    
    model = model.to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created with {total_params:,} total parameters")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # ========================================================================
    # Step 4: Setup loss and optimizer
    # ========================================================================
    print("Step 4: Setting up loss and optimizer...")
    
    criterion = MarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    print("Loss: Margin Loss")
    print("Optimizer: Adam")
    print("Scheduler: ReduceLROnPlateau")
    print()
    
    # ========================================================================
    # Step 5: Training loop
    # ========================================================================
    print("Step 5: Training CapsNet...")
    print("=" * 80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        
        # Evaluate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_capsnet_pytorch.pth')
            print(f"  → Best model saved (val_loss: {val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print()
    print("=" * 80)
    print("Training completed!")
    print()
    
    # ========================================================================
    # Step 6: Save final model
    # ========================================================================
    print("Step 6: Saving final model...")
    
    torch.save(model.state_dict(), 'capsnet_pytorch_final.pth')
    print("✅ Final model saved: capsnet_pytorch_final.pth")
    print("✅ Best model saved: best_capsnet_pytorch.pth")
    print()
    
    # ========================================================================
    # Step 7: Plot training history
    # ========================================================================
    print("Step 7: Plotting training metrics...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Training Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    ax1.set_title('CapsNet Training and Validation Loss (PyTorch)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Training Accuracy', linewidth=2)
    ax2.plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    ax2.set_title('CapsNet Training and Validation Accuracy (PyTorch)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('capsnet_training_metrics_pytorch.png', dpi=300, bbox_inches='tight')
    print("✅ Training metrics plot saved: capsnet_training_metrics_pytorch.png")
    
    # Display plot
    plt.show()
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Final training accuracy: {history['train_acc'][-1]:.4f}")
    print(f"  - Final validation accuracy: {history['val_acc'][-1]:.4f}")
    print(f"  - Best validation loss: {best_val_loss:.4f}")
    print()
    print("Next steps:")
    print("  1. Load the best model: model.load_state_dict(torch.load('best_capsnet_pytorch.pth'))")
    print("  2. Evaluate on test set if available")
    print("  3. Use the model for inference on new images")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
