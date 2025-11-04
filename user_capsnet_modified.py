"""
Capsule Neural Network for Dorsal Hand Veins Classification
Modified from user's CNN code to use CapsNet architecture
"""
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchinfo import summary

# -------------------------------
# 1. Configuration
# -------------------------------
IMG_SIZE = 224
BATCH_SIZE = 16
NUM_CLASSES = 276
EPOCHS = 200
LR = 0.001

DATASET_DIR = "DorsalHandVeins_DB1_png/train"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -------------------------------
# 2. Dataset (unchanged from your code)
# -------------------------------
class VeinDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert('L').resize((IMG_SIZE, IMG_SIZE))
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)

        filename = os.path.basename(img_path)
        person_id = int(filename.split('_')[1]) - 1  # "001" → 0

        return img, person_id

# -------------------------------
# 3. Charger les images + Split (unchanged from your code)
# -------------------------------
all_images = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if f.endswith('.png')]

train_files, val_files = train_test_split(
    all_images, test_size=0.2, random_state=276
)

train_dataset = VeinDataset(train_files)
val_dataset = VeinDataset(val_files)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"✅ Dataset chargé: {len(train_dataset)} images pour l'entraînement, {len(val_dataset)} pour la validation.")

# -------------------------------
# 4. CapsNet Components
# -------------------------------

def squash(vectors, dim=-1):
    """
    Squash activation function for capsules.
    Ensures output vector length is between 0 and 1.
    """
    squared_norm = (vectors ** 2).sum(dim=dim, keepdim=True)
    scale = squared_norm / (1 + squared_norm) / torch.sqrt(squared_norm + 1e-8)
    return scale * vectors


class PrimaryCapsule(nn.Module):
    """
    Primary Capsule Layer: converts conv features to capsule vectors.
    """
    def __init__(self, in_channels, num_capsules, capsule_dim, 
                 kernel_size=9, stride=2, padding=0):
        super(PrimaryCapsule, self).__init__()
        
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        
        # Convolution to generate capsule components
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_capsules * capsule_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, h, w]
        Returns:
            Primary capsules [batch, num_primary_caps_total, capsule_dim]
        """
        # Apply convolution
        out = self.conv(x)  # [batch, num_capsules * capsule_dim, h, w]
        
        batch_size = out.size(0)
        
        # Reshape to capsule format
        out = out.view(batch_size, self.num_capsules, self.capsule_dim, -1)
        
        # Flatten spatial dimensions
        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(batch_size, -1, self.capsule_dim)
        
        # Apply squash activation
        return squash(out)


class DigitCapsule(nn.Module):
    """
    Digit Capsule Layer: high-level capsules for classification.
    Uses weight sharing across spatial locations for memory efficiency.
    """
    def __init__(self, num_capsules, num_routes, in_capsule_dim, 
                 out_capsule_dim, num_routing=3):
        super(DigitCapsule, self).__init__()
        
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_capsule_dim = in_capsule_dim
        self.out_capsule_dim = out_capsule_dim
        self.num_routing = num_routing
        
        # Weight matrix with weight sharing across spatial locations
        # Shape: [1, num_routes, num_capsules, in_capsule_dim, out_capsule_dim]
        self.W = nn.Parameter(
            torch.randn(1, num_routes, num_capsules, in_capsule_dim, out_capsule_dim) * 0.01
        )
    
    def forward(self, x):
        """
        Dynamic routing algorithm.
        
        Args:
            x: Primary capsules [batch, num_primary_caps_total, in_capsule_dim]
        Returns:
            Digit capsules [batch, num_capsules, out_capsule_dim]
        """
        batch_size = x.size(0)
        num_primary_caps_total = x.size(1)
        
        # Reshape and expand for matrix multiplication
        x_expanded = x.unsqueeze(2).unsqueeze(4)
        
        # Tile weights across spatial locations
        num_spatial = num_primary_caps_total // self.num_routes
        
        # Handle case where division doesn't result in exact multiple
        if num_primary_caps_total % self.num_routes != 0:
            # Adjust by trimming or padding
            expected_total = num_spatial * self.num_routes
            if num_primary_caps_total > expected_total:
                # Trim extra capsules
                x_expanded = x_expanded[:, :expected_total, :, :, :]
                x = x[:, :expected_total, :]
                num_primary_caps_total = expected_total
        
        W_tiled = self.W.repeat(1, num_spatial, 1, 1, 1)
        W_tiled = W_tiled.view(1, num_primary_caps_total, self.num_capsules, 
                                self.in_capsule_dim, self.out_capsule_dim)
        
        # Compute prediction vectors
        u_hat = torch.matmul(x_expanded, W_tiled).squeeze(4)
        
        # Dynamic routing
        b = torch.zeros(batch_size, num_primary_caps_total, self.num_capsules).to(x.device)
        
        for iteration in range(self.num_routing):
            # Coupling coefficients
            c = F.softmax(b, dim=2)
            c = c.unsqueeze(3)
            
            # Weighted sum
            s = (c * u_hat).sum(dim=1)
            
            # Squash activation
            v = squash(s, dim=2)
            
            # Update routing logits
            if iteration < self.num_routing - 1:
                v_expand = v.unsqueeze(1)
                agreement = (u_hat * v_expand).sum(dim=3)
                b = b + agreement
        
        return v


class VeinCapsNet(nn.Module):
    """
    Capsule Network for vein classification.
    
    Architecture:
        1. Initial Conv: Extract low-level features
        2. Primary Capsules: Convert to capsule vectors
        3. Digit Capsules: One capsule per class with dynamic routing
    """
    def __init__(self, num_classes=NUM_CLASSES, input_channels=1,
                 primary_caps_dim=8, digit_caps_dim=16, num_routing=3):
        super(VeinCapsNet, self).__init__()
        
        self.num_classes = num_classes
        
        # Layer 1: Initial convolution (like your CNN)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        
        # Layer 2: Primary Capsules
        self.primary_capsules = PrimaryCapsule(
            in_channels=128,
            num_capsules=16,
            capsule_dim=primary_caps_dim,
            kernel_size=3,
            stride=2,
            padding=0
        )
        
        # Layer 3: Digit Capsules (one per person class)
        num_capsule_types = 16
        self.digit_capsules = DigitCapsule(
            num_capsules=num_classes,
            num_routes=num_capsule_types,
            in_capsule_dim=primary_caps_dim,
            out_capsule_dim=digit_caps_dim,
            num_routing=num_routing
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input images [batch, 1, IMG_SIZE, IMG_SIZE]
        Returns:
            Class scores [batch, num_classes]
        """
        # Initial convolution
        x = self.conv1(x)
        x = self.relu(x)
        
        # Primary capsules
        x = self.primary_capsules(x)
        
        # Digit capsules
        x = self.digit_capsules(x)
        
        # Compute capsule lengths (class probabilities)
        lengths = torch.sqrt((x ** 2).sum(dim=2))
        
        return lengths


class MarginLoss(nn.Module):
    """
    Margin loss for CapsNet.
    Encourages correct class capsule length > 0.9, incorrect < 0.1
    """
    def __init__(self, m_plus=0.9, m_minus=0.1, lambda_=0.5):
        super(MarginLoss, self).__init__()
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.lambda_ = lambda_
    
    def forward(self, lengths, labels):
        """
        Args:
            lengths: Predicted capsule lengths [batch, num_classes]
            labels: True labels [batch]
        Returns:
            Loss (scalar)
        """
        batch_size = lengths.size(0)
        num_classes = lengths.size(1)
        
        # Convert labels to one-hot
        labels_one_hot = torch.zeros(batch_size, num_classes).to(lengths.device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1.0)
        
        # Loss for present classes
        present_loss = labels_one_hot * torch.clamp(self.m_plus - lengths, min=0) ** 2
        
        # Loss for absent classes
        absent_loss = self.lambda_ * (1 - labels_one_hot) * torch.clamp(lengths - self.m_minus, min=0) ** 2
        
        # Total loss
        loss = (present_loss + absent_loss).sum(dim=1).mean()
        
        return loss


# -------------------------------
# 5. Créer le modèle CapsNet
# -------------------------------
model = VeinCapsNet(NUM_CLASSES).to(DEVICE)
criterion = MarginLoss()  # CapsNet uses margin loss instead of cross-entropy
optimizer = optim.Adam(model.parameters(), lr=LR)

print("\n" + "="*80)
print("CAPSNET MODEL SUMMARY")
print("="*80)
try:
    summary(model, input_size=(1, 1, IMG_SIZE, IMG_SIZE))
except:
    print("Model architecture created successfully")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
print("="*80 + "\n")

# -------------------------------
# Early Stopping (unchanged from your code)
# -------------------------------
class EarlyStopping:
    def __init__(self, patience=7, delta=0, path="checkpoint.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"⏳ EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# -------------------------------
# 6. Entraînement (adapted for CapsNet)
# -------------------------------
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
early_stopping = EarlyStopping(patience=15, path="best_capsnet_model.pt")  # More patience for CapsNet

print("Starting training...")
print("="*80)

for epoch in range(EPOCHS):
    # Training
    model.train()
    total_loss, correct = 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Predictions are based on capsule lengths
        preds = outputs.argmax(1)
        correct += (preds == y).sum().item()

    train_acc = correct / len(train_dataset)
    avg_train_loss = total_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)

    # Validation
    model.eval()
    val_loss, val_correct = 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            val_loss += criterion(outputs, y).item()
            val_correct += (outputs.argmax(1) == y).sum().item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / len(val_dataset)
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss={avg_train_loss:.4f} Acc={train_acc:.4f} | "
          f"Val Loss={avg_val_loss:.4f} Acc={val_acc:.4f}")

    # Early stopping check
    early_stopping(avg_val_loss, model)
    
    if early_stopping.early_stop:
        print("🛑 Early stopping déclenché")
        break

print("="*80)
print("Training completed!")
print("="*80 + "\n")

# -------------------------------
# 7. Sauvegarde du modèle
# -------------------------------
torch.save(model.state_dict(), "vein_capsnet_model.pth")
print("✅ Modèle sauvegardé : vein_capsnet_model.pth")
print("✅ Meilleur modèle sauvegardé : best_capsnet_model.pt")

# -------------------------------
# 8. Affichage des courbes
# -------------------------------
plt.figure(figsize=(12,5))

# Loss
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("CapsNet - Courbe de Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)

# Accuracy
plt.subplot(1,2,2)
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("CapsNet - Courbe d'Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("capsnet_training_curves.png", dpi=300, bbox_inches='tight')
print("✅ Courbes sauvegardées : capsnet_training_curves.png")
plt.show()

# Print final summary
print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)
print(f"Final Training Accuracy: {train_accuracies[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracies[-1]:.4f}")
print(f"Best Validation Loss: {early_stopping.val_loss_min:.4f}")
print("="*80)
