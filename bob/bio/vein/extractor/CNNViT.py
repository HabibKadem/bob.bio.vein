#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
CNN+ViT Hybrid Model for Vein Recognition
Combines CNN feature extraction with Vision Transformer attention mechanisms
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. CNN+ViT model will not work.")


class VeinCNNViTModel:
    """
    Hybrid CNN+Vision Transformer model for vein recognition.
    
    Architecture:
    1. CNN backbone for local feature extraction
    2. Vision Transformer for global context and attention
    3. Classification head for person identification
    
    Parameters
    ----------
    num_classes : int
        Number of people/classes in the dataset
    img_size : int
        Size of input images (assumed square)
    patch_size : int
        Size of patches for ViT
    embed_dim : int
        Embedding dimension for transformer
    num_heads : int
        Number of attention heads
    num_layers : int
        Number of transformer layers
    dropout : float
        Dropout rate
    """
    
    def __init__(
        self,
        num_classes=138,
        img_size=224,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CNN+ViT model. "
                "Please install it with: pip install torch torchvision"
            )
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Initialize the model
        self.model = CNNViTNetwork(
            num_classes=num_classes,
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training parameters
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()  # Initialize criterion
        self.is_trained = False
    
    def train(
        self,
        train_loader,
        val_loader=None,
        epochs=50,
        learning_rate=1e-4,
        weight_decay=1e-4,
        save_path='cnn_vit_model.pth',
    ):
        """
        Train the CNN+ViT model
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for optimizer
        weight_decay : float
            Weight decay for regularization
        save_path : str
            Path to save the trained model
        """
        self.model.train()
        
        # Setup optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training phase
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
            
            train_acc = 100.0 * train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader)
                
                print(
                    f'Epoch [{epoch+1}/{epochs}] '
                    f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}% '
                    f'Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%'
                )
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model(save_path)
                    print(f'Model saved with validation accuracy: {val_acc:.2f}%')
            else:
                print(
                    f'Epoch [{epoch+1}/{epochs}] '
                    f'Train Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}%'
                )
            
            scheduler.step()
        
        self.is_trained = True
        print('Training completed!')
    
    def evaluate(self, data_loader):
        """
        Evaluate the model on a dataset
        
        Parameters
        ----------
        data_loader : DataLoader
            Data loader for evaluation
        
        Returns
        -------
        loss : float
            Average loss
        accuracy : float
            Classification accuracy
        """
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        accuracy = 100.0 * correct / total
        avg_loss = total_loss / len(data_loader)
        
        self.model.train()
        return avg_loss, accuracy
    
    def extract_features(self, image):
        """
        Extract feature embeddings from an image
        
        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image
        
        Returns
        -------
        features : np.ndarray
            Feature vector
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Convert to tensor
                if image.ndim == 2:
                    image = image[np.newaxis, np.newaxis, ...]
                elif image.ndim == 3:
                    image = image[np.newaxis, ...]
                image = torch.from_numpy(image).float()
            
            image = image.to(self.device)
            features = self.model.extract_features(image)
            
            return features.cpu().numpy()
    
    def predict(self, image):
        """
        Predict the class of an image
        
        Parameters
        ----------
        image : np.ndarray or torch.Tensor
            Input image
        
        Returns
        -------
        class_id : int
            Predicted class ID
        confidence : float
            Confidence score
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                if image.ndim == 2:
                    image = image[np.newaxis, np.newaxis, ...]
                elif image.ndim == 3:
                    image = image[np.newaxis, ...]
                image = torch.from_numpy(image).float()
            
            image = image.to(self.device)
            outputs = self.model(image)
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
            
            return predicted.item(), confidence.item()
    
    def save_model(self, path):
        """Save model weights"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
        }, path)
    
    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_trained = True


class CNNViTNetwork(nn.Module):
    """
    CNN+ViT Hybrid Network Architecture
    """
    
    def __init__(
        self,
        num_classes=138,
        img_size=224,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
        in_channels=1,  # Grayscale images
    ):
        super(CNNViTNetwork, self).__init__()
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN Backbone for local feature extraction
        self.cnn_backbone = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        # Calculate feature map size after CNN
        cnn_output_size = img_size // 4  # After 2 max pooling layers
        
        # Patch embedding for ViT
        self.num_patches = (cnn_output_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            256, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        # Use torch.nn.init.trunc_normal_ (correct function)
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass"""
        # CNN feature extraction
        x = self.cnn_backbone(x)
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Classification using CLS token
        x = self.norm(x[:, 0])
        x = self.classifier(x)
        
        return x
    
    def extract_features(self, x):
        """Extract feature embeddings (before classification)"""
        # CNN feature extraction
        x = self.cnn_backbone(x)
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add class token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Return CLS token embedding
        x = self.norm(x[:, 0])
        
        return x


class VeinDataset(Dataset):
    """
    Dataset class for vein images
    """
    
    def __init__(self, images, labels, transform=None):
        """
        Parameters
        ----------
        images : list of np.ndarray
            List of images
        labels : list of int
            List of labels
        transform : callable, optional
            Transform to apply to images
        """
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = image[np.newaxis, ...]  # Add channel dimension
            image = torch.from_numpy(image).float()
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Normalize to [0, 1] if needed
        if image.max() > 1.0:
            image = image / 255.0
        
        return image, label


def get_transforms(img_size=224, augment=False):
    """
    Get data transforms
    
    Parameters
    ----------
    img_size : int
        Target image size
    augment : bool
        Whether to apply data augmentation
    
    Returns
    -------
    transform : callable
        Transform function
    """
    if augment:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    
    return transform
