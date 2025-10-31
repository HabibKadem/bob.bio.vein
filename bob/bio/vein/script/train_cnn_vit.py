#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Training script for CNN+ViT model on DorsalHandVeins dataset
"""

import os
import argparse
import numpy as np
from pathlib import Path
import glob

try:
    import torch
    from torch.utils.data import DataLoader
    import bob.io.base
    from bob.bio.vein.extractor.CNNViT import (
        VeinCNNViTModel,
        VeinDataset,
        get_transforms,
    )
    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_AVAILABLE = False
    print(f"Warning: Required packages not available: {e}")


def load_dorsal_hand_vein_data(data_dir, split_ratio=(0.7, 0.15, 0.15)):
    """
    Load DorsalHandVeins dataset
    
    Parameters
    ----------
    data_dir : str
        Path to the dataset directory
    split_ratio : tuple
        Train/dev/test split ratio
    
    Returns
    -------
    train_data : tuple
        (images, labels) for training
    dev_data : tuple
        (images, labels) for validation
    test_data : tuple
        (images, labels) for testing
    """
    train_dir = os.path.join(data_dir, 'train')
    
    if not os.path.exists(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    # Get all image files
    image_files = sorted(glob.glob(os.path.join(train_dir, 'person_*_db1_L*.png')))
    
    if not image_files:
        raise ValueError(f"No images found in {train_dir}")
    
    print(f"Found {len(image_files)} images")
    
    # Organize by subject
    subjects = {}
    for img_path in image_files:
        filename = os.path.basename(img_path)
        # Parse: person_XXX_db1_LY.png
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 4:
            person_id = parts[1]  # XXX
            
            if person_id not in subjects:
                subjects[person_id] = []
            subjects[person_id].append(img_path)
    
    print(f"Found {len(subjects)} subjects")
    
    # Create label mapping
    subject_ids = sorted(subjects.keys())
    label_map = {sid: idx for idx, sid in enumerate(subject_ids)}
    
    # Split subjects
    n_subjects = len(subject_ids)
    n_train = int(n_subjects * split_ratio[0])
    n_dev = int(n_subjects * split_ratio[1])
    
    train_subjects = subject_ids[:n_train]
    dev_subjects = subject_ids[n_train:n_train + n_dev]
    test_subjects = subject_ids[n_train + n_dev:]
    
    print(f"Split: {len(train_subjects)} train, {len(dev_subjects)} dev, {len(test_subjects)} test subjects")
    
    # Load images and create datasets
    def load_subject_data(subject_list):
        images = []
        labels = []
        for sid in subject_list:
            for img_path in subjects[sid]:
                try:
                    img = bob.io.base.load(img_path)
                    if img is not None:
                        # Ensure grayscale
                        if img.ndim == 3:
                            img = img.mean(axis=0)
                        images.append(img)
                        labels.append(label_map[sid])
                except Exception as e:
                    print(f"Warning: Failed to load {img_path}: {e}")
        return np.array(images), np.array(labels)
    
    train_images, train_labels = load_subject_data(train_subjects)
    dev_images, dev_labels = load_subject_data(dev_subjects)
    test_images, test_labels = load_subject_data(test_subjects)
    
    print(f"Loaded {len(train_images)} train, {len(dev_images)} dev, {len(test_images)} test images")
    
    return (
        (train_images, train_labels),
        (dev_images, dev_labels),
        (test_images, test_labels),
        label_map,
    )


def train_cnn_vit_model(
    data_dir,
    output_dir='models',
    img_size=224,
    batch_size=16,
    epochs=50,
    learning_rate=1e-4,
    num_workers=4,
):
    """
    Train CNN+ViT model on DorsalHandVeins dataset
    
    Parameters
    ----------
    data_dir : str
        Path to the dataset
    output_dir : str
        Directory to save models
    img_size : int
        Input image size
    batch_size : int
        Batch size for training
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate
    num_workers : int
        Number of data loader workers
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch and required packages are not available. "
            "Please install them with: pip install torch torchvision bob.io.base"
        )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("Loading dataset...")
    (train_images, train_labels), (dev_images, dev_labels), (test_images, test_labels), label_map = load_dorsal_hand_vein_data(data_dir)
    
    num_classes = len(label_map)
    print(f"Number of classes: {num_classes}")
    
    # Create datasets
    train_transform = get_transforms(img_size=img_size, augment=True)
    test_transform = get_transforms(img_size=img_size, augment=False)
    
    train_dataset = VeinDataset(train_images, train_labels, transform=train_transform)
    dev_dataset = VeinDataset(dev_images, dev_labels, transform=test_transform)
    test_dataset = VeinDataset(test_images, test_labels, transform=test_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    # Initialize model
    print("Initializing CNN+ViT model...")
    model = VeinCNNViTModel(
        num_classes=num_classes,
        img_size=img_size,
        patch_size=16,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1,
    )
    
    # Train model
    print("Starting training...")
    model_path = os.path.join(output_dir, 'cnn_vit_dorsalhandveins.pth')
    model.train(
        train_loader=train_loader,
        val_loader=dev_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=model_path,
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_model(model_path)
    test_loss, test_acc = model.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    
    # Save label mapping
    label_map_path = os.path.join(output_dir, 'label_map.npy')
    np.save(label_map_path, label_map)
    print(f"Label mapping saved to {label_map_path}")
    
    print("\nTraining complete!")
    return model


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Train CNN+ViT model for dorsal hand vein recognition'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Path to the DorsalHandVeins dataset directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models (default: models)',
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=224,
        help='Input image size (default: 224)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size for training (default: 16)',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)',
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-4,
        help='Learning rate (default: 1e-4)',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of data loader workers (default: 4)',
    )
    
    args = parser.parse_args()
    
    # Train model
    train_cnn_vit_model(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
