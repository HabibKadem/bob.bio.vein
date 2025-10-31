#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Example script demonstrating how to use the CNN+ViT model for dorsal hand vein recognition
"""

import os
import sys


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic CNN+ViT Model Usage")
    print("=" * 60)
    
    try:
        from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel
        
        # Initialize the model
        model = VeinCNNViTModel(
            num_classes=138,  # Number of people in dataset
            img_size=224,     # Input image size
            patch_size=16,    # Patch size for ViT
            embed_dim=256,    # Embedding dimension
            num_heads=8,      # Number of attention heads
            num_layers=6,     # Number of transformer layers
            dropout=0.1,      # Dropout rate
        )
        
        print("✓ Model initialized successfully")
        print(f"  - Number of classes: {model.num_classes}")
        print(f"  - Image size: {model.img_size}x{model.img_size}")
        print(f"  - Device: {model.device}")
        
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("  Please install PyTorch: pip install torch torchvision")
        return False
    
    return True


def example_database_setup():
    """Database setup example"""
    print("\n" + "=" * 60)
    print("Example 2: DorsalHandVeins Database Setup")
    print("=" * 60)
    
    try:
        from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
        
        # Show available protocols
        protocols = DorsalHandVeinsDatabase.protocols()
        print("✓ Available protocols:")
        for protocol in protocols:
            print(f"  - {protocol}")
        
        print("\nTo use the database, set the path:")
        print("  bob config set bob.bio.vein.dorsalhandveins.directory /path/to/dataset")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def example_training_command():
    """Training command example"""
    print("\n" + "=" * 60)
    print("Example 3: Training the CNN+ViT Model")
    print("=" * 60)
    
    print("Command to train the model:")
    print("")
    print("  bob_bio_vein_train_cnn_vit.py \\")
    print("      --data-dir /path/to/DorsalHandVeins_DB1_png \\")
    print("      --output-dir models \\")
    print("      --img-size 224 \\")
    print("      --batch-size 16 \\")
    print("      --epochs 50 \\")
    print("      --learning-rate 0.0001")
    print("")
    print("This will:")
    print("  1. Load the dataset from the specified directory")
    print("  2. Split data into train/dev/test sets (70%/15%/15%)")
    print("  3. Train the CNN+ViT model for 50 epochs")
    print("  4. Save the best model to 'models/cnn_vit_dorsalhandveins.pth'")
    print("  5. Evaluate on the test set and report accuracy")
    
    return True


def example_inference():
    """Inference example"""
    print("\n" + "=" * 60)
    print("Example 4: Using Trained Model for Inference")
    print("=" * 60)
    
    print("Python code for inference:")
    print("")
    print("```python")
    print("from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel")
    print("import bob.io.base")
    print("")
    print("# Load trained model")
    print("model = VeinCNNViTModel(num_classes=138)")
    print("model.load_model('models/cnn_vit_dorsalhandveins.pth')")
    print("")
    print("# Load an image")
    print("image = bob.io.base.load('path/to/person_001_db1_L1.png')")
    print("")
    print("# Extract features")
    print("features = model.extract_features(image)")
    print("print(f'Feature shape: {features.shape}')")
    print("")
    print("# Predict person ID")
    print("class_id, confidence = model.predict(image)")
    print("print(f'Predicted person: {class_id}, Confidence: {confidence:.4f}')")
    print("```")
    
    return True


def example_dataset_structure():
    """Dataset structure example"""
    print("\n" + "=" * 60)
    print("Example 5: Expected Dataset Structure")
    print("=" * 60)
    
    print("Your dataset should be organized as follows:")
    print("")
    print("DorsalHandVeins_DB1_png/")
    print("    train/")
    print("        person_001_db1_L1.png")
    print("        person_001_db1_L2.png")
    print("        person_001_db1_L3.png")
    print("        person_001_db1_L4.png")
    print("        person_002_db1_L1.png")
    print("        person_002_db1_L2.png")
    print("        ...")
    print("        person_138_db1_L1.png")
    print("        person_138_db1_L2.png")
    print("        person_138_db1_L3.png")
    print("        person_138_db1_L4.png")
    print("")
    print("Key points:")
    print("  - Images must be in PNG format")
    print("  - Images should be grayscale")
    print("  - Naming pattern: person_XXX_db1_LY.png")
    print("    where XXX is person ID (001-138) and Y is image number (1-4)")
    
    return True


def example_custom_training():
    """Custom training example"""
    print("\n" + "=" * 60)
    print("Example 6: Custom Training Script")
    print("=" * 60)
    
    print("For custom training with your own code:")
    print("")
    print("```python")
    print("from bob.bio.vein.extractor.CNNViT import (")
    print("    VeinCNNViTModel, VeinDataset, get_transforms")
    print("from torch.utils.data import DataLoader")
    print("import numpy as np")
    print("")
    print("# Prepare your data")
    print("train_images = [...]  # List of numpy arrays")
    print("train_labels = [...]  # List of integers (person IDs)")
    print("")
    print("# Create dataset and data loader")
    print("transform = get_transforms(img_size=224, augment=True)")
    print("train_dataset = VeinDataset(train_images, train_labels, transform)")
    print("train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)")
    print("")
    print("# Initialize and train model")
    print("model = VeinCNNViTModel(num_classes=138)")
    print("model.train(")
    print("    train_loader=train_loader,")
    print("    epochs=50,")
    print("    learning_rate=1e-4,")
    print("    save_path='my_model.pth'")
    print(")")
    print("```")
    
    return True


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("CNN+ViT Model for Dorsal Hand Vein Recognition")
    print("Examples and Usage Guide")
    print("=" * 60)
    
    examples = [
        example_basic_usage,
        example_database_setup,
        example_training_command,
        example_inference,
        example_dataset_structure,
        example_custom_training,
    ]
    
    success_count = 0
    for example_func in examples:
        try:
            if example_func():
                success_count += 1
        except Exception as e:
            print(f"\n✗ Error in {example_func.__name__}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Completed {success_count}/{len(examples)} examples")
    print("=" * 60)
    
    print("\nFor more information, see:")
    print("  - doc/cnn_vit_guide.md")
    print("  - bob/bio/vein/extractor/CNNViT.py")
    print("  - bob/bio/vein/script/train_cnn_vit.py")
    
    print("\nQuick start:")
    print("  1. Install PyTorch: pip install torch torchvision")
    print("  2. Set dataset path: bob config set bob.bio.vein.dorsalhandveins.directory /path/to/dataset")
    print("  3. Train model: bob_bio_vein_train_cnn_vit.py --data-dir /path/to/dataset")
    
    return success_count == len(examples)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
