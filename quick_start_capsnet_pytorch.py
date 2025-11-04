"""
Quick Start Example: Using CapsNet (PyTorch) for Vein Classification
======================================================================

This is a minimal example showing how to use the PyTorch CapsNet implementation.
Perfect for quick testing or integration into your own pipeline.
"""

import sys
import os

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the main capsnet module
from capsnet_dorsal_vein_classification_pytorch import (
    CapsNet,
    MarginLoss,
    IMG_SIZE,
    NUM_CLASSES,
    PRIMARY_CAPS_DIM,
    DIGIT_CAPS_DIM,
    NUM_ROUTING,
    DEVICE
)

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def quick_inference_example():
    """
    Example: Load a trained model and make a prediction
    """
    print("=" * 60)
    print("QUICK INFERENCE EXAMPLE (PyTorch)")
    print("=" * 60)
    print()
    
    # Load the trained model
    model_path = 'best_capsnet_pytorch.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first by running:")
        print("   python capsnet_dorsal_vein_classification_pytorch.py")
        return
    
    print(f"Loading model: {model_path}")
    
    # Create model
    model = CapsNet(
        num_classes=NUM_CLASSES,
        input_channels=1,
        primary_caps_dim=PRIMARY_CAPS_DIM,
        digit_caps_dim=DIGIT_CAPS_DIM,
        num_routing=NUM_ROUTING
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    print("✅ Model loaded\n")
    
    # Load and preprocess an image
    image_path = 'DorsalHandVeins_DB1_png/train/person_001_db1_L1.png'
    
    if not os.path.exists(image_path):
        print(f"❌ Example image not found: {image_path}")
        print("   Please provide a valid image path")
        return
    
    print(f"Loading image: {image_path}")
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert('L')
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(DEVICE)
    
    print(f"Image shape: {img_tensor.shape}\n")
    
    # Make prediction
    print("Making prediction...")
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Get top prediction
    confidences = outputs.cpu().numpy()[0]
    predicted_class = np.argmax(confidences)
    confidence = confidences[predicted_class]
    predicted_person = predicted_class + 1
    
    print(f"✅ Prediction complete")
    print(f"\nPredicted Person: {predicted_person:03d}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 5 predictions
    print("\nTop 5 Predictions:")
    top_5_idx = np.argsort(confidences)[-5:][::-1]
    for i, idx in enumerate(top_5_idx, 1):
        person_id = idx + 1
        conf = confidences[idx]
        print(f"  {i}. Person {person_id:03d}: {conf:.4f}")


def quick_training_example():
    """
    Example: Train a CapsNet model with minimal code
    """
    print("=" * 60)
    print("QUICK TRAINING EXAMPLE (PyTorch)")
    print("=" * 60)
    print()
    
    print("For a complete training example, run:")
    print("  python capsnet_dorsal_vein_classification_pytorch.py")
    print()
    print("This script contains the full pipeline:")
    print("  • Data loading with PyTorch Dataset and DataLoader")
    print("  • Model building and initialization")
    print("  • Training loop with validation")
    print("  • Model checkpointing and early stopping")
    print("  • Evaluation and visualization")
    print()
    
    # Show minimal training code snippet
    print("Minimal training code:")
    print("-" * 60)
    print("""
from capsnet_dorsal_vein_classification_pytorch import *
import torch

# 1. Prepare data
train_dataset = VeinDataset(train_files, transform=get_transforms(augment=True))
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# 2. Build model
model = CapsNet().to(DEVICE)
criterion = MarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3. Train
for epoch in range(50):
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 4. Save
torch.save(model.state_dict(), 'my_capsnet_pytorch.pth')
    """)
    print("-" * 60)


def quick_architecture_info():
    """
    Example: Display CapsNet architecture information
    """
    print("=" * 60)
    print("CAPSNET ARCHITECTURE INFO (PyTorch)")
    print("=" * 60)
    print()
    
    # Build model
    print("Building CapsNet model...")
    model = CapsNet(
        num_classes=NUM_CLASSES,
        input_channels=1,
        primary_caps_dim=PRIMARY_CAPS_DIM,
        digit_caps_dim=DIGIT_CAPS_DIM,
        num_routing=NUM_ROUTING
    )
    
    print("✅ Model built\n")
    
    # Display model structure
    print("Model Structure:")
    print("-" * 60)
    print(model)
    print("-" * 60)
    print()
    
    # Display layer information
    print("Layer Details:")
    print("-" * 60)
    for name, module in model.named_children():
        print(f"{name}:")
        print(f"  Type: {module.__class__.__name__}")
        
        # Count parameters in this layer
        params = sum(p.numel() for p in module.parameters())
        print(f"  Parameters: {params:,}")
        print()
    
    # Display total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print()
    
    # Compare with typical CNN
    print("Comparison with CNN:")
    print(f"  CapsNet parameters: {total_params:,}")
    print(f"  Typical CNN (similar depth): ~5-10M parameters")
    print(f"  CapsNet is more parameter-efficient!")


def quick_custom_prediction():
    """
    Example: Make predictions with custom preprocessing
    """
    print("=" * 60)
    print("CUSTOM PREDICTION EXAMPLE (PyTorch)")
    print("=" * 60)
    print()
    
    print("This example shows how to:")
    print("  • Load your own preprocessing")
    print("  • Make batch predictions")
    print("  • Extract capsule activations")
    print()
    
    # Load model
    model_path = 'best_capsnet_pytorch.pth'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first")
        return
    
    model = CapsNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    # Custom preprocessing function
    def my_preprocess(image_path):
        """Your custom preprocessing"""
        transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Add your custom preprocessing here
            # For example: histogram equalization, contrast adjustment, etc.
            transforms.ToTensor(),
        ])
        
        img = Image.open(image_path).convert('L')
        img_tensor = transform(img)
        return img_tensor
    
    # Example: Batch prediction
    image_paths = [
        'DorsalHandVeins_DB1_png/train/person_001_db1_L1.png',
        'DorsalHandVeins_DB1_png/train/person_002_db1_L1.png',
        'DorsalHandVeins_DB1_png/train/person_003_db1_L1.png',
    ]
    
    # Check if files exist
    existing_paths = [p for p in image_paths if os.path.exists(p)]
    
    if not existing_paths:
        print("❌ No example images found")
        return
    
    print(f"Processing {len(existing_paths)} images...")
    
    # Preprocess batch
    batch = []
    for path in existing_paths:
        img_tensor = my_preprocess(path)
        batch.append(img_tensor)
    
    batch = torch.stack(batch).to(DEVICE)
    
    # Batch prediction
    with torch.no_grad():
        predictions = model(batch)
    
    predictions = predictions.cpu().numpy()
    
    # Display results
    print("\nBatch Predictions:")
    print("-" * 60)
    for i, (path, pred) in enumerate(zip(existing_paths, predictions)):
        predicted_class = np.argmax(pred)
        confidence = pred[predicted_class]
        predicted_person = predicted_class + 1
        
        filename = os.path.basename(path)
        print(f"{i+1}. {filename}")
        print(f"   Predicted: Person {predicted_person:03d} (confidence: {confidence:.4f})")
        print()


def main():
    """
    Main menu for quick examples
    """
    print("\n" + "=" * 60)
    print(" CAPSNET QUICK START EXAMPLES (PyTorch)")
    print("=" * 60)
    print()
    print("Choose an example:")
    print("  1. Inference Example (load model and predict)")
    print("  2. Training Example (minimal training code)")
    print("  3. Architecture Info (model structure)")
    print("  4. Custom Prediction (batch processing)")
    print("  5. Run All Examples")
    print("  0. Exit")
    print()
    
    try:
        choice = input("Enter your choice (0-5): ").strip()
        print()
        
        if choice == '1':
            quick_inference_example()
        elif choice == '2':
            quick_training_example()
        elif choice == '3':
            quick_architecture_info()
        elif choice == '4':
            quick_custom_prediction()
        elif choice == '5':
            quick_training_example()
            print("\n" * 2)
            quick_architecture_info()
            print("\n" * 2)
            quick_inference_example()
            print("\n" * 2)
            quick_custom_prediction()
        elif choice == '0':
            print("Goodbye!")
            return
        else:
            print("Invalid choice. Please try again.")
            main()
            
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")
        return


if __name__ == '__main__':
    # If no arguments, show menu
    if len(sys.argv) == 1:
        main()
    else:
        # Run specific example from command line
        example = sys.argv[1].lower()
        
        if example in ['inference', 'infer', '1']:
            quick_inference_example()
        elif example in ['train', 'training', '2']:
            quick_training_example()
        elif example in ['arch', 'architecture', 'info', '3']:
            quick_architecture_info()
        elif example in ['custom', 'batch', '4']:
            quick_custom_prediction()
        elif example in ['all', '5']:
            quick_training_example()
            print("\n" * 2)
            quick_architecture_info()
            print("\n" * 2)
            quick_inference_example()
            print("\n" * 2)
            quick_custom_prediction()
        else:
            print(f"Unknown example: {example}")
            print("Usage: python quick_start_capsnet_pytorch.py [inference|train|arch|custom|all]")
