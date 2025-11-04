"""
Quick Start Example: Using CapsNet for Vein Classification
============================================================

This is a minimal example showing how to use the CapsNet implementation.
Perfect for quick testing or integration into your own pipeline.
"""

import sys
import os

# Add the parent directory to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from the main capsnet module
from capsnet_dorsal_vein_classification import (
    build_capsnet,
    margin_loss,
    IMG_SIZE,
    NUM_CLASSES,
    PRIMARY_CAPS_DIM,
    DIGIT_CAPS_DIM,
    NUM_ROUTING
)

import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image


def quick_inference_example():
    """
    Example: Load a trained model and make a prediction
    """
    print("=" * 60)
    print("QUICK INFERENCE EXAMPLE")
    print("=" * 60)
    print()
    
    # Load the trained model
    model_path = 'best_capsnet_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Please train the model first by running:")
        print("   python capsnet_dorsal_vein_classification.py")
        return
    
    print(f"Loading model: {model_path}")
    model = keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded\n")
    
    # Load and preprocess an image
    image_path = 'DorsalHandVeins_DB1_png/train/person_001_db1_L1.png'
    
    if not os.path.exists(image_path):
        print(f"❌ Example image not found: {image_path}")
        print("   Please provide a valid image path")
        return
    
    print(f"Loading image: {image_path}")
    
    # Preprocess
    img = Image.open(image_path).convert('L')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=[0, -1])
    
    print(f"Image shape: {img_array.shape}\n")
    
    # Make prediction
    print("Making prediction...")
    predictions = model.predict(img_array, verbose=0)
    
    # Get top prediction
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_person = predicted_class + 1
    
    print(f"✅ Prediction complete")
    print(f"\nPredicted Person: {predicted_person:03d}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show top 5 predictions
    print("\nTop 5 Predictions:")
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    for i, idx in enumerate(top_5_idx, 1):
        person_id = idx + 1
        conf = predictions[0][idx]
        print(f"  {i}. Person {person_id:03d}: {conf:.4f}")


def quick_training_example():
    """
    Example: Train a CapsNet model with minimal code
    """
    print("=" * 60)
    print("QUICK TRAINING EXAMPLE")
    print("=" * 60)
    print()
    
    print("For a complete training example, run:")
    print("  python capsnet_dorsal_vein_classification.py")
    print()
    print("This script contains the full pipeline:")
    print("  • Data loading and augmentation")
    print("  • Model building and compilation")
    print("  • Training with callbacks")
    print("  • Evaluation and visualization")
    print()
    
    # Show minimal training code snippet
    print("Minimal training code:")
    print("-" * 60)
    print("""
from capsnet_dorsal_vein_classification import *

# 1. Prepare data
train_files = list_image_files("DorsalHandVeins_DB1_png/train/")
train_ds = make_dataset(train_files, batch_size=16, shuffle=True)

# 2. Build model
model = build_capsnet()
model.compile(optimizer='adam', loss=margin_loss, metrics=['accuracy'])

# 3. Train
history = model.fit(train_ds, epochs=50)

# 4. Save
model.save('my_capsnet_model.h5')
    """)
    print("-" * 60)


def quick_architecture_info():
    """
    Example: Display CapsNet architecture information
    """
    print("=" * 60)
    print("CAPSNET ARCHITECTURE INFO")
    print("=" * 60)
    print()
    
    # Build model
    print("Building CapsNet model...")
    model = build_capsnet(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=NUM_CLASSES,
        primary_caps_dim=PRIMARY_CAPS_DIM,
        digit_caps_dim=DIGIT_CAPS_DIM,
        num_routing=NUM_ROUTING
    )
    
    print("✅ Model built\n")
    
    # Display summary
    print("Model Summary:")
    print("-" * 60)
    model.summary()
    print("-" * 60)
    print()
    
    # Display layer information
    print("Layer Details:")
    print("-" * 60)
    for i, layer in enumerate(model.layers, 1):
        print(f"{i}. {layer.name}")
        print(f"   Type: {layer.__class__.__name__}")
        if hasattr(layer, 'output_shape'):
            print(f"   Output Shape: {layer.output_shape}")
        print()
    
    # Display parameters
    total_params = model.count_params()
    print(f"Total Parameters: {total_params:,}")
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
    print("CUSTOM PREDICTION EXAMPLE")
    print("=" * 60)
    print()
    
    print("This example shows how to:")
    print("  • Load your own preprocessing")
    print("  • Make batch predictions")
    print("  • Extract capsule activations")
    print()
    
    # Load model
    model_path = 'best_capsnet_model.h5'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("   Train the model first")
        return
    
    model = keras.models.load_model(model_path, compile=False)
    
    # Custom preprocessing function
    def my_preprocess(image_path):
        """Your custom preprocessing"""
        img = Image.open(image_path).convert('L')
        # Apply your custom preprocessing here
        # For example: histogram equalization, contrast adjustment, etc.
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    
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
        img_array = my_preprocess(path)
        batch.append(img_array)
    
    batch = np.array(batch)
    batch = np.expand_dims(batch, axis=-1)  # Add channel dimension
    
    # Batch prediction
    predictions = model.predict(batch, verbose=0)
    
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
    print(" CAPSNET QUICK START EXAMPLES")
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
            print("Usage: python quick_start_capsnet.py [inference|train|arch|custom|all]")
