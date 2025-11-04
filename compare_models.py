"""
Comparison Script: CNN vs CapsNet for Dorsal Hand Veins Classification
=======================================================================

This script demonstrates how to use both the CNN baseline and the CapsNet model
for vein classification, and compare their predictions.

Usage:
    python compare_models.py --image path/to/test/image.png
"""

import os
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def load_and_preprocess_image(image_path, img_size=224):
    """
    Load and preprocess an image for model inference.
    
    Args:
        image_path: Path to the input image
        img_size: Target size for resizing
        
    Returns:
        Preprocessed image array ready for prediction
    """
    # Load image in grayscale
    img = Image.open(image_path).convert('L')
    
    # Resize to target size
    img = img.resize((img_size, img_size))
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Add batch and channel dimensions: [1, height, width, 1]
    img_array = np.expand_dims(img_array, axis=[0, -1])
    
    return img_array, img


def predict_with_cnn(model, image_array):
    """
    Make prediction using CNN model.
    
    Args:
        model: Loaded CNN model
        image_array: Preprocessed image array
        
    Returns:
        predicted_person: Predicted person ID (1-276)
        confidence: Confidence score [0, 1]
        all_probs: Probability distribution over all classes
    """
    # Get predictions (softmax probabilities)
    predictions = model.predict(image_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    
    # Convert to person ID (add 1 since labels are zero-based)
    predicted_person = predicted_class + 1
    
    return predicted_person, confidence, predictions[0]


def predict_with_capsnet(model, image_array):
    """
    Make prediction using CapsNet model.
    
    Args:
        model: Loaded CapsNet model
        image_array: Preprocessed image array
        
    Returns:
        predicted_person: Predicted person ID (1-276)
        confidence: Confidence score (capsule length)
        all_lengths: Capsule lengths for all classes
    """
    # Get predictions (capsule lengths)
    capsule_lengths = model.predict(image_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_class = np.argmax(capsule_lengths[0])
    confidence = capsule_lengths[0][predicted_class]
    
    # Convert to person ID (add 1 since labels are zero-based)
    predicted_person = predicted_class + 1
    
    return predicted_person, confidence, capsule_lengths[0]


def visualize_predictions(image, cnn_results, capsnet_results, top_k=5):
    """
    Visualize and compare predictions from both models.
    
    Args:
        image: Original PIL image
        cnn_results: Tuple of (predicted_person, confidence, all_probs)
        capsnet_results: Tuple of (predicted_person, confidence, all_lengths)
        top_k: Number of top predictions to show
    """
    # Unpack results
    cnn_person, cnn_conf, cnn_probs = cnn_results
    caps_person, caps_conf, caps_lengths = capsnet_results
    
    # Get top K predictions for each model
    cnn_top_k = np.argsort(cnn_probs)[-top_k:][::-1]
    caps_top_k = np.argsort(caps_lengths)[-top_k:][::-1]
    
    # Create figure
    fig = plt.figure(figsize=(16, 6))
    
    # ========================================================================
    # Subplot 1: Original Image
    # ========================================================================
    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Input Vein Image', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    # ========================================================================
    # Subplot 2: CNN Predictions
    # ========================================================================
    ax2 = plt.subplot(1, 3, 2)
    
    # Prepare data for bar plot
    cnn_persons = [f"P{idx+1:03d}" for idx in cnn_top_k]
    cnn_confidences = [cnn_probs[idx] for idx in cnn_top_k]
    
    # Create bar plot
    bars = ax2.barh(range(top_k), cnn_confidences, color='skyblue')
    
    # Highlight the top prediction
    bars[0].set_color('steelblue')
    
    # Labels and formatting
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels(cnn_persons)
    ax2.set_xlabel('Confidence (Probability)', fontsize=11)
    ax2.set_title(f'CNN Prediction\nTop: Person {cnn_person:03d} ({cnn_conf:.3f})', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, conf) in enumerate(zip(bars, cnn_confidences)):
        ax2.text(conf + 0.02, i, f'{conf:.3f}', 
                va='center', fontsize=9)
    
    # ========================================================================
    # Subplot 3: CapsNet Predictions
    # ========================================================================
    ax3 = plt.subplot(1, 3, 3)
    
    # Prepare data for bar plot
    caps_persons = [f"P{idx+1:03d}" for idx in caps_top_k]
    caps_confidences = [caps_lengths[idx] for idx in caps_top_k]
    
    # Create bar plot
    bars = ax3.barh(range(top_k), caps_confidences, color='lightcoral')
    
    # Highlight the top prediction
    bars[0].set_color('crimson')
    
    # Labels and formatting
    ax3.set_yticks(range(top_k))
    ax3.set_yticklabels(caps_persons)
    ax3.set_xlabel('Confidence (Capsule Length)', fontsize=11)
    ax3.set_title(f'CapsNet Prediction\nTop: Person {caps_person:03d} ({caps_conf:.3f})', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlim([0, 1])
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, conf) in enumerate(zip(bars, caps_confidences)):
        ax3.text(conf + 0.02, i, f'{conf:.3f}', 
                va='center', fontsize=9)
    
    # ========================================================================
    # Final layout
    # ========================================================================
    plt.tight_layout()
    
    # Save figure
    output_path = 'model_comparison_prediction.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved: {output_path}")
    
    # Display
    plt.show()


def print_comparison_table(cnn_results, capsnet_results):
    """
    Print a formatted comparison table of the two models' predictions.
    
    Args:
        cnn_results: Tuple of (predicted_person, confidence, all_probs)
        capsnet_results: Tuple of (predicted_person, confidence, all_lengths)
    """
    cnn_person, cnn_conf, _ = cnn_results
    caps_person, caps_conf, _ = capsnet_results
    
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON RESULTS")
    print("=" * 70)
    print()
    print(f"{'Model':<20} {'Predicted Person':<20} {'Confidence':<15} {'Match':<10}")
    print("-" * 70)
    print(f"{'CNN':<20} {f'Person {cnn_person:03d}':<20} {cnn_conf:<15.4f} {'-':<10}")
    print(f"{'CapsNet':<20} {f'Person {caps_person:03d}':<20} {caps_conf:<15.4f} {'-':<10}")
    print("-" * 70)
    
    # Check if predictions match
    if cnn_person == caps_person:
        print("\n✅ Both models agree on the prediction!")
        print(f"   Predicted Person: {cnn_person:03d}")
    else:
        print("\n⚠️  Models disagree on the prediction!")
        print(f"   CNN predicts: Person {cnn_person:03d} (confidence: {cnn_conf:.4f})")
        print(f"   CapsNet predicts: Person {caps_person:03d} (confidence: {caps_conf:.4f})")
    
    print("\n" + "=" * 70)


def main():
    """
    Main function to compare CNN and CapsNet predictions.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Compare CNN and CapsNet predictions for vein classification'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to the input vein image'
    )
    parser.add_argument(
        '--cnn-model',
        type=str,
        default='best_model_tf.h5',
        help='Path to the CNN model file (default: best_model_tf.h5)'
    )
    parser.add_argument(
        '--capsnet-model',
        type=str,
        default='best_capsnet_model.h5',
        help='Path to the CapsNet model file (default: best_capsnet_model.h5)'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (default: 5)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print(" CNN vs CAPSNET COMPARISON FOR VEIN CLASSIFICATION")
    print("=" * 70)
    print()
    
    # ========================================================================
    # 1. Check if files exist
    # ========================================================================
    if not os.path.exists(args.image):
        print(f"❌ Error: Image file not found: {args.image}")
        return
    
    if not os.path.exists(args.cnn_model):
        print(f"❌ Error: CNN model not found: {args.cnn_model}")
        print("   Please train the CNN model first or specify correct path with --cnn-model")
        return
    
    if not os.path.exists(args.capsnet_model):
        print(f"❌ Error: CapsNet model not found: {args.capsnet_model}")
        print("   Please train the CapsNet model first or specify correct path with --capsnet-model")
        return
    
    # ========================================================================
    # 2. Load models
    # ========================================================================
    print("Loading models...")
    
    try:
        # Load CNN model
        print(f"  Loading CNN model from: {args.cnn_model}")
        cnn_model = keras.models.load_model(args.cnn_model)
        print("  ✅ CNN model loaded successfully")
        
        # Load CapsNet model with custom objects
        print(f"  Loading CapsNet model from: {args.capsnet_model}")
        # Note: margin_loss needs to be defined if model was saved with it
        capsnet_model = keras.models.load_model(
            args.capsnet_model,
            compile=False  # Don't need to compile for inference
        )
        print("  ✅ CapsNet model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return
    
    print()
    
    # ========================================================================
    # 3. Load and preprocess image
    # ========================================================================
    print(f"Loading image: {args.image}")
    
    try:
        image_array, original_image = load_and_preprocess_image(args.image)
        print(f"  Image shape: {image_array.shape}")
        print("  ✅ Image loaded and preprocessed")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    print()
    
    # ========================================================================
    # 4. Make predictions
    # ========================================================================
    print("Making predictions...")
    
    try:
        # CNN prediction
        print("  Running CNN inference...")
        cnn_results = predict_with_cnn(cnn_model, image_array)
        print("  ✅ CNN prediction complete")
        
        # CapsNet prediction
        print("  Running CapsNet inference...")
        capsnet_results = predict_with_capsnet(capsnet_model, image_array)
        print("  ✅ CapsNet prediction complete")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return
    
    # ========================================================================
    # 5. Display results
    # ========================================================================
    print_comparison_table(cnn_results, capsnet_results)
    
    # ========================================================================
    # 6. Visualize results
    # ========================================================================
    print("\nGenerating visualization...")
    try:
        visualize_predictions(
            original_image,
            cnn_results,
            capsnet_results,
            top_k=args.top_k
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not generate visualization: {e}")
    
    print("\n" + "=" * 70)
    print(" COMPARISON COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
