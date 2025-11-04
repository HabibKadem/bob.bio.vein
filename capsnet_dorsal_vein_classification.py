"""
Capsule Neural Network for Dorsal Hand Veins Classification
============================================================

This module implements a Capsule Neural Network (CapsNet) for classifying
dorsal hand vein images. The architecture is designed to be efficient and
easy to understand, with detailed comments throughout.

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
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

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
gpus = tf.config.list_physical_devices('GPU')
DEVICE = "GPU" if len(gpus) > 0 else "CPU"
print(f"Using device: {DEVICE}")
if DEVICE == "GPU":
    print(f"Number of GPUs available: {len(gpus)}")


# ============================================================================
# 2. DATASET UTILITIES
# ============================================================================

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


def parse_label_from_filename(path):
    """
    Extract person ID from filename pattern: person_XXX_db1_LY.png
    
    Args:
        path: Full path to image file
        
    Returns:
        Zero-based integer label (person_id - 1)
        
    Example:
        person_001_db1_L1.png -> label 0
        person_276_db1_L4.png -> label 275
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


# ============================================================================
# 3. TENSORFLOW DATA PIPELINE
# ============================================================================

def load_and_preprocess_image(path):
    """
    Load and preprocess a single image for CapsNet.
    
    Args:
        path: TensorFlow string tensor with image path
        
    Returns:
        Preprocessed image tensor of shape [IMG_SIZE, IMG_SIZE, 1]
        Normalized to [0, 1] range
    """
    # Read image file
    image = tf.io.read_file(path)
    
    # Decode image (grayscale, 1 channel)
    image = tf.image.decode_image(image, channels=1, expand_animations=False)
    
    # Convert to float32 and normalize to [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)
    
    # Resize to target size
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    
    # Set explicit shape for CapsNet
    image.set_shape([IMG_SIZE, IMG_SIZE, 1])
    
    return image


def path_to_label_tf(path):
    """
    TensorFlow function to extract label from filename.
    
    Args:
        path: TensorFlow string tensor with full file path
        
    Returns:
        Zero-based integer label
    """
    # Extract filename from path
    filename = tf.strings.split(path, os.sep)[-1]
    
    # Split by underscore
    parts = tf.strings.split(filename, '_')
    
    # Extract person ID token and remove extension
    person_token = tf.strings.regex_replace(parts[1], r'\..*$', '')
    
    # Convert to integer and make zero-based
    label = tf.strings.to_number(person_token, out_type=tf.int32) - 1
    
    return label


def make_dataset(file_list, batch_size=BATCH_SIZE, shuffle=True, augment_fn=None):
    """
    Create TensorFlow dataset from file list.
    
    Args:
        file_list: List of image file paths
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        augment_fn: Optional augmentation function
        
    Returns:
        tf.data.Dataset ready for training/validation
    """
    # Create dataset from file paths
    ds = tf.data.Dataset.from_tensor_slices(file_list)
    
    # Shuffle if requested
    if shuffle:
        ds = ds.shuffle(buffer_size=len(file_list), reshuffle_each_iteration=True)
    
    # Load and preprocess images
    ds = ds.map(
        lambda p: (load_and_preprocess_image(p), path_to_label_tf(p)),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Apply augmentation if provided
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch for efficiency
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds


def augment_image(image, label):
    """
    Apply data augmentation to training images.
    Light augmentation is sufficient for CapsNet due to its equivariance properties.
    
    Args:
        image: Input image tensor
        label: Corresponding label
        
    Returns:
        Augmented image and original label
    """
    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    
    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    
    # Small random rotation (helps with vein pattern variations)
    # Note: tf.image doesn't have built-in rotation, so we use a simple approach
    # For production, consider using tf.keras.layers.RandomRotation
    
    # Clip values to ensure they stay in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image, label


# ============================================================================
# 4. CAPSULE NETWORK LAYERS
# ============================================================================

def squash(vectors, axis=-1):
    """
    Squash function: non-linear activation for capsules.
    Ensures capsule output has length between 0 and 1.
    
    Formula:
        squash(s) = ||s||^2 / (1 + ||s||^2) * s / ||s||
    
    Args:
        vectors: Capsule vectors to squash
        axis: Axis along which to compute the norm
        
    Returns:
        Squashed vectors with same shape as input
    """
    # Compute squared norm
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    
    # Compute scaling factor
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + keras.backend.epsilon())
    
    # Apply squashing
    return scale * vectors


class PrimaryCapsule(layers.Layer):
    """
    Primary Capsule Layer: Converts conv features to capsule vectors.
    
    This layer:
    1. Applies convolution to extract features
    2. Reshapes output into capsule format
    3. Applies squash activation
    
    Args:
        num_capsules: Number of primary capsules
        capsule_dim: Dimension of each capsule vector
        kernel_size: Size of convolutional kernel
        strides: Stride for convolution
        padding: Padding type ('valid' or 'same')
    """
    
    def __init__(self, num_capsules, capsule_dim, kernel_size, strides=1, padding='valid', **kwargs):
        super(PrimaryCapsule, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        
        # Convolution layer to generate capsule components
        # Output channels = num_capsules * capsule_dim
        self.conv = layers.Conv2D(
            filters=num_capsules * capsule_dim,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=None
        )
    
    def call(self, inputs):
        """
        Forward pass through primary capsule layer.
        
        Args:
            inputs: Feature maps from previous layer [batch, height, width, channels]
            
        Returns:
            Primary capsules [batch, num_capsules, capsule_dim]
        """
        # Apply convolution
        outputs = self.conv(inputs)  # [batch, h, w, num_capsules * capsule_dim]
        
        # Get output spatial dimensions
        batch_size = tf.shape(outputs)[0]
        h, w = outputs.shape[1], outputs.shape[2]
        
        # Reshape to capsule format
        # [batch, h, w, num_capsules, capsule_dim]
        outputs = tf.reshape(outputs, [batch_size, h, w, self.num_capsules, self.capsule_dim])
        
        # Flatten spatial dimensions
        # [batch, h * w * num_capsules, capsule_dim]
        outputs = tf.reshape(outputs, [batch_size, h * w * self.num_capsules, self.capsule_dim])
        
        # Apply squash activation
        return squash(outputs)
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(PrimaryCapsule, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'capsule_dim': self.capsule_dim,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


class DigitCapsule(layers.Layer):
    """
    Digit Capsule Layer: High-level capsules for classification.
    
    Implements dynamic routing between capsules:
    1. Transform lower capsules via learned weight matrices
    2. Route information through iterative agreement process
    3. Output one capsule per class
    
    Args:
        num_capsules: Number of digit capsules (= num_classes)
        capsule_dim: Dimension of each digit capsule vector
        num_routing: Number of routing iterations
    """
    
    def __init__(self, num_capsules, capsule_dim, num_routing=3, **kwargs):
        super(DigitCapsule, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.num_routing = num_routing
    
    def build(self, input_shape):
        """
        Build layer weights.
        
        Args:
            input_shape: [batch_size, num_primary_capsules, primary_capsule_dim]
        """
        # Number of input capsules
        self.num_input_capsules = input_shape[1]
        self.input_capsule_dim = input_shape[2]
        
        # Weight matrix W: transforms primary capsules to digit capsule space
        # Shape: [num_input_capsules, num_capsules, input_capsule_dim, capsule_dim]
        self.W = self.add_weight(
            shape=[self.num_input_capsules, self.num_capsules, 
                   self.input_capsule_dim, self.capsule_dim],
            initializer='glorot_uniform',
            trainable=True,
            name='transformation_weights'
        )
        
        super(DigitCapsule, self).build(input_shape)
    
    def call(self, inputs):
        """
        Forward pass with dynamic routing.
        
        Args:
            inputs: Primary capsules [batch, num_primary_capsules, primary_capsule_dim]
            
        Returns:
            Digit capsules [batch, num_capsules, capsule_dim]
        """
        batch_size = tf.shape(inputs)[0]
        
        # Expand dimensions for broadcasting
        # [batch, num_input_capsules, 1, input_capsule_dim, 1]
        inputs_expand = tf.expand_dims(tf.expand_dims(inputs, axis=2), axis=-1)
        
        # Tile for each output capsule
        # [batch, num_input_capsules, num_capsules, input_capsule_dim, 1]
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1, 1])
        
        # Apply transformation: u_hat = W * u
        # [batch, num_input_capsules, num_capsules, capsule_dim]
        u_hat = tf.reduce_sum(
            inputs_tiled * self.W,  # Element-wise multiplication
            axis=3  # Sum over input capsule dimension
        )
        
        # Dynamic Routing Algorithm
        # Initialize routing logits b to zero
        # [batch, num_input_capsules, num_capsules]
        b = tf.zeros([batch_size, self.num_input_capsules, self.num_capsules])
        
        # Routing iterations
        for i in range(self.num_routing):
            # Compute coupling coefficients via softmax
            # c_ij represents how much capsule i should send to capsule j
            c = tf.nn.softmax(b, axis=2)  # [batch, num_input_capsules, num_capsules]
            
            # Expand for broadcasting
            c_expand = tf.expand_dims(c, axis=-1)  # [batch, num_input_capsules, num_capsules, 1]
            
            # Weighted sum of predictions: s = sum(c_ij * u_hat_j|i)
            s = tf.reduce_sum(c_expand * u_hat, axis=1)  # [batch, num_capsules, capsule_dim]
            
            # Apply squash activation to get output capsules
            v = squash(s)  # [batch, num_capsules, capsule_dim]
            
            # Update routing logits (except on last iteration)
            if i < self.num_routing - 1:
                # Expand v for agreement calculation
                v_expand = tf.expand_dims(v, axis=1)  # [batch, 1, num_capsules, capsule_dim]
                
                # Agreement: dot product between u_hat and v
                # Higher agreement increases routing coefficient
                agreement = tf.reduce_sum(u_hat * v_expand, axis=3)  # [batch, num_input_capsules, num_capsules]
                
                # Update routing logits
                b = b + agreement
        
        return v
    
    def get_config(self):
        """Return layer configuration for serialization."""
        config = super(DigitCapsule, self).get_config()
        config.update({
            'num_capsules': self.num_capsules,
            'capsule_dim': self.capsule_dim,
            'num_routing': self.num_routing
        })
        return config


def capsule_length(capsules):
    """
    Compute length (norm) of capsule vectors.
    Used to convert capsules to class probabilities.
    
    Args:
        capsules: Capsule vectors [batch, num_capsules, capsule_dim]
        
    Returns:
        Capsule lengths [batch, num_capsules]
    """
    return tf.sqrt(tf.reduce_sum(tf.square(capsules), axis=-1) + keras.backend.epsilon())


# ============================================================================
# 5. CAPSULE NETWORK MODEL
# ============================================================================

def build_capsnet(input_shape=(IMG_SIZE, IMG_SIZE, 1), 
                  num_classes=NUM_CLASSES,
                  primary_caps_dim=PRIMARY_CAPS_DIM,
                  digit_caps_dim=DIGIT_CAPS_DIM,
                  num_routing=NUM_ROUTING):
    """
    Build complete Capsule Network for vein classification.
    
    Architecture:
        1. Initial Conv Layer: Extract low-level features (edges, textures)
        2. Primary Capsules: Convert features to capsule vectors
        3. Digit Capsules: One capsule per person class
        4. Length Layer: Convert capsule lengths to probabilities
    
    Args:
        input_shape: Shape of input images
        num_classes: Number of person classes
        primary_caps_dim: Dimension of primary capsule vectors
        digit_caps_dim: Dimension of digit capsule vectors
        num_routing: Number of routing iterations
        
    Returns:
        Keras Model ready for training
    """
    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # ========================================================================
    # Layer 1: Initial Convolution
    # Extract basic features from the vein image
    # ========================================================================
    x = layers.Conv2D(
        filters=256,           # Number of feature maps
        kernel_size=9,         # Large kernel to capture vein patterns
        strides=1,
        padding='valid',
        activation='relu',
        name='initial_conv'
    )(inputs)
    # Output: [batch, 216, 216, 256] for 224x224 input
    
    # ========================================================================
    # Layer 2: Primary Capsules
    # Convert conv features to capsule representation
    # ========================================================================
    # Use 32 capsule types, each with 8D vectors
    primary_capsules = PrimaryCapsule(
        num_capsules=32,
        capsule_dim=primary_caps_dim,
        kernel_size=9,
        strides=2,
        padding='valid',
        name='primary_capsules'
    )(x)
    # Output: [batch, num_primary_caps, 8]
    # num_primary_caps = 104 * 104 * 32 = 345,088 for our dimensions
    
    # ========================================================================
    # Layer 3: Digit Capsules
    # One capsule per person class with dynamic routing
    # ========================================================================
    digit_capsules = DigitCapsule(
        num_capsules=num_classes,
        capsule_dim=digit_caps_dim,
        num_routing=num_routing,
        name='digit_capsules'
    )(primary_capsules)
    # Output: [batch, num_classes, digit_caps_dim]
    
    # ========================================================================
    # Output Layer: Capsule Lengths as Probabilities
    # Length of capsule vector represents existence probability
    # ========================================================================
    out_caps = layers.Lambda(
        capsule_length,
        name='capsule_length'
    )(digit_capsules)
    # Output: [batch, num_classes]
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=out_caps, name='CapsNet')
    
    return model


# ============================================================================
# 6. MARGIN LOSS FOR CAPSNET
# ============================================================================

def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lambda_=0.5):
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
        y_true: True labels [batch, num_classes] (one-hot encoded)
        y_pred: Predicted capsule lengths [batch, num_classes]
        m_plus: Margin for present class
        m_minus: Margin for absent class
        lambda_: Down-weighting factor
        
    Returns:
        Margin loss (scalar)
    """
    # Convert labels to one-hot if needed
    if len(y_true.shape) == 1:
        y_true = tf.one_hot(y_true, depth=NUM_CLASSES)
    
    # Loss for present classes
    # Penalize if capsule length < m_plus
    present_loss = y_true * tf.square(tf.maximum(0., m_plus - y_pred))
    
    # Loss for absent classes
    # Penalize if capsule length > m_minus
    absent_loss = lambda_ * (1 - y_true) * tf.square(tf.maximum(0., y_pred - m_minus))
    
    # Total loss: sum over all classes, mean over batch
    return tf.reduce_mean(tf.reduce_sum(present_loss + absent_loss, axis=1))


# ============================================================================
# 7. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """
    Main function to train the CapsNet model.
    """
    print("=" * 80)
    print("CAPSULE NETWORK FOR DORSAL HAND VEINS CLASSIFICATION")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Step 1: Load and prepare dataset
    # ========================================================================
    print("Step 1: Loading dataset...")
    
    # Get all image files
    all_files = list_image_files(DATASET_DIR)
    print(f"Found {len(all_files)} images")
    
    # Optionally infer number of classes
    def infer_num_classes(files):
        labels = [parse_label_from_filename(p) for p in files]
        return max(labels) + 1
    
    actual_num_classes = infer_num_classes(all_files)
    print(f"Detected {actual_num_classes} unique persons")
    
    if actual_num_classes != NUM_CLASSES:
        print(f"WARNING: NUM_CLASSES ({NUM_CLASSES}) doesn't match detected classes ({actual_num_classes})")
        print("Update NUM_CLASSES variable if needed")
    
    # Split into train and validation sets (70-30 split with stratification)
    labels = [parse_label_from_filename(p) for p in all_files]
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
    # Step 2: Create data pipelines
    # ========================================================================
    print("Step 2: Creating data pipelines...")
    
    # Create training dataset with augmentation
    train_ds = make_dataset(
        train_files,
        batch_size=BATCH_SIZE,
        shuffle=True,
        augment_fn=augment_image  # Enable augmentation for training
    )
    
    # Create validation dataset without augmentation
    val_ds = make_dataset(
        val_files,
        batch_size=BATCH_SIZE,
        shuffle=False,
        augment_fn=None  # No augmentation for validation
    )
    
    print("Data pipelines created successfully")
    print()
    
    # ========================================================================
    # Step 3: Build CapsNet model
    # ========================================================================
    print("Step 3: Building Capsule Network...")
    
    model = build_capsnet(
        input_shape=(IMG_SIZE, IMG_SIZE, 1),
        num_classes=NUM_CLASSES,
        primary_caps_dim=PRIMARY_CAPS_DIM,
        digit_caps_dim=DIGIT_CAPS_DIM,
        num_routing=NUM_ROUTING
    )
    
    # Display model architecture
    model.summary()
    print()
    
    # ========================================================================
    # Step 4: Compile model
    # ========================================================================
    print("Step 4: Compiling model...")
    
    # Use custom margin loss for CapsNet
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=margin_loss,
        metrics=['accuracy']
    )
    
    print("Model compiled successfully")
    print()
    
    # ========================================================================
    # Step 5: Setup callbacks
    # ========================================================================
    print("Step 5: Setting up training callbacks...")
    
    # ModelCheckpoint: Save best model based on validation loss
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath='best_capsnet_model.h5',
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    # EarlyStopping: Stop training if validation loss doesn't improve
    earlystop_cb = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,  # More patience for CapsNet
        restore_best_weights=True,
        mode='min',
        verbose=1
    )
    
    # ReduceLROnPlateau: Reduce learning rate when validation loss plateaus
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode='min',
        verbose=1
    )
    
    # TensorBoard: Visualize training metrics (optional)
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir='./logs_capsnet',
        histogram_freq=1
    )
    
    callbacks = [checkpoint_cb, earlystop_cb, reduce_lr_cb, tensorboard_cb]
    
    print("Callbacks configured:")
    print("  - ModelCheckpoint: saves best model")
    print("  - EarlyStopping: stops if no improvement")
    print("  - ReduceLROnPlateau: adjusts learning rate")
    print("  - TensorBoard: logs for visualization")
    print()
    
    # ========================================================================
    # Step 6: Train the model
    # ========================================================================
    print("Step 6: Training CapsNet...")
    print("=" * 80)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    print()
    print("=" * 80)
    print("Training completed!")
    print()
    
    # ========================================================================
    # Step 7: Save final model
    # ========================================================================
    print("Step 7: Saving final model...")
    
    model.save('capsnet_final_model.h5')
    print("✅ Final model saved: capsnet_final_model.h5")
    print("✅ Best model saved: best_capsnet_model.h5")
    print()
    
    # ========================================================================
    # Step 8: Plot training history
    # ========================================================================
    print("Step 8: Plotting training metrics...")
    
    # Extract metrics from history
    train_loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    train_acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(train_loss, label='Training Loss', linewidth=2)
    ax1.plot(val_loss, label='Validation Loss', linewidth=2)
    ax1.set_title('CapsNet Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy
    ax2.plot(train_acc, label='Training Accuracy', linewidth=2)
    ax2.plot(val_acc, label='Validation Accuracy', linewidth=2)
    ax2.set_title('CapsNet Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('capsnet_training_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Training metrics plot saved: capsnet_training_metrics.png")
    
    # Display plot
    plt.show()
    
    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  - Final training accuracy: {train_acc[-1]:.4f}")
    print(f"  - Final validation accuracy: {val_acc[-1]:.4f}")
    print(f"  - Best validation loss: {min(val_loss):.4f}")
    print()
    print("Next steps:")
    print("  1. Load the best model: model = keras.models.load_model('best_capsnet_model.h5')")
    print("  2. Evaluate on test set if available")
    print("  3. Use TensorBoard to visualize: tensorboard --logdir=./logs_capsnet")
    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    main()
