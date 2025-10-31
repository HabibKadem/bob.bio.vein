"""
Configuration for CNN+ViT model on DorsalHandVeins dataset
"""

from sklearn.pipeline import make_pipeline
from bob.bio.vein.extractor.CNNViT import VeinCNNViTModel
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase

# Database
database = DorsalHandVeinsDatabase(protocol="train-test")

# Model parameters
model_params = {
    'num_classes': 138,
    'img_size': 224,
    'patch_size': 16,
    'embed_dim': 256,
    'num_heads': 8,
    'num_layers': 6,
    'dropout': 0.1,
}

# Create model instance
extractor = VeinCNNViTModel(**model_params)

# Pipeline configuration
pipeline = make_pipeline(extractor)
