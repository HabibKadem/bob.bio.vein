# Working with Dorsal Hand Vein ROI in bob.bio.vein

This guide explains how to work with dorsal hand vein Region of Interest (ROI) data in the bob.bio.vein library.

## Overview

The bob.bio.vein library now supports dorsal hand vein recognition, which focuses on the vein patterns on the back (dorsal side) of the hand. This is in addition to the existing support for finger, palm, and wrist veins.

## Database Structure

The dorsal hand vein database expects images organized in the following structure:

```
DorsalHandVeins_DB1_png/
  train/
    person_001_db1_L1.png
    person_001_db1_L2.png
    person_001_db1_L3.png
    person_001_db1_L4.png
    person_002_db1_L1.png
    person_002_db1_L2.png
    ...
    person_138_db1_L4.png
```

Where:
- Images should be grayscale PNG files
- Naming convention: `person_XXX_db1_LY.png`
  - XXX: Person ID (e.g., 001, 002, ..., 138)
  - Y: Sample number (1, 2, 3, 4)

## Configuration

### 1. Set the Database Directory

Configure the path to your dorsal hand vein database:

```bash
bob config set bob.bio.vein.dorsalhandveins.directory /path/to/DorsalHandVeins_DB1_png/train
```

### 2. Set the ROI Annotations Directory (Optional)

If you have ROI annotations for the images:

```bash
bob config set bob.bio.vein.dorsalhandveins.roi /path/to/roi_annotations
```

ROI annotation files should be text files with the same base name as the image files but with a `.txt` extension. Each line should contain a point in `(y, x)` format that defines the ROI polygon.

### 3. Generate Protocol CSV File

Generate a CSV file that defines the protocol for your database:

```bash
bob_bio_vein_generate_dorsalhandveins_csv.py /path/to/DorsalHandVeins_DB1_png/train protocol.csv
```

Then set the CSV path:

```bash
bob config set bob.bio.vein.dorsalhandveins.csv /path/to/protocol.csv
```

## Using the Database

### In Python Code

```python
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase

# Create database instance
database = DorsalHandVeinsDatabase(protocol="default")

# Access samples
samples = database.background_model_samples()
```

### With ROI Annotations

The database automatically handles ROI annotations if configured. The ROI data will be available in the sample's metadata:

```python
from bob.bio.vein.preprocessor.mask import AnnotatedRoIMask

# Create a masker that uses annotated ROI
masker = AnnotatedRoIMask()

# Apply to a sample with ROI annotations
mask = masker(sample)
```

## ROI Annotation Format

ROI annotation files are text files with one point per line:

```
5 10
5 200
150 200
150 10
```

Each line represents a point `(y, x)` that forms a polygon outlining the region of interest in the image.

## Features

The dorsal hand vein support includes:

1. **Database Management**: `DorsalHandVeinsDatabase` class for managing dorsal hand vein datasets
2. **ROI Annotations**: Integration with existing ROI annotation infrastructure
3. **Masking**: Support for `AnnotatedRoIMask` to create masks from ROI polygons
4. **Preprocessing**: Access to all existing preprocessing tools (cropping, normalization, filtering)
5. **CSV Protocol Generation**: Utility script to auto-generate protocol files

## Example Workflow

1. Organize your dorsal hand vein images
2. Create ROI annotations (if available)
3. Generate protocol CSV
4. Configure bob with paths
5. Use the database in your recognition pipeline

```python
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
from bob.bio.vein.preprocessor.mask import AnnotatedRoIMask
from bob.bio.vein.preprocessor.crop import FixedCrop

# Initialize database
db = DorsalHandVeinsDatabase(protocol="default")

# Create preprocessing pipeline
cropper = FixedCrop(top=10, bottom=10, left=10, right=10)
masker = AnnotatedRoIMask()

# Process samples
for sample in db.background_model_samples():
    image = sample.data
    cropped = cropper(image)
    mask = masker(cropped)
    # Continue with feature extraction...
```

## Deep Learning Integration

For deep learning approaches (e.g., CNN+ViT models mentioned in the custom agent), you can:

1. Use the database to load and organize samples
2. Extract ROI regions using the masking functionality
3. Feed the preprocessed images to your neural network
4. Leverage PyTorch/TensorFlow data loaders with the database samples

## References

- Main library documentation: https://www.idiap.ch/software/bob/docs/bob/bob.bio.vein/master/index.html
- Bob framework: https://www.idiap.ch/software/bob
