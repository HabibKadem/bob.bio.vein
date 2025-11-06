# Dorsal Hand Vein ROI Support - Implementation Summary

## Overview

This implementation adds comprehensive support for dorsal hand vein Region of Interest (ROI) recognition to the bob.bio.vein library. Dorsal hand veins refer to the vein patterns visible on the back (dorsal side) of the hand.

## Changes Made

### 1. Core Database Support

**File: `bob/bio/vein/database/dorsalhandveins.py`**
- New `DorsalHandVeinsDatabase` class that extends `CSVDataset`
- Supports configurable database directory, ROI annotations, and protocol files
- Integrates with existing ROI annotation infrastructure
- Compatible with bob.bio.base pipeline system

**File: `bob/bio/vein/config/database/dorsalhandveins_default.py`**
- Configuration file for the default protocol
- Instantiates `DorsalHandVeinsDatabase` with "default" protocol

### 2. Utility Scripts

**File: `bob/bio/vein/script/generate_dorsalhandveins_csv.py`**
- Command-line utility to generate CSV protocol files from image directories
- Parses dorsal hand vein image naming conventions (person_XXX_db1_LY.png)
- Automatically extracts person IDs and sample numbers
- Registered as console script: `bob_bio_vein_generate_dorsalhandveins_csv.py`

### 3. Documentation

**File: `doc/DORSAL_HAND_VEIN_ROI.md`**
- Comprehensive guide for working with dorsal hand vein ROI data
- Database structure explanation
- Configuration instructions
- ROI annotation format specification
- Complete workflow examples
- Deep learning integration notes

**File: `examples/dorsalhandveins_example.py`**
- Executable example script demonstrating:
  - Basic database usage
  - ROI annotation handling
  - Preprocessing pipeline
- Educational tool for users

**File: `examples/README.md`**
- Quick reference for examples directory

### 4. Testing

**File: `bob/bio/vein/tests/test_databases.py`**
- Added `test_dorsalhandveins_basic()` function
- Tests database instantiation
- Verifies protocol structure
- Ensures proper attribute initialization

### 5. Configuration Updates

**File: `setup.py`**
- Added entry points for dorsal hand veins database:
  - `bob.bio.config`: `dorsalhandveins_default`
  - `bob.bio.database`: `dorsalhandveins`
  - Console script: `bob_bio_vein_generate_dorsalhandveins_csv.py`

**File: `README.rst`**
- Updated description to mention dorsal hand vein support
- Changed from "finger, palm and wrist" to "finger, palm, wrist, and dorsal hand"

**File: `.gitignore`**
- Added `__pycache__/` to prevent committing Python cache directories

## Expected Database Structure

The implementation expects dorsal hand vein images organized as:

```
DorsalHandVeins_DB1_png/
  train/
    person_001_db1_L1.png
    person_001_db1_L2.png
    person_001_db1_L3.png
    person_001_db1_L4.png
    person_002_db1_L1.png
    ...
    person_138_db1_L4.png
```

## Configuration Commands

Users configure the database with:

```bash
# Set database directory
bob config set bob.bio.vein.dorsalhandveins.directory /path/to/DorsalHandVeins_DB1_png/train

# Generate CSV protocol
bob_bio_vein_generate_dorsalhandveins_csv.py /path/to/DorsalHandVeins_DB1_png/train protocol.csv

# Set CSV protocol path
bob config set bob.bio.vein.dorsalhandveins.csv protocol.csv

# Set ROI annotations (optional)
bob config set bob.bio.vein.dorsalhandveins.roi /path/to/roi_annotations
```

## Usage Example

```python
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
from bob.bio.vein.preprocessor.mask import AnnotatedRoIMask

# Initialize database
database = DorsalHandVeinsDatabase(protocol="default")

# Load samples
samples = database.background_model_samples()

# Process with ROI masking
masker = AnnotatedRoIMask()
for sample in samples:
    image = sample.data
    if hasattr(sample, 'metadata') and 'roi' in sample.metadata:
        mask = masker(image)
        # Continue processing...
```

## Integration with Existing Infrastructure

The implementation leverages existing bob.bio.vein components:
- `ROIAnnotation` transformer for reading ROI annotation files
- `AnnotatedRoIMask` for creating masks from ROI polygons
- `FixedCrop` for preprocessing
- All existing preprocessing filters and normalization tools

## Compatibility

- Fully compatible with existing bob.bio.vein infrastructure
- Works with bob.bio.base pipeline system
- Supports sklearn pipeline integration
- Compatible with deep learning frameworks (PyTorch, TensorFlow) through sample loading

## Benefits

1. **Minimal Changes**: Surgical additions that don't modify existing functionality
2. **Consistency**: Follows patterns established by existing databases (utfvp, verafinger_contactless)
3. **Extensibility**: Easy to add new protocols or database variants
4. **Documentation**: Comprehensive guides and examples for users
5. **Testing**: Basic tests ensure functionality works as expected

## Files Modified/Created

### Created (7 files):
1. `bob/bio/vein/database/dorsalhandveins.py`
2. `bob/bio/vein/config/database/dorsalhandveins_default.py`
3. `bob/bio/vein/script/generate_dorsalhandveins_csv.py`
4. `doc/DORSAL_HAND_VEIN_ROI.md`
5. `examples/dorsalhandveins_example.py`
6. `examples/README.md`

### Modified (3 files):
1. `setup.py` - Added entry points
2. `README.rst` - Updated description
3. `.gitignore` - Added __pycache__
4. `bob/bio/vein/tests/test_databases.py` - Added test

## Future Enhancements

Possible future improvements:
1. Additional protocols (train/test splits, cross-validation)
2. Pre-trained deep learning models for dorsal hand vein recognition
3. Automatic ROI detection algorithms specific to dorsal hand veins
4. Performance benchmarks and baseline results
5. Dataset quality metrics and validation tools
