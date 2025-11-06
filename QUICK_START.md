# Quick Start Guide: Dorsal Hand Vein ROI Support

## What Was Added

This implementation adds full support for working with dorsal hand vein (back of hand) Region of Interest (ROI) data in bob.bio.vein.

## Files Created

1. **bob/bio/vein/database/dorsalhandveins.py** - Main database class
2. **bob/bio/vein/config/database/dorsalhandveins_default.py** - Configuration
3. **bob/bio/vein/script/generate_dorsalhandveins_csv.py** - CSV generator utility
4. **doc/DORSAL_HAND_VEIN_ROI.md** - Comprehensive documentation
5. **examples/dorsalhandveins_example.py** - Working example
6. **IMPLEMENTATION_SUMMARY.md** - Technical details

## Quick Start

### Step 1: Organize Your Images

Ensure your dorsal hand vein images follow this structure:
```
DorsalHandVeins_DB1_png/
  train/
    person_001_db1_L1.png
    person_001_db1_L2.png
    person_001_db1_L3.png
    person_001_db1_L4.png
    person_002_db1_L1.png
    ...
```

### Step 2: Generate Protocol CSV

```bash
bob_bio_vein_generate_dorsalhandveins_csv.py \
    /path/to/DorsalHandVeins_DB1_png/train \
    protocol.csv
```

### Step 3: Configure bob.bio.vein

```bash
bob config set bob.bio.vein.dorsalhandveins.directory /path/to/DorsalHandVeins_DB1_png/train
bob config set bob.bio.vein.dorsalhandveins.csv protocol.csv
bob config set bob.bio.vein.dorsalhandveins.roi /path/to/roi_annotations  # Optional
```

### Step 4: Use in Your Code

```python
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
from bob.bio.vein.preprocessor.mask import AnnotatedRoIMask
from bob.bio.vein.preprocessor.crop import FixedCrop

# Initialize database
database = DorsalHandVeinsDatabase(protocol="default")

# Get samples
samples = database.background_model_samples()

# Process with ROI masking
cropper = FixedCrop(top=10, bottom=10, left=10, right=10)
masker = AnnotatedRoIMask()

for sample in samples:
    image = sample.data
    cropped = cropper(image)
    if hasattr(sample, 'metadata') and 'roi' in sample.metadata:
        mask = masker(cropped)
    # Continue with feature extraction...
```

## ROI Annotation Format

If you have ROI annotations, create text files with one point per line:
```
5 10
5 200
150 200
150 10
```

Each line is a `(y, x)` coordinate forming a polygon around the region of interest.

## Run the Example

```bash
python examples/dorsalhandveins_example.py
```

## Documentation

For complete details, see:
- **doc/DORSAL_HAND_VEIN_ROI.md** - Full documentation
- **IMPLEMENTATION_SUMMARY.md** - Technical implementation details
- **examples/dorsalhandveins_example.py** - Working code example

## Integration with Deep Learning

The database can be easily integrated with PyTorch or TensorFlow:

```python
from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
import torch
from torch.utils.data import Dataset

class DorsalHandVeinDataset(Dataset):
    def __init__(self, database, transform=None):
        self.database = database
        self.samples = database.background_model_samples()
        self.transform = transform
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = sample.data
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample.subject

# Use with DataLoader
database = DorsalHandVeinsDatabase(protocol="default")
dataset = DorsalHandVeinDataset(database)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
```

## Features

✅ ROI annotation support
✅ Preprocessing pipeline (crop, mask, normalize)
✅ CSV protocol generation
✅ Configuration management
✅ Compatible with bob.bio.base pipelines
✅ Easy integration with deep learning frameworks
✅ Comprehensive documentation
✅ Working examples

## Support

For questions or issues:
1. Check **doc/DORSAL_HAND_VEIN_ROI.md** for detailed documentation
2. Review **examples/dorsalhandveins_example.py** for working code
3. See **IMPLEMENTATION_SUMMARY.md** for technical details

## What's Next

You can now:
1. Use the database with your dorsal hand vein images
2. Apply ROI annotations if available
3. Build recognition pipelines with existing bob.bio.vein tools
4. Integrate with deep learning models (CNN, ViT, etc.)
5. Extend with additional protocols or features as needed
