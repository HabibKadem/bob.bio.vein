#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Example script demonstrating how to use the dorsal hand vein database with ROI

This example shows:
1. How to configure the database
2. How to load samples
3. How to work with ROI annotations
4. How to apply preprocessing
"""

import sys
from pathlib import Path


def example_basic_usage():
    """Example 1: Basic database usage"""
    print("=" * 70)
    print("Example 1: Basic Database Usage")
    print("=" * 70)
    
    from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
    
    try:
        database = DorsalHandVeinsDatabase(protocol="default")
        print(f"✓ Database created: {database.name}")
        print(f"  Protocol: {database.protocol}")
        print(f"  Available protocols: {database.protocols()}")
    except Exception as e:
        print(f"✗ Error creating database: {e}")
        print("  Make sure to configure the database paths first!")
        return False
    
    return True


def example_roi_annotation():
    """Example 2: Working with ROI annotations"""
    print("\n" + "=" * 70)
    print("Example 2: Working with ROI Annotations")
    print("=" * 70)
    
    try:
        from bob.bio.vein.database.roi_annotation import ROIAnnotation
        import numpy as np
        from bob.pipelines import DelayedSample
        
        mock_key = "person_001_db1_L1"
        mock_data = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        sample = DelayedSample(lambda: mock_data, key=mock_key)
        
        print(f"✓ Created mock sample with key: {mock_key}")
        print(f"  Data shape: {mock_data.shape}")
        
        roi_annotator = ROIAnnotation(roi_path=None)
        print("✓ ROI annotator created (no path configured)")
        print("  To use ROI annotations, configure:")
        print("  bob config set bob.bio.vein.dorsalhandveins.roi [ROI_PATH]")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def example_preprocessing():
    """Example 3: Preprocessing with ROI"""
    print("\n" + "=" * 70)
    print("Example 3: Preprocessing Pipeline")
    print("=" * 70)
    
    try:
        from bob.bio.vein.preprocessor.crop import FixedCrop
        from bob.bio.vein.preprocessor.mask import FixedMask
        import numpy as np
        
        sample_image = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
        print(f"✓ Created sample image with shape: {sample_image.shape}")
        
        cropper = FixedCrop(top=10, bottom=10, left=20, right=20)
        cropped = cropper(sample_image)
        print(f"✓ Applied cropping: {sample_image.shape} -> {cropped.shape}")
        
        masker = FixedMask(top=5, bottom=5, left=10, right=10)
        mask = masker(cropped)
        print(f"✓ Created mask with shape: {mask.shape}")
        print(f"  Mask dtype: {mask.dtype}, True pixels: {mask.sum()}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    return True


def main():
    """Main function to run all examples"""
    print("\n" + "=" * 70)
    print("DORSAL HAND VEIN ROI - Example Usage")
    print("=" * 70)
    print()
    print("This script demonstrates how to work with dorsal hand vein")
    print("ROI data in bob.bio.vein library.")
    print()
    
    results = []
    results.append(("Basic Usage", example_basic_usage()))
    results.append(("ROI Annotation", example_roi_annotation()))
    results.append(("Preprocessing", example_preprocessing()))
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {name}")
    
    print("\nFor more information, see:")
    print("  - doc/DORSAL_HAND_VEIN_ROI.md")
    print()


if __name__ == "__main__":
    main()
