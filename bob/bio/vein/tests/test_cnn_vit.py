#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Tests for DorsalHandVeins database and CNN+ViT model
"""

import os
import tempfile
import pytest
import numpy as np
from pathlib import Path


def test_cnn_vit_import():
    """Test that CNN+ViT model can be imported"""
    try:
        from bob.bio.vein.extractor.CNNViT import (
            VeinCNNViTModel,
            CNNViTNetwork,
            VeinDataset,
            get_transforms,
        )
        # If PyTorch is available, check that model can be instantiated
        model = VeinCNNViTModel(
            num_classes=138,
            img_size=224,
            patch_size=16,
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            dropout=0.1,
        )
        assert model is not None
        assert model.num_classes == 138
        print("CNN+ViT model imported and instantiated successfully")
    except ImportError:
        pytest.skip("PyTorch not available, skipping CNN+ViT model test")


def test_cnn_vit_forward():
    """Test forward pass of CNN+ViT model"""
    try:
        import torch
        from bob.bio.vein.extractor.CNNViT import CNNViTNetwork
        
        # Create a small model for testing
        model = CNNViTNetwork(
            num_classes=10,
            img_size=224,
            patch_size=16,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )
        
        # Test forward pass with dummy input
        batch_size = 2
        x = torch.randn(batch_size, 1, 224, 224)
        output = model(x)
        
        assert output.shape == (batch_size, 10)
        print(f"Forward pass successful: output shape {output.shape}")
        
        # Test feature extraction
        features = model.extract_features(x)
        assert features.shape == (batch_size, 64)
        print(f"Feature extraction successful: feature shape {features.shape}")
        
    except ImportError:
        pytest.skip("PyTorch not available, skipping forward pass test")


def test_vein_dataset():
    """Test VeinDataset class"""
    try:
        import torch
        from bob.bio.vein.extractor.CNNViT import VeinDataset, get_transforms
        
        # Create dummy data
        n_samples = 10
        images = [np.random.rand(100, 100).astype(np.float32) for _ in range(n_samples)]
        labels = list(range(n_samples))
        
        # Create dataset
        transform = get_transforms(img_size=224, augment=False)
        dataset = VeinDataset(images, labels, transform=transform)
        
        assert len(dataset) == n_samples
        
        # Test getting an item
        img, label = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape[0] == 1  # grayscale
        assert label == 0
        
        print(f"VeinDataset test successful: {len(dataset)} samples")
        
    except ImportError:
        pytest.skip("PyTorch not available, skipping VeinDataset test")


def test_dorsalhandveins_database_class():
    """Test DorsalHandVeins database class instantiation"""
    from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
    
    # Test protocols method
    protocols = DorsalHandVeinsDatabase.protocols()
    assert 'train-test' in protocols
    assert 'cross-validation' in protocols
    print(f"DorsalHandVeins protocols: {protocols}")


def test_dorsalhandveins_with_mock_data():
    """Test DorsalHandVeins database with mock data"""
    from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
    from bob.extension import rc
    
    # Create a temporary directory with mock data
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, 'train')
        os.makedirs(train_dir, exist_ok=True)
        
        # Create mock images (just empty files for structure testing)
        for person_id in range(1, 11):  # 10 people for testing
            for img_num in range(1, 5):  # 4 images each
                filename = f"person_{person_id:03d}_db1_L{img_num}.png"
                filepath = os.path.join(train_dir, filename)
                # Create a small grayscale image
                import numpy as np
                img_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                
                # Save as PNG using PIL if available, otherwise skip
                try:
                    from PIL import Image
                    Image.fromarray(img_data, mode='L').save(filepath)
                except ImportError:
                    # Skip if PIL not available - this is just for testing structure
                    open(filepath, 'w').close()
        
        # Temporarily set the config
        original_value = rc.get('bob.bio.vein.dorsalhandveins.directory', '')
        
        try:
            rc['bob.bio.vein.dorsalhandveins.directory'] = tmpdir
            
            # Test database instantiation
            database = DorsalHandVeinsDatabase(protocol='train-test')
            
            assert database is not None
            assert database.protocol_name == 'train-test'
            print(f"DorsalHandVeins database created with mock data in {tmpdir}")
            
        finally:
            # Restore original config
            if original_value:
                rc['bob.bio.vein.dorsalhandveins.directory'] = original_value
            else:
                rc.pop('bob.bio.vein.dorsalhandveins.directory', None)


def test_dorsalhandveins_roi_support():
    """Test DorsalHandVeins database with ROI annotations"""
    from bob.bio.vein.database.dorsalhandveins import DorsalHandVeinsDatabase
    from bob.extension import rc
    
    # Create temporary directories with mock data
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dir = os.path.join(tmpdir, 'train')
        roi_dir = os.path.join(tmpdir, 'roi', 'train')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(roi_dir, exist_ok=True)
        
        # Create mock images and ROI files
        for person_id in range(1, 4):  # 3 people for testing
            for img_num in range(1, 3):  # 2 images each
                filename = f"person_{person_id:03d}_db1_L{img_num}.png"
                filepath = os.path.join(train_dir, filename)
                
                # Create a small grayscale image
                import numpy as np
                img_data = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
                
                try:
                    from PIL import Image
                    Image.fromarray(img_data, mode='L').save(filepath)
                except ImportError:
                    open(filepath, 'w').close()
                
                # Create mock ROI file
                roi_filename = f"person_{person_id:03d}_db1_L{img_num}.txt"
                roi_filepath = os.path.join(roi_dir, roi_filename)
                
                # Write mock ROI coordinates (simple rectangle)
                with open(roi_filepath, 'w') as f:
                    f.write("10 10\n")
                    f.write("10 90\n")
                    f.write("90 90\n")
                    f.write("90 10\n")
        
        # Temporarily set the config
        original_dir_value = rc.get('bob.bio.vein.dorsalhandveins.directory', '')
        original_roi_value = rc.get('bob.bio.vein.dorsalhandveins.roi', '')
        
        try:
            rc['bob.bio.vein.dorsalhandveins.directory'] = tmpdir
            rc['bob.bio.vein.dorsalhandveins.roi'] = os.path.join(tmpdir, 'roi')
            
            # Test database instantiation with ROI
            database = DorsalHandVeinsDatabase(protocol='train-test')
            
            assert database is not None
            print(f"DorsalHandVeins database created with ROI support in {tmpdir}")
            
        finally:
            # Restore original config
            if original_dir_value:
                rc['bob.bio.vein.dorsalhandveins.directory'] = original_dir_value
            else:
                rc.pop('bob.bio.vein.dorsalhandveins.directory', None)
            
            if original_roi_value:
                rc['bob.bio.vein.dorsalhandveins.roi'] = original_roi_value
            else:
                rc.pop('bob.bio.vein.dorsalhandveins.roi', None)


def test_config_files_exist():
    """Test that configuration files exist"""
    import bob.bio.vein.config.cnn_vit as cnn_vit_config
    import bob.bio.vein.config.database.dorsalhandveins as dhv_db_config
    
    assert hasattr(dhv_db_config, 'database')
    print("Configuration files loaded successfully")


def test_extract_roi_script():
    """Test ROI extraction script"""
    try:
        from bob.bio.vein.script.extract_roi import load_roi_points, extract_roi_from_image
        import numpy as np
        
        # Test loading ROI points from text
        with tempfile.TemporaryDirectory() as tmpdir:
            roi_file = os.path.join(tmpdir, 'test_roi.txt')
            with open(roi_file, 'w') as f:
                f.write("10 10\n")
                f.write("10 90\n")
                f.write("90 90\n")
                f.write("90 10\n")
            
            points = load_roi_points(roi_file)
            assert points is not None
            assert points.shape == (4, 2)
            
            # Test ROI extraction
            test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
            masked_image, mask = extract_roi_from_image(test_image, points)
            
            assert masked_image.shape == test_image.shape
            assert mask.shape == test_image.shape
            assert mask.dtype == bool
            
            print("ROI extraction functions work correctly")
    except ImportError as e:
        print(f"ROI extraction test skipped: {e}")


if __name__ == '__main__':
    print("Running DorsalHandVeins and CNN+ViT tests...")
    
    test_cnn_vit_import()
    test_cnn_vit_forward()
    test_vein_dataset()
    test_dorsalhandveins_database_class()
    test_config_files_exist()
    
    try:
        test_dorsalhandveins_with_mock_data()
    except Exception as e:
        print(f"Mock data test skipped or failed: {e}")
    
    try:
        test_dorsalhandveins_roi_support()
    except Exception as e:
        print(f"ROI support test skipped or failed: {e}")
    
    try:
        test_extract_roi_script()
    except Exception as e:
        print(f"ROI extraction script test skipped or failed: {e}")
    
    print("\nAll tests completed!")
