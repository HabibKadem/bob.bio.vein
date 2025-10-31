#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
DorsalHandVeins database implementation for biometric recognition
"""

import os
import glob
from pathlib import Path

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.extension import rc
import bob.io.base


class DorsalHandVeinsDatabase(CSVDataset):
    """
    DorsalHandVeins Database for person recognition using dorsal hand vein patterns.
    
    The database consists of grayscale images from 138 subjects, with 4 images per subject.
    Images are named following the pattern: person_XXX_db1_LY.png where:
    - XXX is the person ID (001-138)
    - Y is the image number (1-4)
    
    The dataset structure is:
        DorsalHandVeins_DB1_png/
            train/
                person_001_db1_L1.png
                person_001_db1_L2.png
                person_001_db1_L3.png
                person_001_db1_L4.png
                ...
                person_138_db1_L1.png
                person_138_db1_L2.png
                person_138_db1_L3.png
                person_138_db1_L4.png
    
    Configuration:
        Set the database path using:
        bob config set bob.bio.vein.dorsalhandveins.directory [DATABASE PATH]
    
    Protocols:
        - 'train-test': Uses first 3 images for enrollment and last image for probing
        - 'cross-validation': Uses all combinations for cross-validation
    """
    
    def __init__(self, protocol='train-test'):
        """
        Initialize the DorsalHandVeins database
        
        Parameters
        ----------
        protocol : str
            Protocol to use ('train-test' or 'cross-validation')
        """
        self.protocol_name = protocol
        
        # Get the database directory from configuration
        self.base_dir = rc.get('bob.bio.vein.dorsalhandveins.directory', '')
        
        if not self.base_dir or not os.path.exists(self.base_dir):
            raise ValueError(
                f"Database directory not found: {self.base_dir}. "
                "Please set it using: bob config set bob.bio.vein.dorsalhandveins.directory [PATH]"
            )
        
        # Create CSV protocol files if they don't exist
        self._create_protocol_files()
        
        # Initialize parent class - we'll use a simple file-based loader for now
        # since we're creating CSV files on the fly
        super().__init__(
            name='dorsalhandveins',
            dataset_protocol_path=self._get_protocol_path(),
            protocol=protocol,
            csv_to_sample_loader=CSVToSampleLoaderBiometrics(
                data_loader=bob.io.base.load,
                dataset_original_directory=self.base_dir,
                extension='',
                reference_id_equal_subject_id=True,
            ),
            score_all_vs_all=True,
        )
    
    def _get_protocol_path(self):
        """Get the path to the protocol CSV files"""
        protocol_dir = os.path.join(
            os.path.dirname(__file__), 
            'protocols', 
            'dorsalhandveins'
        )
        os.makedirs(protocol_dir, exist_ok=True)
        return os.path.join(protocol_dir, f'{self.protocol_name}.csv')
    
    def _create_protocol_files(self):
        """Create CSV protocol files for the database"""
        train_dir = os.path.join(self.base_dir, 'train')
        
        if not os.path.exists(train_dir):
            raise ValueError(f"Train directory not found: {train_dir}")
        
        # Get all image files
        image_files = sorted(glob.glob(os.path.join(train_dir, 'person_*_db1_L*.png')))
        
        if not image_files:
            raise ValueError(f"No images found in {train_dir}")
        
        # Parse image files and organize by subject
        subjects = {}
        for img_path in image_files:
            filename = os.path.basename(img_path)
            # Parse: person_XXX_db1_LY.png
            parts = filename.replace('.png', '').split('_')
            if len(parts) >= 4:
                person_id = parts[1]  # XXX
                image_num = parts[3][1]  # Y from LY
                
                if person_id not in subjects:
                    subjects[person_id] = []
                subjects[person_id].append({
                    'path': os.path.join('train', filename),
                    'image_num': int(image_num),
                    'full_path': img_path
                })
        
        # Sort images for each subject by image number
        for person_id in subjects:
            subjects[person_id].sort(key=lambda x: x['image_num'])
        
        # Create protocol CSV file
        protocol_path = self._get_protocol_path()
        
        with open(protocol_path, 'w') as f:
            # Write header
            f.write('filename,subject_id,reference_id,group,purpose\n')
            
            # Split subjects: 70% train, 15% dev, 15% eval
            subject_ids = sorted(subjects.keys())
            n_subjects = len(subject_ids)
            n_train = int(n_subjects * 0.7)
            n_dev = int(n_subjects * 0.15)
            
            train_subjects = subject_ids[:n_train]
            dev_subjects = subject_ids[n_train:n_train + n_dev]
            eval_subjects = subject_ids[n_train + n_dev:]
            
            # Write entries based on protocol
            if self.protocol_name == 'train-test':
                # Training set: all images
                for person_id in train_subjects:
                    for img_info in subjects[person_id]:
                        f.write(f"{img_info['path']},{person_id},{person_id},train,train\n")
                
                # Dev set: first 3 images for enrollment, last for probing
                for person_id in dev_subjects:
                    images = subjects[person_id]
                    if len(images) >= 4:
                        # First 3 for enrollment
                        for img_info in images[:3]:
                            f.write(f"{img_info['path']},{person_id},{person_id},dev,enroll\n")
                        # Last for probing
                        f.write(f"{images[3]['path']},{person_id},{person_id},dev,probe\n")
                    else:
                        # If less than 4 images, use all but last for enrollment
                        for img_info in images[:-1]:
                            f.write(f"{img_info['path']},{person_id},{person_id},dev,enroll\n")
                        f.write(f"{images[-1]['path']},{person_id},{person_id},dev,probe\n")
                
                # Eval set: same as dev set
                for person_id in eval_subjects:
                    images = subjects[person_id]
                    if len(images) >= 4:
                        for img_info in images[:3]:
                            f.write(f"{img_info['path']},{person_id},{person_id},eval,enroll\n")
                        f.write(f"{images[3]['path']},{person_id},{person_id},eval,probe\n")
                    else:
                        for img_info in images[:-1]:
                            f.write(f"{img_info['path']},{person_id},{person_id},eval,enroll\n")
                        f.write(f"{images[-1]['path']},{person_id},{person_id},eval,probe\n")
            
            elif self.protocol_name == 'cross-validation':
                # All images can be used for both enrollment and probing
                for person_id in train_subjects:
                    for img_info in subjects[person_id]:
                        f.write(f"{img_info['path']},{person_id},{person_id},train,train\n")
                
                for person_id in dev_subjects:
                    for img_info in subjects[person_id]:
                        f.write(f"{img_info['path']},{person_id},{person_id},dev,enroll\n")
                        f.write(f"{img_info['path']},{person_id},{person_id},dev,probe\n")
                
                for person_id in eval_subjects:
                    for img_info in subjects[person_id]:
                        f.write(f"{img_info['path']},{person_id},{person_id},eval,enroll\n")
                        f.write(f"{img_info['path']},{person_id},{person_id},eval,probe\n")
    
    @staticmethod
    def protocols():
        """Return available protocols"""
        return ['train-test', 'cross-validation']
