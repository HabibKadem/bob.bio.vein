#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Dorsal Hand Veins database implementation for ROI-based recognition
"""

from pathlib import Path
from sklearn.pipeline import make_pipeline

import bob.io.base

from bob.bio.base.database import CSVDataset, CSVToSampleLoaderBiometrics
from bob.bio.vein.database.roi_annotation import ROIAnnotation
from bob.extension import rc


class DorsalHandVeinsDatabase(CSVDataset):
    """
    Dorsal Hand Veins Database for biometric recognition

    This database is designed for dorsal hand vein recognition experiments.
    It supports ROI (Region of Interest) annotations for the dorsal hand vein images.

    The database structure expects images organized as:
        DorsalHandVeins_DB1_png/
          train/
            person_001_db1_L1.png
            person_001_db1_L2.png
            person_001_db1_L3.png
            person_001_db1_L4.png
            person_002_db1_L1.png
            ...
            person_138_db1_L4.png

    Images are expected to be grayscale PNG files.

    **Configuration**

    To use this dataset, you need to configure the database directory:

        .. code-block:: sh

            bob config set bob.bio.vein.dorsalhandveins.directory [DATABASE PATH]

    **ROI Annotations**

    If you have ROI annotations for the dorsal hand vein images, you can configure
    the ROI annotation path:

        .. code-block:: sh

            bob config set bob.bio.vein.dorsalhandveins.roi [ANNOTATION PATH]

    ROI annotation files should be text files with one annotation per line in the
    format ``(y, x)``, respecting Bob's image encoding convention. The
    interconnection of these points in a polygon forms the RoI.

    **Protocols**

    This database supports the following protocols:

     * default: Uses all available samples for development/evaluation

    """

    def __init__(self, protocol="default", csv_file_name=None):
        """
        Initialize the Dorsal Hand Veins Database

        Parameters:
            protocol (str): The protocol to use (default: "default")
            csv_file_name (str): Path to a CSV file defining the protocol.
                                If None, will attempt to auto-discover the database structure.
        """
        # Get the database directory from configuration
        database_dir = rc.get("bob.bio.vein.dorsalhandveins.directory", "")
        roi_path = rc.get("bob.bio.vein.dorsalhandveins.roi", "")

        # If a CSV file is provided, use it; otherwise, use dataset_original_directory
        if csv_file_name:
            dataset_protocol_path = csv_file_name
        else:
            # For now, we'll require a CSV file or the user to provide one
            # This is a placeholder - in a real implementation, you might
            # auto-generate the CSV from the directory structure
            dataset_protocol_path = rc.get(
                "bob.bio.vein.dorsalhandveins.csv", ""
            )

        super().__init__(
            name="dorsalhandveins",
            dataset_protocol_path=dataset_protocol_path,
            protocol=protocol,
            csv_to_sample_loader=make_pipeline(
                CSVToSampleLoaderBiometrics(
                    data_loader=bob.io.base.load,
                    dataset_original_directory=database_dir,
                    extension=".png",
                    reference_id_equal_subject_id=True,
                ),
                ROIAnnotation(roi_path=roi_path if roi_path else None),
            ),
            score_all_vs_all=True,
        )

    @staticmethod
    def protocols():
        """Return list of available protocols"""
        return [
            "default",
        ]
