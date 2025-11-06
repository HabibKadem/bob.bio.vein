#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""
Utility script to generate CSV protocol file for dorsal hand veins database

This script scans a directory containing dorsal hand vein images and generates
a CSV file that can be used with the DorsalHandVeinsDatabase class.

Usage:
    python generate_dorsalhandveins_csv.py <image_directory> <output_csv>

Example:
    python generate_dorsalhandveins_csv.py /path/to/DorsalHandVeins_DB1_png/train/ protocol.csv
"""

import os
import sys
from pathlib import Path


def generate_csv_from_directory(image_dir, output_csv, image_extension=".png"):
    """
    Generate a CSV protocol file from a directory of dorsal hand vein images.
    
    The expected naming convention for images is:
        person_XXX_db1_LY.png
    where:
        XXX is the person ID (e.g., 001, 002, ..., 138)
        Y is the sample number (e.g., 1, 2, 3, 4)
    
    Parameters:
        image_dir (str): Path to directory containing images
        output_csv (str): Path to output CSV file
        image_extension (str): Image file extension (default: ".png")
    """
    image_dir = Path(image_dir)
    
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        sys.exit(1)
    
    # Collect all image files
    image_files = sorted(image_dir.glob(f"*{image_extension}"))
    
    if not image_files:
        print(f"Error: No {image_extension} files found in {image_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    # Generate CSV content
    # CSV format: filename,subject_id,sample_id,session_id
    csv_lines = ["filename,subject_id,sample_id,session_id"]
    
    for img_file in image_files:
        filename = img_file.stem  # Remove extension
        
        # Parse filename: person_XXX_db1_LY
        # Extract person ID and sample ID
        parts = filename.split("_")
        if len(parts) >= 4 and parts[0] == "person":
            person_id = parts[1]  # e.g., "001"
            sample_id_raw = parts[3].replace("L", "")  # e.g., "1" from "L1"
            
            # Validate extracted components
            if not person_id.isdigit():
                print(
                    f"Warning: Invalid person ID '{person_id}' in file: {filename}"
                )
                print("  Expected format: person_XXX_db1_LY.png where XXX is numeric")
                continue
            
            if not sample_id_raw.isdigit():
                print(
                    f"Warning: Invalid sample ID '{sample_id_raw}' in file: {filename}"
                )
                print("  Expected format: person_XXX_db1_LY.png where Y is numeric")
                continue
            
            # For simplicity, we'll use sample_id as session_id
            # and treat each image as a separate enrollment
            csv_lines.append(f"{filename}{image_extension},{person_id},{sample_id_raw},1")
        else:
            print(f"Warning: Skipping file with unexpected format: {filename}")
            print("  Expected format: person_XXX_db1_LY.png")
            print("    where XXX is person ID (e.g., 001, 002)")
            print("    and Y is sample number (e.g., 1, 2, 3, 4)")
    
    # Write CSV file
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("\n".join(csv_lines))
    
    print(f"Generated CSV file: {output_csv}")
    print(f"Total entries: {len(csv_lines) - 1}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    
    image_dir = sys.argv[1]
    output_csv = sys.argv[2]
    
    generate_csv_from_directory(image_dir, output_csv)


if __name__ == "__main__":
    main()
