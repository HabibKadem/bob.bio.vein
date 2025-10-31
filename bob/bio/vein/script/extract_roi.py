#!/usr/bin/env python
# vim: set fileencoding=utf-8 :

"""Extract and save ROI (Region of Interest) regions from dorsal hand vein images

This script reads dorsal hand vein images along with their ROI annotation files,
extracts the ROI regions, and saves them to a specified output directory.

ROI annotation files should contain (y, x) coordinate pairs, one per line,
forming a polygon that defines the region of interest.

Usage: %(prog)s [-v...] [-m] <input-dir> <roi-dir> <output-dir>
       %(prog)s --help
       %(prog)s --version


Arguments:
  <input-dir>   Directory containing the dorsal hand vein images
  <roi-dir>     Directory containing the ROI annotation files (.txt)
  <output-dir>  Directory where extracted ROI images will be saved


Options:
  -h, --help       Shows this help message and exits
  -V, --version    Prints the version and exits
  -v, --verbose    Increases the output verbosity level
  -m, --mask-only  Save only the mask images instead of masked images


Examples:

  1. Extract ROI regions from dorsal hand vein images:

     $ %(prog)s -vv DorsalHandVeins_DB1_png/train roi_annotations/train output/roi_extracted

  2. Extract only mask images:

     $ %(prog)s -vv -m DorsalHandVeins_DB1_png/train roi_annotations/train output/masks

"""

import os
import sys
import glob
from pathlib import Path

import numpy as np

import bob.extension.log

logger = bob.extension.log.setup("bob.bio.vein")

import bob.io.base

from ..preprocessor import utils


def load_roi_points(roi_file):
    """Load ROI points from annotation file
    
    Parameters
    ----------
    roi_file : str
        Path to ROI annotation file containing (y, x) coordinates
    
    Returns
    -------
    numpy.ndarray
        Array of (y, x) coordinate pairs
    """
    try:
        points = np.loadtxt(roi_file, dtype='uint16')
        if points.ndim == 1:
            points = points.reshape(1, -1)
        return points
    except Exception as e:
        logger.error(f"Failed to load ROI file {roi_file}: {e}")
        return None


def extract_roi_from_image(image, roi_points):
    """Extract ROI region from image using polygon coordinates
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (2D grayscale)
    roi_points : numpy.ndarray
        Array of (y, x) coordinate pairs forming a polygon
    
    Returns
    -------
    numpy.ndarray
        Extracted ROI image with background masked to 0
    numpy.ndarray
        Binary mask of the ROI region
    """
    # Create mask from ROI polygon
    mask = utils.poly_to_mask(image.shape, roi_points)
    
    # Apply mask to image
    masked_image = image.copy()
    masked_image[~mask] = 0
    
    return masked_image, mask


def crop_to_roi_bounds(image, mask):
    """Crop image to the bounding box of the ROI
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    mask : numpy.ndarray
        Binary mask of the ROI
    
    Returns
    -------
    numpy.ndarray
        Cropped image
    """
    # Find bounding box of ROI
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        logger.warning("Empty mask detected, returning original image")
        return image
    
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    
    # Crop image to bounding box
    cropped = image[ymin:ymax+1, xmin:xmax+1]
    
    return cropped


def process_images(input_dir, roi_dir, output_dir, mask_only=False):
    """Process all images and extract ROI regions
    
    Parameters
    ----------
    input_dir : str
        Directory containing input images
    roi_dir : str
        Directory containing ROI annotation files
    output_dir : str
        Directory where output images will be saved
    mask_only : bool
        If True, save only mask images instead of masked images
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all image files
    image_patterns = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
    image_files = []
    for pattern in image_patterns:
        image_files.extend(glob.glob(os.path.join(input_dir, pattern)))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} images to process")
    
    processed_count = 0
    skipped_count = 0
    
    for image_path in sorted(image_files):
        filename = os.path.basename(image_path)
        filestem = os.path.splitext(filename)[0]
        
        # Construct ROI annotation file path
        roi_file = os.path.join(roi_dir, filestem + '.txt')
        
        if not os.path.exists(roi_file):
            logger.warning(f"ROI file not found for {filename}, skipping")
            skipped_count += 1
            continue
        
        try:
            # Load image
            image = bob.io.base.load(image_path)
            
            # Ensure grayscale
            if image.ndim == 3:
                # Convert to grayscale using weighted formula
                image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Load ROI points
            roi_points = load_roi_points(roi_file)
            if roi_points is None:
                skipped_count += 1
                continue
            
            # Extract ROI
            masked_image, mask = extract_roi_from_image(image, roi_points)
            
            # Crop to ROI bounds
            if mask_only:
                output_image = crop_to_roi_bounds(
                    utils.mask_to_image(mask, dtype=np.uint8), 
                    mask
                )
            else:
                output_image = crop_to_roi_bounds(masked_image, mask)
            
            # Save output image
            output_path = os.path.join(output_dir, filename)
            bob.io.base.save(output_image, output_path)
            
            logger.debug(f"Processed {filename} -> {output_path}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            skipped_count += 1
            continue
    
    logger.info(f"Processing complete: {processed_count} images processed, {skipped_count} skipped")


def main(user_input=None):
    """Main entry point"""
    
    if user_input is not None:
        argv = user_input
    else:
        argv = sys.argv[1:]
    
    import docopt
    import pkg_resources
    
    completions = dict(
        prog=os.path.basename(sys.argv[0]),
        version=pkg_resources.require("bob.bio.vein")[0].version,
    )
    
    args = docopt.docopt(
        __doc__ % completions,
        argv=argv,
        version=completions["version"],
    )
    
    # Set up logging
    verbosity = int(args["--verbose"])
    bob.extension.log.set_verbosity_level(logger, verbosity)
    
    input_dir = args["<input-dir>"]
    roi_dir = args["<roi-dir>"]
    output_dir = args["<output-dir>"]
    mask_only = args["--mask-only"]
    
    # Validate input directories
    if not os.path.isdir(input_dir):
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    if not os.path.isdir(roi_dir):
        logger.error(f"ROI directory does not exist: {roi_dir}")
        return 1
    
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"ROI directory: {roi_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Mode: {'Mask only' if mask_only else 'Masked images'}")
    
    # Process images
    process_images(input_dir, roi_dir, output_dir, mask_only)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
