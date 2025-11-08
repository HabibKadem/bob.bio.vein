# Hand Geometry Biometric Feature Extraction

## Overview

This module implements hand/palm geometry biometric feature extraction, transcoded from the MATLAB implementation `pfehdm180.m`. The system extracts geometric features from hand images including finger positions, widths, lengths, and palm dimensions.

## Status: 🚧 In Progress

The hand geometry extractor is currently under development. The MATLAB source code has been analyzed and a Python module structure has been created.

## MATLAB Source Analysis

The original MATLAB file `pfehdm180.m` is a comprehensive hand biometrics system with approximately 2000+ lines of code organized into multiple functions:

### Main Function
- **pfehdm180**: Main entry point for hand geometry feature extraction

### Core Processing Functions
1. **SEGHAND180**: Hand segmentation from background
   - K-means clustering in L*a*b color space
   - Binary mask creation
   - Morphological operations

2. **Unhand18**: Finger point detection
   - Convex hull analysis
   - Detection of concave regions (finger valleys - dmin)
   - Detection of convex regions (finger tips/peaks - dmax)
   - Sorting and organizing detected points

3. **dist18**: Distance and orientation computation
   - Calculates distances between key points
   - Computes orientation angles
   - Prepares data for normalization

4. **orimhd18**: Orientation normalization
   - Rotates hand to standard orientation
   - Applies Discrete Curve Evolution (DCE)
   - Extracts skeleton using morphological operations
   - Identifies reference points

5. **MMpmain18**: Geometric measurement extraction
   - Measures finger lengths and widths
   - Calculates palm dimensions
   - Extracts contour features
   - Organizes all measurements into output structure

### Helper Functions
- **GetContour**: Extracts boundary points from binary image
- **evolution**: Discrete Curve Evolution for polygon simplification
- **FindConvex**: Identifies convex and concave points
- **brlinexya**: Bresenham line drawing algorithm
- **rtim**: Point rotation around a center
- **mussk18**: Skeleton extraction and DCE application
- **cadrg/cadrg2**: Image framing and border handling
- **rsz**: Image resizing
- **xylim**: Extract image limits
- **skex**: Skeleton extraction
- **MMexthd18/MMtdt18**: Finger extremity detection
- **MMdsline**: Draw line and find intersections
- **MMgmh18**: Geometric measurements for hand

## Python Implementation Structure

```python
from bob.bio.vein.extractor import HandGeometry

# Create extractor
extractor = HandGeometry(
    resolution=1000,  # Target image resolution
    border=10,        # Border padding
    n_colors=2        # Number of color clusters for segmentation
)

# Extract features (when implemented)
# result = extractor(hand_image)
# coord = result['coord']     # Geometric measurements
# ddist = result['ddist']     # Distance/orientation info
# sgh = result['sgh']         # Segmented hand image
```

## Implementation Roadmap

### Phase 1: Core Segmentation (Priority: High)
- [ ] Implement `segment_hand` (SEGHAND180)
  - RGB to L*a*b conversion
  - K-means clustering
  - Binary mask creation
  - Morphological operations
- [ ] Create unit tests for segmentation

### Phase 2: Feature Detection (Priority: High)
- [ ] Implement `detect_finger_points` (Unhand18)
  - Convex hull computation
  - Valley point detection (dmin)
  - Peak point detection (dmax)
  - Point sorting and organization
- [ ] Create unit tests for feature detection

### Phase 3: Distance and Orientation (Priority: Medium)
- [ ] Implement `compute_distances` (dist18)
  - Distance calculations
  - Orientation computation
- [ ] Implement `normalize_orientation` (orimhd18)
  - Image rotation
  - Skeleton extraction
  - Reference point identification

### Phase 4: Geometric Measurements (Priority: Medium)
- [ ] Implement `extract_measurements` (MMpmain18)
  - Finger length/width measurements
  - Palm dimension calculations
  - Feature organization

### Phase 5: Helper Functions (Priority: Low)
- [x] `bresenham_line` - Basic implementation complete
- [x] `rotate_point` - Basic implementation complete
- [ ] `get_contour` - Boundary extraction
- [ ] `evolution` - Discrete Curve Evolution
- [ ] `find_convex` - Convex/concave point detection
- [ ] Additional helper functions as needed

### Phase 6: Testing and Validation (Priority: High)
- [ ] Create comprehensive test suite
- [ ] Validate against MATLAB reference outputs
- [ ] Performance optimization
- [ ] Documentation completion

## Technical Challenges

1. **K-means in L*a*b Color Space**: Need to use `skimage.color.rgb2lab` for color space conversion
2. **Discrete Curve Evolution (DCE)**: Complex algorithm for polygon simplification
3. **Multiple Coordinate Systems**: Careful handling of MATLAB 1-based vs Python 0-based indexing
4. **Morphological Operations**: Translation from MATLAB's `bwmorph` to scikit-image equivalents
5. **Large Code Base**: ~2000 lines requires careful modular design

## Dependencies

- NumPy: Array operations and numerical computing
- SciPy: Signal processing and optimization
- scikit-image: Image processing operations
- PIL/Pillow: Image I/O and basic operations
- bob.io.base: Bob framework integration

## Contributing

The hand geometry extractor is a large transcoding project. Contributions are welcome! Please focus on:
1. Implementing one phase at a time
2. Adding comprehensive tests for each component
3. Documenting the mapping between MATLAB and Python code
4. Validating outputs against MATLAB references

## References

- Original MATLAB implementation: `matlab/lib/pfehdm180.m`
- Migration guide: `MATLAB_TO_PYTHON_MIGRATION.md`
- Python implementation: `bob/bio/vein/extractor/HandGeometry.py`

## Contact

For questions about the hand geometry implementation:
- Bob Development Mailing List: https://www.idiap.ch/software/bob/discuss
- GitLab Issues: https://gitlab.idiap.ch/bob/bob.bio.vein/issues

## License

Simplified BSD License (consistent with other bob.bio.vein modules)
