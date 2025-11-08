# MATLAB to Python Migration Guide

This document provides a comprehensive guide for users transitioning from the MATLAB implementations to the Python implementations of Miura et al.'s finger vein extraction methods.

## Overview

All MATLAB code from the original Bram Ton implementation has been successfully transcoded to Python and integrated into the `bob.bio.vein` package. The Python implementations maintain functional equivalence with the MATLAB versions while leveraging Python's scientific computing ecosystem (NumPy, SciPy).

## File Mapping

### Finger Vein Recognition (Completed)

| MATLAB File | Python Implementation | Description | Status |
|-------------|----------------------|-------------|---------|
| `matlab/lib/miura_max_curvature.m` | `bob/bio/vein/extractor/MaximumCurvature.py` | Maximum curvature method for vein extraction | ✅ Complete |
| `matlab/lib/miura_repeated_line_tracking.m` | `bob/bio/vein/extractor/RepeatedLineTracking.py` | Repeated line tracking method for vein extraction | ✅ Complete |
| `matlab/lib/miura_match.m` | `bob/bio/vein/algorithm/MiuraMatch.py` | Cross-correlation matching algorithm | ✅ Complete |
| `matlab/lib/miura_usage.m` | See usage examples below | Example usage (not a core algorithm) | ✅ Complete |

### Hand/Palm Geometry Recognition (In Progress)

| MATLAB File | Python Implementation | Description | Status |
|-------------|----------------------|-------------|---------|
| `matlab/lib/pfehdm180.m` | `bob/bio/vein/extractor/HandGeometry.py` | Hand geometry feature extraction system | 🚧 In Progress |

**Note**: The hand geometry extractor (`pfehdm180.m`) is a comprehensive system with ~2000 lines of code including multiple sub-functions. The Python implementation is being developed in phases:
1. ✅ Core module structure created
2. 🚧 Hand segmentation (SEGHAND180)
3. 🚧 Finger point detection (Unhand18)
4. 🚧 Distance calculations (dist18)
5. 🚧 Orientation normalization (orimhd18)
6. 🚧 Geometric measurements (MMpmain18)
7. 🚧 Helper functions and utilities
8. 🚧 Testing and validation

## Detailed Migration Examples

### 1. Maximum Curvature Method

**MATLAB Usage:**
```matlab
% Load and prepare image
img = im2double(imread('finger.png'));
fvr = lee_region(img, 4, 40);  % Get finger region mask

% Extract veins using maximum curvature
sigma = 3;
veins = miura_max_curvature(img, fvr, sigma);

% Binarize
md = median(veins(veins > 0));
veins_bin = veins > md;
```

**Python Equivalent:**
```python
import numpy as np
from bob.bio.vein.extractor import MaximumCurvature
from bob.io.base import load

# Load and prepare image
finger_image = load('finger.hdf5').astype('float64')
finger_mask = load('mask.hdf5').astype('bool')

# Extract veins using maximum curvature
extractor = MaximumCurvature(sigma=3)
veins_bin = extractor([finger_image, finger_mask])
# Note: Binarization is already performed internally
```

**Key Differences:**
- Python version combines all steps (extraction + binarization) in a single call
- Python uses NumPy arrays instead of MATLAB matrices
- Image loading uses HDF5 format instead of PNG for better precision

### 2. Repeated Line Tracking Method

**MATLAB Usage:**
```matlab
% Load and prepare image
img = im2double(imread('finger.png'));
fvr = lee_region(img, 4, 40);

% Extract veins using repeated line tracking
max_iterations = 3000;
r = 1;
W = 17;
veins = miura_repeated_line_tracking(img, fvr, max_iterations, r, W);

% Binarize
md = median(veins(veins > 0));
veins_bin = veins > md;
```

**Python Equivalent:**
```python
import numpy as np
from bob.bio.vein.extractor import RepeatedLineTracking
from bob.io.base import load

# Load and prepare image
finger_image = load('finger.hdf5')
finger_mask = load('mask.hdf5').astype('bool')

# Extract veins using repeated line tracking
extractor = RepeatedLineTracking(
    iterations=3000,
    r=1,
    profile_w=17,
    seed=0  # For reproducibility
)
veins_bin = extractor([finger_image, finger_mask])
# Note: Binarization is already performed internally
```

**Key Differences:**
- Python version includes automatic binarization and skeletonization
- Parameter `W` is renamed to `profile_w` for clarity
- Added `seed` parameter for reproducible random walks
- Added `rescale` parameter (default: True) for image scaling

### 3. Miura Matching

**MATLAB Usage:**
```matlab
% Match two vein patterns
I = veins_probe_bin;
R = veins_enrolled_bin;
cw = 80;  % Max horizontal displacement
ch = 30;  % Max vertical displacement

score = miura_match(I, R, cw, ch);
fprintf('Match score: %6.4f\n', score);
```

**Python Equivalent:**
```python
import numpy as np
from bob.bio.vein.algorithm import MiuraMatch

# Match two vein patterns
probe = veins_probe_bin.astype('float64')
enrolled = veins_enrolled_bin.astype('float64')

# Create matcher
matcher = MiuraMatch(ch=30, cw=80)

# Compute match score
score = matcher.score(enrolled, probe)
print(f'Match score: {score:.4f}')
```

**Key Differences:**
- Python version uses object-oriented interface
- Supports batch processing of multiple probes/enrollments
- Parameter order is (ch, cw) instead of (cw, ch) for consistency

## Parameter Mapping

### MaximumCurvature

| MATLAB Parameter | Python Parameter | Default | Description |
|------------------|------------------|---------|-------------|
| `sigma` | `sigma` | 5 | Standard deviation for Gaussian smoothing |
| `stem` (optional) | N/A | - | File stem for debug output (removed in Python) |

### RepeatedLineTracking

| MATLAB Parameter | Python Parameter | Default | Description |
|------------------|------------------|---------|-------------|
| `iterations` | `iterations` | 3000 | Maximum number of tracking iterations |
| `r` | `r` | 1 | Distance between tracking point and profile cross-section |
| `W` | `profile_w` | 21 | Width of the profile (must be odd) |
| N/A | `rescale` | True | Whether to rescale image (0.6x) |
| N/A | `seed` | 0 | Random seed for reproducibility |

### MiuraMatch

| MATLAB Parameter | Python Parameter | Default | Description |
|------------------|------------------|---------|-------------|
| `cw` | `cw` | 90 | Maximum search displacement in x-direction |
| `ch` | `ch` | 80 | Maximum search displacement in y-direction |

## Function Call Mapping

### MATLAB Functions → Python/NumPy Equivalents

| MATLAB Function | Python/NumPy Equivalent |
|----------------|------------------------|
| `im2double()` | `array.astype('float64') / 255.0` |
| `imread()` | `bob.io.base.load()` or `PIL.Image.open()` |
| `imfilter()` | `scipy.ndimage.convolve()` |
| `meshgrid()` | `numpy.meshgrid()` |
| `conv2()` | `scipy.signal.convolve2d()` |
| `rot90()` | `numpy.rot90()` |
| `median()` | `numpy.median()` |
| `zeros()` | `numpy.zeros()` |
| `ones()` | `numpy.ones()` |
| `size()` | `array.shape` |
| `find()` | `numpy.argwhere()` or `numpy.where()` |
| `randperm()` | `numpy.random.permutation()` |
| `rand()` | `numpy.random.random()` or `numpy.random.random_sample()` |

## Data Type Mapping

| MATLAB | Python/NumPy |
|--------|--------------|
| `double` | `numpy.float64` |
| `uint8` | `numpy.uint8` |
| `logical` | `numpy.bool_` |
| Matrix indexing (1-based) | Array indexing (0-based) |
| `matrix(y,x)` | `array[y, x]` |

## Installation

### MATLAB
Requires MATLAB with Image Processing Toolbox:
```matlab
% Download from MATLAB Central File Exchange
% Add to path
addpath('matlab/lib');
```

### Python
```bash
# Using conda (recommended)
conda install bob.bio.vein

# Or using pip
pip install bob.bio.vein
```

## Testing and Validation

The Python implementations have been validated against MATLAB references. A comparison script is available:

```bash
cd matlab
./run.sh ../bob/bio/vein/tests/extractors/image.hdf5 ../bob/bio/vein/tests/extractors/mask.hdf5 mc
python compare.py
```

This will output the sum of absolute differences between MATLAB and Python implementations, which should be near zero (accounting for floating-point precision).

## Performance Considerations

### MATLAB
- Optimized for matrix operations
- JIT compilation for loops
- Parallel Computing Toolbox support

### Python
- Vectorized NumPy operations (comparable speed to MATLAB)
- SciPy FFT-based convolutions for large kernels
- Can leverage multiple cores with joblib or multiprocessing

**Performance Tips:**
- Use `scipy.signal.fftconvolve()` for large images (already used in Python implementation)
- Enable NumPy/SciPy optimizations: `conda install numpy scipy -c conda-forge`
- For batch processing, consider using `dask` or `joblib` for parallelization

## Common Pitfalls

1. **Indexing**: MATLAB uses 1-based indexing, Python uses 0-based
   ```matlab
   % MATLAB
   first_element = array(1, 1)
   ```
   ```python
   # Python
   first_element = array[0, 0]
   ```

2. **Array Dimensions**: MATLAB stores images as (height, width), often transposed in file I/O
   ```python
   # May need to transpose when loading MATLAB files
   image = load_matlab_file('image.mat')
   image = image.T  # Transpose if needed
   ```

3. **Random Number Generation**: Different algorithms, results will vary
   ```python
   # Use seed parameter for reproducibility
   extractor = RepeatedLineTracking(seed=42)
   ```

4. **Data Types**: Ensure correct types to avoid unexpected behavior
   ```python
   # Always convert to correct type
   image = image.astype('float64')
   mask = mask.astype('bool')
   ```

## References

- **Original MATLAB Code**: [Bram Ton's MATLAB Central File Exchange](http://ch.mathworks.com/matlabcentral/fileexchange/35716-miura-et-al-vein-extraction-methods)
- **Maximum Curvature Paper**: N. Miura, A. Nagasaka, and T. Miyatake, "Extraction of Finger-Vein Pattern Using Maximum Curvature Points in Image Profiles," IAPR MVA, 2005
- **Repeated Line Tracking Paper**: N. Miura, A. Nagasaka, and T. Miyatake, "Feature extraction of finger vein patterns based on repeated line tracking and its application to personal identification," Machine Vision and Applications, Vol. 15, No. 4, 2004
- **Python Documentation**: [Bob Bio Vein Documentation](https://www.idiap.ch/software/bob/docs/bob/bob.bio.vein/master/index.html)

## Support

For questions or issues:
- **MATLAB Code**: Original author - Bram Ton <b.t.ton@alumnus.utwente.nl>
- **Python Implementation**: Bob Development [Mailing List](https://www.idiap.ch/software/bob/discuss)
- **Bug Reports**: [GitLab Issues](https://gitlab.idiap.ch/bob/bob.bio.vein/issues)

## License

Both MATLAB and Python implementations are released under the Simplified BSD License.
