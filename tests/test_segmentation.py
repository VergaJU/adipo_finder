import numpy as np
import pytest
from adipo_finder.segmentation import Segmentation

def test_find_local_maxima(binary_image):
    """Test finding local maxima."""
    distance, markers = Segmentation.find_local_maxima(binary_image)
    assert distance.shape == binary_image.shape
    assert markers.shape == binary_image.shape
    # We expect at least two markers since there are two circles
    assert len(np.unique(markers)) >= 3 # 0 (background) + 2 markers

def test_apply_watershed_segmentation(binary_image):
    """Test watershed segmentation."""
    distance, markers = Segmentation.find_local_maxima(binary_image)
    # Use smaller window for small test image objects
    segmented = Segmentation.apply_watershed_segmentation(binary_image, markers, distance, window=2)
    assert segmented.shape == binary_image.shape
    assert len(np.unique(segmented)) >= 3

# ...

def test_run_segmentation(binary_image):
    """Test the full segmentation pipeline."""
    # Use smaller window for small test image objects
    result = Segmentation.run_segmentation(binary_image, min_size=10, pixels=1, window=2)
    assert result.shape == binary_image.shape
    assert len(np.unique(result)) >= 3

def test_size_shape_filter(labeled_image, distance_image):
    """Test size and shape filtering."""
    # Filter strictly to remove everything
    filtered = Segmentation.size_shape_filter(
        labeled_image, distance_image, 
        min_dist=100, max_dist=200, min_area=10000, max_area=20000
    )
    assert np.sum(filtered) == 0

def test_in_the_middle_of_the_ocean_filter(labeled_image, tissue_mask):
    """Test ocean filter."""
    # The circle at (25, 25) is in the tissue mask (left side)
    # The circle at (75, 75) is NOT in the tissue mask (right side)
    
    filtered, removed, kept = Segmentation.in_the_middle_of_the_ocean_filter(
        labeled_image, tissue_mask, ring_size=5, min_tissue_overlap=0.1
    )
    
    # We expect one object to be removed (the one on the right)
    assert len(removed) >= 1
    # We expect one object to be kept (the one on the left)
    assert len(kept) >= 1
