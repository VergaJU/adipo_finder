import pytest
import numpy as np
from adipo_finder.features import FeatureExtraction

def test_extract_shape_size_features(labeled_image, distance_image):
    """Test shape/size feature extraction."""
    df = FeatureExtraction.extract_shape_size_features(labeled_image, distance_image)
    
    assert 'segment_id' in df.columns
    assert 'area' in df.columns
    assert 'eccentricity' in df.columns
    assert 'compactness' in df.columns
    assert 'max_dist' in df.columns
    
    # We have 2 objects + background (which regionprops ignores)
    # Actually regionprops returns properties for labeled regions 1, 2, ...
    assert len(df) == 2
    
    # Check if area is roughly correct (sqrt(area) is returned)
    # Larger circle (r=15, area~706) -> sqrt ~ 26.5
    # Smaller circle (r=10, area~314) -> sqrt ~ 17.7
    areas = df['area'].values
    assert np.any(areas > 25)
    assert np.any(areas < 20)

def test_extract_distance_from_shore(labeled_image, tissue_mask):
    """Test distance from shore extraction."""
    df = FeatureExtraction.extract_distance_from_shore(labeled_image, tissue_mask)
    
    assert 'segment_id' in df.columns
    assert 'distance_from_shore' in df.columns
    assert len(df) == 2
    
    # One object is deep in tissue (left side), one is outside (right side)
    # Tissue mask is 255 on left, 0 on right.
    # Gaussian blur spreads this.
    # Object on left should have higher 'distance_from_shore' (intensity) than object on right.
    
    # Note: 'distance_from_shore' is implemented as min_intensity in the blurred tissue mask.
    # So 'shore' is high intensity (tissue), 'ocean' is low intensity.
    
    # Get IDs of objects
    # Circle 1 at (25, 25) -> Left -> High intensity
    # Circle 2 at (75, 75) -> Right -> Low intensity
    
    # We can't easily know which label is which without checking, but one should be significantly higher than the other.
    assert df['distance_from_shore'].max() > df['distance_from_shore'].min()

def test_compute_ring_features(labeled_image, tissue_mask):
    """Test ring feature extraction."""
    df = FeatureExtraction.compute_ring_features(labeled_image, tissue_mask)
    
    assert 'segment_id' in df.columns
    assert 'frac_ring_other_objects' in df.columns
    assert 'frac_ring_tissue' in df.columns
    assert len(df) == 2
    
    # Object on left (25,25) is fully in tissue -> frac_ring_tissue should be ~1.0
    # Object on right (75,75) is fully out of tissue -> frac_ring_tissue should be ~0.0
    
    fracs = df['frac_ring_tissue'].values
    assert np.any(fracs > 0.9)
    assert np.any(fracs < 0.1)

def test_calculate_features(labeled_image, distance_image, tissue_mask):
    """Test full feature calculation."""
    df = FeatureExtraction.calculate_features(labeled_image, distance_image, tissue_mask)
    
    expected_cols = [
        'segment_id', 'area', 'eccentricity', 'compactness', 'max_dist', 
        'distance_from_shore', 'frac_ring_other_objects', 'frac_ring_tissue'
    ]
    for col in expected_cols:
        assert col in df.columns
        
    assert len(df) == 2
