import numpy as np
import pytest
from adipo_finder.utils import Preprocessing, Exporting, shuffle_labels, remove_segments
import pandas as pd

def test_preprocess_image(binary_image):
    """Test preprocessing."""
    # Preprocessing expects an image, does gaussian filter (blur), thresholding, and inversion
    # Our binary image is black background (0), white objects (255)
    
    # After inversion: white background (255), black objects (0)
    # The gaussian filter will blur edges.
    
    seg_image, inverted = Preprocessing.preprocess_image(image=binary_image, sigma=1.0)
    
    assert seg_image.shape == binary_image.shape
    assert inverted.shape == binary_image.shape
    
    # Inverted image should be mostly white (255) where original was black (0)
    # Original (0,0) is 0. Inverted (0,0) should be 255.
    assert inverted[0, 0] == 255

def test_export_adipocytes(labeled_image):
    """Test exporting adipocytes."""
    # We create a new segmentation mask (subset of original)
    # Let's just use the labeled image itself as 'new_segmentation'
    
    updated_seg, df = Exporting.export_adipocytes(
        segmentation_image=labeled_image,
        new_segmentation=labeled_image
    )
    
    assert updated_seg.shape == labeled_image.shape
    assert not df.empty
    assert 'area' in df.columns
    assert 'centroid-0' in df.columns
    
    # Check if number of rows equals number of objects
    num_objects = len(np.unique(labeled_image)) - 1
    assert len(df) == num_objects

def test_shuffle_labels(labeled_image):
    """Test label shuffling."""
    shuffled = shuffle_labels(labeled_image)
    assert shuffled.shape == labeled_image.shape
    
    # Should have same number of unique labels
    assert len(np.unique(shuffled)) == len(np.unique(labeled_image))
    
    # Background should stay 0
    assert shuffled[0, 0] == 0

def test_remove_segments(labeled_image):
    """Test removing segments."""
    # Remove label 1
    filtered = remove_segments(labeled_image, remove_ids=[1])
    
    unique_labels = np.unique(filtered)
    assert 1 not in unique_labels
    assert 0 in unique_labels # background should persist
    
    # Original image had label 1
    assert 1 in np.unique(labeled_image)

def test_expand_df(sample_adata):
    """Test expanding dataframe with AnnData info."""
    # Create a dummy dataframe
    df = pd.DataFrame({'Cell_ID': [1, 2], 'val': [10, 20]})
    library_id = "lib_1" # Matches conftest.py
    
    # sample_adata has 'sample_col' in obs
    expanded_df = Exporting.expand_df(
        adata=sample_adata,
        df=df,
        library_id=library_id,
        missing_columns=['sample_col'],
        cell_annot_key='cell_type'
    )
    
    assert 'sample_col' in expanded_df.columns
    assert 'cell_type' in expanded_df.columns
    assert 'library_id' in expanded_df.columns
    assert expanded_df['library_id'].iloc[0] == library_id

def test_create_adipo_adata(sample_adata):
    """Test creating new AnnData."""
    df = pd.DataFrame({
        'centroid-0': [10, 20],
        'centroid-1': [10, 20],
        'Cell_ID': [1, 2]
    })
    
    new_adata = Exporting.create_adipo_adata(sample_adata, df)
    
    assert new_adata.shape[0] == 2
    assert new_adata.shape[1] == sample_adata.shape[1]
    assert 'spatial' in new_adata.obsm
