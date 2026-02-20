import numpy as np
import pandas as pd

from adipo_finder.utils import (Exporting, Preprocessing, remove_segments,
                                shuffle_labels)


def test_segmentation_to_binary(labeled_image):
    """Test segmentation to binary conversion."""
    # labeled_image has labels 0, 1, 2...
    # threshold default is 0.5.
    binary = Preprocessing.segmentation_to_binary(labeled_image, threshold=0.5)
    
    assert binary.shape == labeled_image.shape
    # Should be 255 where labeled_image > 0
    assert np.all(binary[labeled_image > 0] == 255)
    assert np.all(binary[labeled_image == 0] == 0)


def test_apply_gaussian_filter(binary_image):
    """Test gaussian filter."""
    filtered = Preprocessing.apply_gaussian_filter(binary_image, sigma=1.0)
    assert filtered.shape == binary_image.shape
    # It preserves range and casts to uint8, so max should be near 255
    assert filtered.max() > 0
    assert filtered.dtype == np.uint8


def test_invert_image(binary_image):
    """Test image inversion."""
    inverted = Preprocessing.invert_image(binary_image)
    assert inverted.shape == binary_image.shape
    # Where original is 0, inverted should be 255
    assert np.all(inverted[binary_image == 0] == 255)
    assert np.all(inverted[binary_image == 255] == 0)


def test_extract_segmentation_image(sample_adata):
    """Test extracting segmentation image from adata."""
    # sample_adata setup in conftest: 
    # adata.uns["spatial"]["lib_1"]["images"]["segmentation"]
    
    img = Preprocessing.extract_segmentation_image(
        sample_adata, library_id="lib_1", spatial_key="spatial"
    )
    
    assert img.shape == (100, 100)
    assert isinstance(img, np.ndarray)


def test_merge_adatas(sample_adata):
    """Test merging AnnDatas."""
    # Create a temp adata to merge
    df = pd.DataFrame(
        {"centroid-0": [15], "centroid-1": [15], "Cell_ID": [100]},
        index=["cell_100"]
    )
    # The fixture sample_adata has 5 vars (gene_0..4). 
    # create_adipo_adata creates an AnnData with same var_names as input adata.
    adata_tmp = Exporting.create_adipo_adata(sample_adata, df)
    
    # New segmentation image
    new_seg = np.zeros((100, 100), dtype=int)
    
    # merge_adatas is an instance method? No, defined without @staticmethod in class Exporting?
    # Let's check utils.py... It's defined as `def merge_adatas(adata, ...)` inside class Exporting but missing @staticmethod decorator!
    # Wait, if it's missing @staticmethod, it expects 'self'.
    # I should check utils.py again.
    
    merged = Exporting.merge_adatas(
        sample_adata, adata_tmp, new_seg, library_id="lib_1"
    )
    
    # Should have more observations now? 
    # sample_adata has 10 cells. adata_tmp has 1. Merged should have 11 (outer join).
    assert merged.n_obs == 11
    
    # Check if segmentation was updated in uns
    assert np.array_equal(
        merged.uns["spatial"]["lib_1"]["images"]["segmentation"], 
        new_seg
    )


def test_export_adipocytes(labeled_image):
    """Test exporting adipocytes."""
    # We create a new segmentation mask (subset of original)
    # Let's just use the labeled image itself as 'new_segmentation'

    updated_seg, df = Exporting.export_adipocytes(
        segmentation_image=labeled_image, new_segmentation=labeled_image
    )

    assert updated_seg.shape == labeled_image.shape
    assert not df.empty
    assert "area" in df.columns
    assert "centroid-0" in df.columns

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
    assert 0 in unique_labels  # background should persist

    # Original image had label 1
    assert 1 in np.unique(labeled_image)


def test_expand_df(sample_adata):
    """Test expanding dataframe with AnnData info."""
    # Create a dummy dataframe
    df = pd.DataFrame({"Cell_ID": [1, 2], "val": [10, 20]})
    library_id = "lib_1"  # Matches conftest.py

    # sample_adata has 'sample_col' in obs
    expanded_df = Exporting.expand_df(
        adata=sample_adata,
        df=df,
        library_id=library_id,
        missing_columns=["sample_col"],
        cell_annot_key="cell_type",
    )

    assert "sample_col" in expanded_df.columns
    assert "cell_type" in expanded_df.columns
    assert "library_id" in expanded_df.columns
    assert expanded_df["library_id"].iloc[0] == library_id


def test_create_adipo_adata(sample_adata):
    """Test creating new AnnData."""
    df = pd.DataFrame(
        {"centroid-0": [10, 20], "centroid-1": [10, 20], "Cell_ID": [1, 2]}
    )

    new_adata = Exporting.create_adipo_adata(sample_adata, df)

    assert new_adata.shape[0] == 2
    assert new_adata.shape[1] == sample_adata.shape[1]
    assert "spatial" in new_adata.obsm
