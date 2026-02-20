import numpy as np

from adipo_finder.segmentation import Segmentation


def test_find_local_maxima(binary_image):
    """Test finding local maxima."""
    # We create a binary mask (bool) and an inverted opened image (markers)
    # For testing, we can use the binary image as the marker source
    bin_mask = binary_image > 0
    distance, markers = Segmentation.find_local_maxima(bin_mask, binary_image)
    
    assert distance.shape == binary_image.shape
    assert markers.shape == binary_image.shape
    # We expect at least two markers since there are two circles
    assert len(np.unique(markers)) >= 3  # 0 (background) + 2 markers


def test_apply_watershed_segmentation(binary_image):
    """Test watershed segmentation."""
    bin_mask = binary_image > 0
    distance, markers = Segmentation.find_local_maxima(bin_mask, binary_image)
    
    # Use smaller window for small test image objects
    segmented = Segmentation.apply_watershed_segmentation(
        binary_image, markers, distance, window=2
    )
    assert segmented.shape == binary_image.shape
    assert len(np.unique(segmented)) >= 3


def test_preprocess_input_image(binary_image):
    """Test the full preprocessing and segmentation pipeline."""
    # This replaces the old run_segmentation test
    bin_mask, distance, segmented, cleaned = Segmentation.preprocess_input_image(
        binary_image, min_distance_seg_init=10, min_island_area=0
    )
    
    assert bin_mask.shape == binary_image.shape
    assert distance.shape == binary_image.shape
    assert segmented.shape == binary_image.shape
    assert cleaned.shape == binary_image.shape
    
    # Check we found objects
    assert len(np.unique(segmented)) >= 3


def test_remove_jagged_spindle_objects(labeled_image):
    """Test removing jagged/elongated objects."""
    # labeled_image has 2 round circles. They should NOT be removed.
    filtered, removed = Segmentation.remove_jagged_spindle_objects(
        labeled_image, ecc_thresh=0.9, compactness_thresh=0.5
    )
    # None should be removed
    assert len(removed) == 0
    assert np.array_equal(filtered, labeled_image)
    
    # Now let's create a jagged/elongated object
    # Create a long thin line
    from skimage.draw import line
    bad_obj = np.zeros_like(labeled_image)
    rr, cc = line(10, 50, 90, 50) # vertical line
    bad_obj[rr, cc] = 1 
    
    # Make it slightly thicker so it has area > 0
    bad_obj[rr, cc+1] = 1
    
    # It needs to be labeled
    from skimage.measure import label
    bad_lbl = label(bad_obj)
    
    # With stricter thresholds
    filtered_bad, removed_bad = Segmentation.remove_jagged_spindle_objects(
        bad_lbl, ecc_thresh=0.8, compactness_thresh=0.5
    )
    
    # Should be removed
    assert len(removed_bad) > 0
    assert np.sum(filtered_bad) == 0


def test_far_from_shore_filter(labeled_image, tissue_mask):
    """Test far from shore filter."""
    # Create a tissue mask that is far away from objects
    far_tissue_mask = np.zeros_like(tissue_mask)
    far_tissue_mask[:, :5] = 255 # Only first 5 pixels
    
    # Objects are at 25 and 75. 
    # Gaussian blur (sigma=40) will spread, but intensity should be low far away.
    
    filtered = Segmentation.far_from_shore_filter(
        labeled_image, far_tissue_mask, min_shore_intensity=200
    )
    
    # Expect things to be removed if they are far/low intensity
    # With min_shore_intensity=200, basically everything outside the bright mask should go.
    
    # Check if we removed something (labeled_image has objects, filtered should have fewer/none)
    assert np.sum(filtered) < np.sum(labeled_image)


def test_size_shape_filter(labeled_image, distance_image):
    """Test size and shape filtering."""
    # Filter strictly to remove everything
    filtered = Segmentation.size_shape_filter(
        labeled_image,
        distance_image,
        min_dist=100,
        max_dist=200,
        min_area=10000,
        max_area=20000,
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

def test_preprocess_and_segment(binary_image):
    """Test preprocess_and_segment."""
    # This method can take 'image' directly.
    # It returns a tuple of (features_df, bin_mask, distance, segmented, cleaned_mask)
    
    # We pass None for gt_img, so ground_truth column should be NaN
    
    df, bin_mask, distance, segmented, cleaned = Segmentation.preprocess_and_segment(
        image=binary_image,
        min_distance=10,
        min_island_area=0
    )
    
    assert bin_mask.shape == binary_image.shape
    assert distance.shape == binary_image.shape
    assert segmented.shape == binary_image.shape
    assert len(np.unique(segmented)) >= 3
    assert not df.empty
    assert "area" in df.columns
    assert "ground_truth" in df.columns
    assert np.all(df["ground_truth"].isna())


def test_prepare_datasets(binary_image):
    """Test prepare_datasets (batch processing)."""
    # Create two dummy images
    images = [binary_image, binary_image]
    ids = ["img1", "img2"]
    
    full_df, bin_masks, dists, segs, cleaned_masks = Segmentation.prepare_datasets(
        all_ids=ids,
        input_images=images,
        min_distance_seg_init=10
    )
    
    assert len(full_df) > 0
    assert len(bin_masks) == 2
    assert len(segs) == 2
    
    # Check if image_id column is correct
    assert "img1" in full_df["image_id"].values
    assert "img2" in full_df["image_id"].values
