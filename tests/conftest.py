import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import distance_transform_edt, label
from skimage import draw


@pytest.fixture
def binary_image():
    """Creates a simple 100x100 binary image with two circles."""
    img = np.zeros((100, 100), dtype=np.uint8)
    rr, cc = draw.disk((25, 25), 10)
    img[rr, cc] = 255
    rr, cc = draw.disk((75, 75), 15)
    img[rr, cc] = 255
    return img


@pytest.fixture
def labeled_image(binary_image):
    """Creates a labeled version of the binary image."""
    labeled, _ = label(binary_image)
    return labeled


@pytest.fixture
def distance_image(binary_image):
    """Creates a distance transform of the binary image."""
    return distance_transform_edt(binary_image)


@pytest.fixture
def tissue_mask():
    """Creates a 'tissue' mask that covers the left half of the image."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[:, :50] = 255
    return mask


@pytest.fixture
def sample_adata():
    """Creates a dummy AnnData object."""
    counts = np.random.randint(0, 100, (10, 5))
    adata = ad.AnnData(counts)
    adata.obs_names = [f"cell_{i}" for i in range(10)]
    adata.var_names = [f"gene_{i}" for i in range(5)]

    # Mock spatial data structure
    library_id = "lib_1"
    adata.uns["spatial"] = {
        library_id: {"images": {"segmentation": np.random.randint(0, 2, (100, 100))}}
    }
    # Mock observation data needed for expand_df
    adata.obs["library_id"] = [library_id] * 10
    adata.obs["sample_col"] = ["sample_val"] * 10

    return adata


@pytest.fixture
def sample_features_df():
    """Creates a dummy features DataFrame."""
    data = {
        "segment_id": [1, 2, 3, 4, 5, 6],
        "area": [100.0, 200.0, 150.0, 120.0, 180.0, 130.0],
        "eccentricity": [0.1, 0.8, 0.5, 0.2, 0.7, 0.4],
        "compactness": [0.8, 0.4, 0.6, 0.9, 0.5, 0.7],
        "max_dist": [10.0, 20.0, 15.0, 12.0, 18.0, 14.0],
        "distance_from_shore": [5.0, 50.0, 0.0, 10.0, 40.0, 2.0],
        "ground_truth": [1, 0, 1, 1, 0, 1],
        "image_id": ["img1", "img1", "img2", "img2", "img3", "img3"],
    }
    return pd.DataFrame(data)
