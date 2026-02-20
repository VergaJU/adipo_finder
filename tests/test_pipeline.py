import os
import pickle
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from adipo_finder.pipeline import AdipoFinder


def test_extract_images(sample_adata):
    """Test extracting images from adata."""
    all_ids, input_images = AdipoFinder.extract_images(
        sample_adata, library_id="library_id"
    )
    assert isinstance(all_ids, (list, np.ndarray))
    assert len(all_ids) == 1
    assert all_ids[0] == "lib_1"
    assert isinstance(input_images, list)
    assert len(input_images) == 1
    assert input_images[0].shape == (100, 100)


@patch("adipo_finder.pipeline.Segmentation.prepare_datasets")
def test_segment_all(mock_prepare, binary_image):
    """Test segment_all method."""
    # Mock return values
    full_df = pd.DataFrame({"id": [1]})
    bin_masks = [binary_image]
    distances = [binary_image]
    segmented = [binary_image]
    cleaned = [binary_image]
    
    mock_prepare.return_value = (full_df, bin_masks, distances, segmented, cleaned)
    
    all_ids = ["lib_1"]
    input_images = [binary_image]
    
    res_df, res_seg, res_bin = AdipoFinder.segment_all(all_ids, input_images)
    
    assert res_df.equals(full_df)
    assert res_seg == segmented
    assert res_bin == bin_masks
    mock_prepare.assert_called_once()


@patch("adipo_finder.pipeline.AdipoModel.predict_and_clean_images")
@patch("adipo_finder.pipeline.pickle.load")
@patch("adipo_finder.pipeline.AdipoModel.load_model")
@patch("adipo_finder.pipeline.resources.files")
def test_predict(mock_files, mock_load_model, mock_pickle_load, mock_predict_clean, binary_image):
    """Test predict method."""
    # Mock resources
    mock_files.return_value.__truediv__.return_value = "dummy_path"
    mock_files.return_value.joinpath.return_value.open.return_value.__enter__.return_value = MagicMock()
    
    # Mock model loading
    mock_model = MagicMock()
    mock_load_model.return_value = (mock_model, [], [], [])
    
    # Mock scaler loading
    mock_scaler = MagicMock()
    mock_pickle_load.return_value = mock_scaler
    
    # Mock prediction result
    mock_predict_clean.return_value = [binary_image]
    
    # Inputs
    all_ids = ["lib_1"]
    full_df = pd.DataFrame()
    all_segmented = [binary_image]
    all_bin_mask = [binary_image]
    
    clean_images = AdipoFinder.predict(all_ids, full_df, all_segmented, all_bin_mask)
    
    assert len(clean_images) == 1
    assert clean_images[0] is binary_image
    mock_predict_clean.assert_called_once()


@patch("adipo_finder.pipeline.Exporting.updated_adata")
def test_export_results(mock_updated_adata, sample_adata, binary_image):
    """Test export_results method."""
    # Mock updated_adata to return a new adata
    mock_updated_adata.return_value = sample_adata
    
    all_ids = ["lib_1"]
    clean_images = [binary_image]
    input_images = [binary_image]
    missing_cols = ["col1"]
    cell_annot_key = ["type"]
    
    result_adata = AdipoFinder.export_results(
        sample_adata, all_ids, clean_images, input_images, missing_cols, cell_annot_key
    )
    
    assert isinstance(result_adata, AnnData)
    mock_updated_adata.assert_called_once()


@patch("adipo_finder.pipeline.AdipoFinder.extract_images")
@patch("adipo_finder.pipeline.AdipoFinder.segment_all")
@patch("adipo_finder.pipeline.AdipoFinder.predict")
@patch("adipo_finder.pipeline.AdipoFinder.export_results")
def test_run(mock_export, mock_predict, mock_segment, mock_extract, sample_adata):
    """Test the full run pipeline."""
    # Setup mocks
    mock_extract.return_value = (["lib_1"], [np.zeros((100, 100))])
    mock_segment.return_value = (pd.DataFrame(), [np.zeros((100, 100))], [np.zeros((100, 100))])
    mock_predict.return_value = [np.zeros((100, 100))]
    mock_export.return_value = sample_adata
    
    result = AdipoFinder.run(
        sample_adata, 
        missing_cols=[], 
        cell_anot_key=[],
        library_id="library_id"
    )
    
    assert isinstance(result, AnnData)
    mock_extract.assert_called_once()
    mock_segment.assert_called_once()
    mock_predict.assert_called_once()
    mock_export.assert_called_once()
