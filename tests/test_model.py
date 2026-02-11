import pytest
import torch
import os
import pandas as pd
import numpy as np
from adipo_finder.model import AdipoModel, AdipoNN

def test_adiponn_init():
    """Test neural network initialization."""
    input_dim = 10
    model = AdipoNN(input_dim)
    # Check architecture basics
    assert isinstance(model.net, torch.nn.Sequential)
    # First layer input
    assert model.net[0].in_features == input_dim

def test_adiponn_forward():
    """Test forward pass."""
    input_dim = 5
    model = AdipoNN(input_dim)
    x = torch.randn(2, input_dim) # Batch of 2
    y = model(x)
    assert y.shape == (2, 1)

def test_training_pipeline(sample_features_df):
    """Test the full training pipeline with dummy data."""
    # Run training for a few epochs
    model, train_ids, val_ids, test_ids, scaler = AdipoModel.train_model(
        sample_features_df, 
        n_epochs=2, 
        seed=42, 
        val_frac=0.33, 
        test_frac=0.33
    )
    
    assert isinstance(model, AdipoNN)
    assert len(train_ids) > 0
    # Since we have only 1 unique image_id ('img1') in sample_features_df, splits might be empty depending on logic.
    # Actually train_test_split on single item array puts it in train usually if test_size < 1.0
    # But let's check basic types.
    assert scaler is not None

def test_save_load_model(tmp_path):
    """Test saving and loading the model."""
    input_dim = 4
    model = AdipoNN(input_dim)
    filename = os.path.join(tmp_path, "test_model.pth")
    
    AdipoModel.save_model(filename, model, ['t1'], ['v1'], ['test1'])
    
    assert os.path.exists(filename)
    
    loaded_model, train, val, test = AdipoModel.load_model(filename)
    assert isinstance(loaded_model, AdipoNN)
    assert train == ['t1']
    assert val == ['v1']
    assert test == ['test1']
    # Check input dim matches
    assert loaded_model.net[0].in_features == input_dim

def test_predict_and_clean_image(sample_features_df):
    """Test prediction and cleaning."""
    # Train a dummy model first
    model, _, _, _, scaler = AdipoModel.train_model(
        sample_features_df, n_epochs=1, seed=42
    )
    
    # Create a dummy segmentation mask matching segment_ids
    # segment_ids are 1, 2, 3
    # Let's say we want to clean this mask based on model predictions
    seg_mask = np.zeros((10, 10), dtype=int)
    seg_mask[0:2, 0:2] = 1
    seg_mask[2:4, 2:4] = 2
    seg_mask[4:6, 4:6] = 3
    
    cleaned = AdipoModel.predict_and_clean_image(
        model, scaler, 'img1', sample_features_df, seg_mask
    )
    
    assert cleaned.shape == seg_mask.shape
    # Ideally we'd check if it actually removed something, but with random initialization 
    # and tiny data, prediction is random. Just check it runs and returns valid array.
    assert np.all(np.isin(cleaned, [0, 1, 2, 3]))
