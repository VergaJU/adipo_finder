import os

import numpy as np
import torch

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
    x = torch.randn(2, input_dim)  # Batch of 2
    y = model(x)
    assert y.shape == (2, 1)


def test_training_pipeline(sample_features_df):
    """Test the full training pipeline with dummy data."""
    # Run training for a few epochs
    model, train_ids, val_ids, test_ids, scaler = AdipoModel.train_model(
        sample_features_df, n_epochs=2, seed=42, val_frac=0.33, test_frac=0.33
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

    AdipoModel.save_model(filename, model, ["t1"], ["v1"], ["test1"])

    assert os.path.exists(filename)

    loaded_model, train, val, test = AdipoModel.load_model(filename)
    assert isinstance(loaded_model, AdipoNN)
    assert train == ["t1"]
    assert val == ["v1"]
    assert test == ["test1"]
    # Check input dim matches
    assert loaded_model.net[0].in_features == input_dim


def test_predict_and_clean_images(sample_features_df, tmp_path):
    """Test batch prediction and cleaning."""
    model, _, _, _, scaler = AdipoModel.train_model(
        sample_features_df, n_epochs=1, seed=42
    )
    
    # Setup dummy data
    # Two images: img1, img2. 
    all_ids = ["img1", "img2"]
    
    # Create dummy segmentation masks
    seg1 = np.zeros((10, 10), dtype=int)
    seg1[1, 1] = 1 # segment 1
    seg1[5, 5] = 2 # segment 2
    
    seg2 = np.zeros((10, 10), dtype=int)
    seg2[2, 2] = 3 # segment 3
    
    all_unfiltered_seg = [seg1, seg2]
    all_bin_mask = [seg1 > 0, seg2 > 0]
    
    # Output paths
    out_path = str(tmp_path / "preds/") + "/"
    plot_path = str(tmp_path / "plots/") + "/"
    
    # Run batch prediction
    cleaned_images = AdipoModel.predict_and_clean_images(
        model=model,
        scaler=scaler,
        full_df=sample_features_df,
        all_ids=all_ids,
        all_unfiltered_seg=all_unfiltered_seg,
        all_bin_mask=all_bin_mask,
        path=out_path,
        plot_path=plot_path,
        threshold=0.5,
        plot=True # Test plotting too
    )
    
    assert len(cleaned_images) == 2
    assert cleaned_images[0].shape == (10, 10)
    assert os.path.exists(out_path)
    assert os.path.exists(plot_path)
    # Check if files were created
    assert os.path.exists(os.path.join(out_path, "img1 pred_adipocytes.png"))
    assert os.path.exists(os.path.join(plot_path, "img1 prediction_plot.png"))


def test_evaluate_model(sample_features_df):
    """Test model evaluation."""
    model, _, _, _, scaler = AdipoModel.train_model(
        sample_features_df, n_epochs=1, seed=42
    )
    
    # Evaluate on "img1" which is in sample_features_df
    # We need to make sure 'img1' is in the test set or we just force it.
    test_ids = ["img1"]
    
    # This just prints to stdout, but ensures no runtime error
    AdipoModel.evaluate_model(model, scaler, sample_features_df, test_ids)
