import pytest
import numpy as np
from adipo_finder.evaluation import Evaluation

def test_evaluate_segmentation():
    """Test segmentation evaluation (IoU)."""
    # Create simple Ground Truth
    gt = np.zeros((10, 10), dtype=int)
    gt[1:4, 1:4] = 1 # Square 3x3
    
    # Create simple Prediction (shifted by 1 pixel)
    pred = np.zeros((10, 10), dtype=int)
    pred[2:5, 2:5] = 1 # Shifted square
    
    # Overlap region:
    # GT: rows 1,2,3, cols 1,2,3
    # Pred: rows 2,3,4, cols 2,3,4
    # Intersection: rows 2,3, cols 2,3 -> 2x2 = 4 pixels
    # Union: GT (9) + Pred (9) - Intersection (4) = 14 pixels
    # IoU = 4/14 = 0.2857
    
    # If overlap_threshold is 0.2, it should be a TP
    res_tp = Evaluation.evaluate_segmentation(pred, gt, overlap_threshold=0.2)
    assert 1 in res_tp['TP']
    assert 1 not in res_tp['FP']
    assert 1 not in res_tp['FN_labels_in_GT']
    assert res_tp['tp_coverage'][0] == pytest.approx(4/14)
    
    # If overlap_threshold is 0.5, it should be a FP (and GT is missed -> FN)
    res_fp = Evaluation.evaluate_segmentation(pred, gt, overlap_threshold=0.5)
    assert 1 in res_fp['FP']
    assert 1 in res_fp['FN_labels_in_GT'] # GT label 1 was missed

def test_assign_gt_labels():
    """Test assigning GT labels to predictions."""
    gt = np.zeros((10, 10), dtype=int)
    gt[0:3, 0:3] = 1 # Area 9
    
    pred = np.zeros((10, 10), dtype=int)
    pred[0:3, 0:3] = 1 # Perfect match
    pred[5:8, 5:8] = 2 # No match
    
    mapping = Evaluation.assign_gt_labels(pred, gt, overlap_threshold=0.5)
    
    assert mapping[1] == 1 # Pred 1 matches GT 1
    assert mapping[2] == 0 # Pred 2 matches nothing

def test_compute_segmentation_metrics():
    """Test metric computation from raw counts."""
    metrics_dict = {
        "TP": [1, 2],
        "FP": [3],
        "FN": [4],
        "tp_coverage": [0.8, 0.9]
    }
    
    stats = Evaluation.compute_segmentation_metrics(metrics_dict)
    
    # Precision = TP / (TP + FP) = 2 / 3 = 0.66
    # Recall = TP / (TP + FN) = 2 / 3 = 0.66
    
    assert stats['precision'] == pytest.approx(2/3)
    assert stats['recall'] == pytest.approx(2/3)
    assert stats['mean_IoU_TP'] == pytest.approx(0.85)
