import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

class Evaluation:
    """
    Class for evaluating segmentation results.
    """

    @staticmethod
    def evaluate_segmentation(segmented_final: np.ndarray, segmented_ground_truth: np.ndarray, overlap_threshold: float = 0.4) -> dict:
        """
        Fast evaluation against ground truth using Jaccard index (IoU), selecting matches by *max IoU*.

        Returns a dict with:
            - TP, FP, FN: segment IDs (TP/FP from 'segmented_final', FN are GT labels)
            - TP_labels_in_GT: GT label matched to each TP
            - FP_best_GT: best-overlap GT label for each FP (or None)
            - FN_labels_in_GT: GT labels that were missed
            - tp_coverage, fp_coverage, fn_coverage: best IoU values for each list above
        """

        # Flatten
        pred = segmented_final.ravel()
        gt   = segmented_ground_truth.ravel()

        # Unique labels (exclude background=0) and their true pixel counts
        f_labels, f_counts = np.unique(pred[pred > 0], return_counts=True)
        g_labels, g_counts = np.unique(gt[gt > 0],   return_counts=True)
        nf, ng = len(f_labels), len(g_labels)

        # Short-circuit trivial case
        if nf == 0 and ng == 0:
            return {
                "TP": [], "FP": [], "FN": [],
                "TP_labels_in_GT": [], "FP_best_GT": [], "FN_labels_in_GT": [],
                "tp_coverage": [], "fp_coverage": [], "fn_coverage": []
            }

        # Build sparse intersection matrix only over pixels where both labels > 0
        both = (pred > 0) & (gt > 0)
        if np.any(both):
            pred_pos = pred[both]
            gt_pos   = gt[both]
            # Map raw labels -> compressed indices [0..nf-1], [0..ng-1]
            f_idx = np.searchsorted(f_labels, pred_pos)
            g_idx = np.searchsorted(g_labels, gt_pos)
            data  = np.ones_like(f_idx, dtype=np.int64)
            inter = coo_matrix((data, (f_idx, g_idx)), shape=(nf, ng)).tocsr()
        else:
            inter = csr_matrix((nf, ng), dtype=np.int64)

        # Prepare outputs
        TP, FP, FN = [], [], []
        TP_labels_in_GT, FP_best_GT, FN_labels_in_GT = [], [], []
        tp_coverage, fp_coverage, fn_coverage = [], [], []

        # Convenience handles for CSR
        indptr, indices, values = inter.indptr, inter.indices, inter.data

        # --- Classify each predicted segment as TP or FP by max IoU over overlapping GTs ---
        matched_gt_labels = set()
        for i in range(nf):
            start, end = indptr[i], indptr[i+1]
            if start == end:
                FP.append(int(f_labels[i]))
                FP_best_GT.append(None)
                fp_coverage.append(0.0)
                continue

            cols = indices[start:end]            # GT indices overlapping this predicted segment
            inter_vals = values[start:end].astype(np.float64)
            unions = f_counts[i] + g_counts[cols] - inter_vals
            ious = inter_vals / unions

            best_k = int(np.argmax(ious))
            j = int(cols[best_k])
            best_iou = float(ious[best_k])

            if best_iou >= overlap_threshold:
                TP.append(int(f_labels[i]))
                TP_labels_in_GT.append(int(g_labels[j]))
                tp_coverage.append(best_iou)
                matched_gt_labels.add(int(g_labels[j]))
            else:
                FP.append(int(f_labels[i]))
                FP_best_GT.append(int(g_labels[j]))
                fp_coverage.append(best_iou)

        # --- Now mark any GT segment not matched as FN ---
        for j, g in enumerate(g_labels):
            if g not in matched_gt_labels:
                FN.append(None)
                FN_labels_in_GT.append(int(g))
                fn_coverage.append(0.0)

        return {
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "TP_labels_in_GT": TP_labels_in_GT,
            "FP_best_GT": FP_best_GT,
            "FN_labels_in_GT": FN_labels_in_GT,
            "tp_coverage": tp_coverage,
            "fp_coverage": fp_coverage,
            "fn_coverage": fn_coverage
        }

    @staticmethod
    def calc_f1(metrics: dict) -> tuple[float, float, float]:
        """
        Calculate precision, recall, and F1 score from metrics.
        """
        tp = len(metrics['TP'])
        fp = len(metrics['FP'])
        fn = len(metrics['FN'])
        P = tp / (tp + fp) if (tp + fp) else 0.0 #precision
        R = tp / (tp + fn) if (tp + fn) else 0.0 #recall
        F1 = 2*P*R/(P+R) if (P+R) else 0.0
        return P, R, F1

    @classmethod
    def get_pred_eval_dataframe(cls, gt_ids: list, old_metrics: list, new_metrics: list, train_ids: list, val_ids: list, test_ids: list) -> pd.DataFrame:
        """
        Create a DataFrame summarizing evaluation metrics for multiple images.
        """
        rows = []
        for gt_id, m_old, m_new in zip(gt_ids, old_metrics, new_metrics):
            P_old, R_old, F1_old = cls.calc_f1(m_old)
            P_new, R_new, F1_new = cls.calc_f1(m_new)
            
            if gt_id in train_ids:
                split = "train"
            elif gt_id in val_ids:
                split = "val"
            elif gt_id in test_ids:
                split = "test"
            else:
                split = "unknown"
            
            rows.append({
                "gt_id": gt_id,
                "split": split,
                "P_old": P_old, "R_old": R_old, "F1_old": F1_old,
                "P_new": P_new, "R_new": R_new, "F1_new": F1_new
            })

        df = pd.DataFrame(rows)
        return df

    @staticmethod
    def compute_segmentation_metrics(metrics_dict: dict) -> dict:
        """
        Compute precision, recall, F1, and mean Jaccard from segmentation results.
        """
        TP = metrics_dict["TP"]
        FP = metrics_dict["FP"]
        FN = metrics_dict["FN"]
        tp_coverage = metrics_dict["tp_coverage"]

        precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0.0
        recall = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0.0
        F1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_IoU_TP = np.mean(tp_coverage) if tp_coverage else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "F1": F1,
            "mean_IoU_TP": mean_IoU_TP
        }

    @staticmethod
    def assign_gt_labels(segmented_final: np.ndarray, segmented_ground_truth: np.ndarray, overlap_threshold: float = 0.5) -> dict:
        """
        Assign each predicted object in segmented_final the GT label with which it has
        the highest IoU (>= overlap_threshold). If no GT passes threshold, assign 0.

        Returns:
            mapping: dict {pred_label -> assigned_gt_label}
                     (background excluded, unmatched preds map to 0)
        """

        pred = segmented_final.ravel()
        gt   = segmented_ground_truth.ravel()

        # Unique nonzero labels
        f_labels, f_counts = np.unique(pred[pred > 0], return_counts=True)
        g_labels, g_counts = np.unique(gt[gt > 0],   return_counts=True)
        nf, ng = len(f_labels), len(g_labels)

        # Empty case
        if nf == 0 or ng == 0:
            return {int(f): 0 for f in f_labels}

        # Build sparse intersection matrix only where pred>0 and gt>0
        both = (pred > 0) & (gt > 0)
        if np.any(both):
            f_idx = np.searchsorted(f_labels, pred[both])
            g_idx = np.searchsorted(g_labels, gt[both])
            data  = np.ones_like(f_idx, dtype=np.int64)
            inter = coo_matrix((data, (f_idx, g_idx)), shape=(nf, ng)).tocsr()
        else:
            inter = csr_matrix((nf, ng), dtype=np.int64)

        mapping = {}
        indptr, indices, values = inter.indptr, inter.indices, inter.data

        for i in range(nf):
            start, end = indptr[i], indptr[i+1]
            if start == end:
                mapping[int(f_labels[i])] = 0
                continue

            cols = indices[start:end]
            inter_vals = values[start:end].astype(np.float64)
            unions = f_counts[i] + g_counts[cols] - inter_vals
            ious = inter_vals / unions

            best_k = int(np.argmax(ious))
            j = int(cols[best_k])
            best_iou = float(ious[best_k])

            if best_iou >= overlap_threshold:
                mapping[int(f_labels[i])] = int(g_labels[j])
            else:
                mapping[int(f_labels[i])] = 0

        return mapping
