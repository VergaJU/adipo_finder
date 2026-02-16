import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.measure import regionprops
from skimage.morphology import disk


class FeatureExtraction:
    """
    Class for extracting features from segmented images.
    """

    @staticmethod
    def extract_shape_size_features(
        segmented_image: np.ndarray, distance_image: np.ndarray
    ) -> pd.DataFrame:
        """
        Extract shape and size features for each segmented object.

        Features included:
          - segment_id
          - area (as sqrt(area))
          - eccentricity
          - compactness
          - max_dist (from distance transform, requires distance_image)

        parameters:
        segmented_image: np.ndarray, labeled segmentation mask
        distance_image: np.ndarray, distance transform image used to compute max_dist feature

        return:
        pd.DataFrame, dataframe with extracted features
        """
        features = []
        props = regionprops(segmented_image, intensity_image=distance_image)

        for region in props:
            ecc = region.eccentricity
            area = region.area
            if region.perimeter == 0:  # div by zero otherwise
                compactness = 0
            else:
                compactness = 4 * np.pi * region.area / (region.perimeter**2)

            # distance-based feature
            max_dist = region.max_intensity

            features.append(
                {
                    "segment_id": region.label,
                    "area": np.sqrt(area),
                    "eccentricity": ecc,
                    "compactness": compactness,
                    "max_dist": max_dist,
                }
            )

        df = pd.DataFrame(features)
        return df

    @staticmethod
    def extract_distance_from_shore(
        segmented_image: np.ndarray,
        bin_tissue_image: np.ndarray,
        smoothing_sigma: float = 40,
    ) -> pd.DataFrame:
        """
        Measure how close each object is to tissue/shore.

        parameters:
        segmented_image: np.ndarray, labeled segmentation mask
        bin_tissue_image: np.ndarray, tissue mask
        smoothing_sigma: float, sigma for gaussian smoothing of tissue mask

        return:
        pd.DataFrame, dataframe with segment_id and distance_from_shore
        """
        from .utils import \
            Preprocessing  # Import here to avoid circular dependency if utils imports features

        features = []

        # removes object that are far from land
        blurred_image = Preprocessing.apply_gaussian_filter(
            bin_tissue_image, sigma=smoothing_sigma
        )
        props = regionprops(segmented_image, intensity_image=blurred_image)

        for region in props:
            # min intensity inside blurred shore mask tells how far from "shore" object is
            distance_from_shore = region.min_intensity
            features.append(
                {"segment_id": region.label, "distance_from_shore": distance_from_shore}
            )

        return pd.DataFrame(features)

    @staticmethod
    def compute_ring_features(
        segmented_image: np.ndarray, tissue_mask: np.ndarray, ring_size: int = 10
    ) -> pd.DataFrame:
        """
        Compute features for each object in segmented_image:
            - fraction of ring covered by other objects
            - fraction of ring covered by tissue

        parameters:
        segmented_image: np.ndarray, labeled segmented image
        tissue_mask: np.ndarray, binary mask of tissue
        ring_size: int, radius of ring for feature calculation

        return:
        pd.DataFrame, dataframe with columns ['segment_id', 'frac_ring_other_objects', 'frac_ring_tissue']
        """
        object_ids = np.unique(segmented_image)
        object_ids = object_ids[object_ids != 0]  # exclude background

        selem = disk(ring_size)
        tissue_mask_bin = tissue_mask != 0
        features = []

        nrows, ncols = segmented_image.shape

        for region in regionprops(segmented_image):
            obj_id = region.label
            minr, minc, maxr, maxc = region.bbox

            # buffered patch
            r0 = max(minr - ring_size, 0)
            r1 = min(maxr + ring_size, nrows)
            c0 = max(minc - ring_size, 0)
            c1 = min(maxc + ring_size, ncols)

            obj_patch = segmented_image[r0:r1, c0:c1] == obj_id
            seg_patch = segmented_image[r0:r1, c0:c1]
            tissue_patch = tissue_mask_bin[r0:r1, c0:c1]

            # ring around object
            ring = binary_dilation(obj_patch, selem) & ~obj_patch

            # feature 1: fraction of ring covered by other objects
            ring_other_objects = ring & (seg_patch != 0)
            ring_size_nonzero = np.sum(ring)
            frac_other_objects = (
                np.sum(ring_other_objects) / ring_size_nonzero
                if ring_size_nonzero > 0
                else 0.0
            )

            # feature 2: fraction of ring covered by tissue
            frac_tissue = (
                np.sum(tissue_patch[ring]) / ring_size_nonzero
                if ring_size_nonzero > 0
                else 0.0
            )

            features.append(
                {
                    "segment_id": obj_id,
                    "frac_ring_other_objects": frac_other_objects,
                    "frac_ring_tissue": frac_tissue,
                }
            )

        df = pd.DataFrame(features)
        return df

    @classmethod
    def calculate_features(
        cls,
        prefiltered_seg_image: np.ndarray,
        distance_image: np.ndarray,
        bin_mask_image: np.ndarray,
    ) -> pd.DataFrame:
        """
        Calculate all features for segmented objects.

        parameters:
        prefiltered_seg_image: np.ndarray, segmented image
        distance_image: np.ndarray, distance transform image
        bin_mask_image: np.ndarray, binary mask of tissue

        return:
        pd.DataFrame, merged features
        """
        sz_feat = cls.extract_shape_size_features(prefiltered_seg_image, distance_image)
        dfs_feat = cls.extract_distance_from_shore(
            prefiltered_seg_image, bin_mask_image
        )
        itmoto_feat = cls.compute_ring_features(
            prefiltered_seg_image, bin_mask_image, ring_size=10
        )

        # Merge the three feature dataframes on 'segment_id'
        if sz_feat.empty:  # handle empty case
            return pd.DataFrame(
                columns=[
                    "segment_id",
                    "area",
                    "eccentricity",
                    "compactness",
                    "max_dist",
                    "distance_from_shore",
                    "frac_ring_other_objects",
                    "frac_ring_tissue",
                ]
            )

        merged_feat = sz_feat.merge(dfs_feat, on="segment_id", how="outer").merge(
            itmoto_feat, on="segment_id", how="outer"
        )
        # very small objects get NaN values for these features, set those to zero
        cols_to_fill = ["area", "eccentricity", "compactness", "max_dist"]
        merged_feat[cols_to_fill] = merged_feat[cols_to_fill].fillna(0)

        return merged_feat
