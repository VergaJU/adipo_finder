from .segmentation import Segmentation
from .features import FeatureExtraction
from .model import AdipoModel
from .utils import Preprocessing, Exporting
from anndata import AnnData
import numpy as np
import pandas as pd
import pickle
from importlib import resources
from typing import Tuple

class AdipoFinder:
    @staticmethod
    def extract_images(adata: AnnData,
                      library_id: str = 'library_id') -> Tuple[
                          list[str],
                          list[np.ndarray]
                      ]:
        all_ids = adata.obs[library_id].unique()
        input_images = [Preprocessing.extract_segmentation_image(adata, id) for id in all_ids]
        return all_ids,input_images
    
    @staticmethod
    def segment_all(all_ids: list[str],
                    input_images:list[np.ndarray],
                    min_distance_seg_init:int = 20, 
                    min_island_area:int = 50):
        full_df, all_bin_mask, _, all_segmented, _ = Segmentation.prepare_datasets(all_ids=all_ids,
                                                                                                              input_images=input_images,
                                                                                                              min_distance_seg_init=min_distance_seg_init,
                                                                                                              min_island_area=min_island_area)
        return full_df, all_segmented,all_bin_mask


    @staticmethod
    def predict(
            all_ids: list[str],
            full_df: pd.DataFrame,
            all_segmented: list[np.ndarray],
            all_bin_mask: list[np.ndarray]
            ):
        
        model_path = str(resources.files("adipo_finder.data") / "trained_model_main.pth")
        scaler_path = resources.files("adipo_finder.data").joinpath(
            "trained_model_scaler.pkl")
        model, _, _, _ = AdipoModel.load_model(model_path)

        with scaler_path.open("rb") as f:
            scaler = pickle.load(f)

        clean_images = AdipoModel.predict_and_clean_images(model=model,
                                                            scaler=scaler,
                                                            full_df=full_df,
                                                            all_ids=all_ids,
                                                            all_unfiltered_seg=all_segmented, 
                                                            all_bin_mask=all_bin_mask,
                                                            path="new_predictions/", 
                                                            plot_path="new_prediction_plots/")
        return clean_images


    @staticmethod
    def export_results(adata: AnnData,
                       all_ids: list[str], 
                       clean_images: list[np.ndarray], 
                       input_images: list[np.ndarray], 
                       missing_cols: list[str],
                       cell_anot_key: list[str]) -> AnnData:
    
        for id,new_img,old_seg in zip(all_ids, clean_images, input_images):
            adata = Exporting.updated_adata(adata = adata, 
                                            library_id = id,
                                            missing_columns=missing_cols,
                                            old_seg=old_seg,
                                            new_img=new_img,
                                            cell_annot_key=cell_anot_key)
        return adata
    
    @classmethod
    def run(cls,
            adata: AnnData,
            missing_cols: list[str],
            cell_anot_key: list[str],
            library_id: str = 'library_id',
            min_distance_seg_init:int = 20, 
            min_island_area:int = 50
            ) -> AnnData:
        
        all_ids,input_images=cls.extract_images(adata=adata,library_id=library_id)
        full_df, all_segmented,all_bin_mask = cls.segment_all(all_ids=all_ids,
                                                              input_images=input_images,
                                                              min_distance_seg_init=min_distance_seg_init,
                                                              min_island_area=min_island_area)
        clean_images = cls.predict(all_ids=all_ids,
                                   full_df=full_df,
                                   all_segmented=all_segmented,
                                   all_bin_mask=all_bin_mask)
        adata = cls.export_results(adata=adata,
                                   all_ids=all_ids,
                                   clean_images=clean_images,
                                   input_images=input_images,
                                   missing_cols=missing_cols,
                                   cell_anot_key=cell_anot_key
                                   )
        
        return adata