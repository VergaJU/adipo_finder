import scanpy as sc
import numpy as np 
from anndata import AnnData
import anndata as ad
from matplotlib import pyplot as plt 
from skimage import measure, morphology, filters
from scipy.ndimage import distance_transform_edt
import pandas as pd

class Preprocessing:
    @staticmethod
    def extract_segmentation_image(adata: AnnData,
                                library_id: str,
                                spatial_key: str = 'spatial',
                                image_key: str = 'images',
                                segmentation_key: str = 'segmentation', **kwargs) -> np.ndarray:
        """
        Extract segmentation image from AnnData object as np.ndarray.
        Specify the library_id, image_key and segmentation_key.

        parameters:
        adata: AnnData object
        spatial_key: str, key of the spatial coordinates in adata.uns
        library_id: str, library id of the image in adata.uns[spatial_key]
        image_key: str, key of the image in adata.uns[spatial_key][library_id]
        segmentation_key: str, key of the segmentation adata.uns[spatial_key][library_id][image_key]

        return:
        np.ndarray, segmentation image
        """
        img=adata.uns[spatial_key][library_id][image_key][segmentation_key].copy()
        return img

    @staticmethod
    def segmentation_to_binary(segmentation: np.ndarray, threshold: float = 0.5,**kwargs) -> np.ndarray:
        """
        Convert segmentation image to binary mask.

        parameters:
        segmentation: np.ndarray, segmentation image
        threshold: float, threshold to binarize the segmentation image

        return:
        np.ndarray, binary mask
        """
        return (255 * (segmentation > threshold))


    @staticmethod
    def apply_gaussian_filter(image: np.ndarray, sigma: float = 1.0,**kwargs) -> np.ndarray:
        """
        Apply Gaussian filter to the image.

        parameters:
        image: np.ndarray, input image
        sigma: float, standard deviation of the Gaussian filter

        return:
        np.ndarray, filtered image
        """
        return filters.gaussian(image, sigma=sigma,preserve_range=True).astype('uint8')


    @staticmethod
    def invert_image(image: np.ndarray) -> np.ndarray:
        """
        Invert the image.

        parameters:
        image: np.ndarray, input image

        return:
        np.ndarray, inverted image
        """
        return np.where(image == 0, 255, 0).astype('uint8')


    @classmethod
    def preprocess_image(cls, adata: AnnData,library_id: str, **kwargs) -> np.ndarray:
        """
        Preprocess the image for segmentation.
        Extract the segmentation image from the AnnData object.
        Convert the segmentation image to binary mask.
        Apply Gaussian filter to the binary mask.
        Invert the binary mask.

        parameters:
        adata: AnnData object
        library_id: str, library id of the image in adata.uns[spatial_key]

        return:
        np.ndarray, preprocessed image
        """
        # set kwargs for the functions
        spatial_key = kwargs.get('spatial_key', 'spatial')
        image_key = kwargs.get('image_key', 'images')
        segmentation_key = kwargs.get('segmentation_key', 'segmentation')
        threshold = kwargs.get('threshold', 0.5)
        sigma = kwargs.get('sigma', 1.0)
        # Extract the segmentation image
        segmentation_image = cls.extract_segmentation_image(adata, library_id, **kwargs)
        # apply gaussian filter
        blurred_image = cls.apply_gaussian_filter(segmentation_image,**kwargs)
        # convert to binary
        binary_image = cls.segmentation_to_binary(blurred_image,**kwargs)
        # invert image
        inverted_image = cls.invert_image(binary_image)
        return segmentation_image,inverted_image


class Plotting:
    @staticmethod
    def plot_3_channel_image(image: np.ndarray, invert_image: np.ndarray, segmented_image: np.ndarray) -> None:
        """
        Create 3 channel image and plot it.

        parameters:
        image: np.ndarray, input image
        invert_image: np.ndarray, inverted image
        segmented_image: np.ndarray, segmented image

        return:
        None
        """
        # Stack images along the last axis to create a 3-channel image
        three_channel_image = np.stack((image, invert_image,segmented_image), axis=-1)

        # Display the 3-channel image using matplotlib
        fig, axes = plt.subplots(1, 1, figsize=(10, 5))
        axes.imshow(three_channel_image)
        axes.set_title('3-Channel Image (binary,inverted and segmented)')
        axes.axis('off')

        plt.show()


    @staticmethod
    def plot_centroids(image: np.ndarray,size:float=10., figsize: tuple=(10,10)) -> None:
        """
        Plot the centroids on the image.

        parameters:
        image: np.ndarray, input image

        return:
        None
        """

        labeled_img=measure.label(image)
        regions=measure.regionprops(labeled_img)
        # compute centroids
        centroids = np.array([region.centroid for region in regions])

        # plot labeled image
    
        fig, axes = plt.subplots(1, 1, figsize=figsize)
        axes.imshow(image, cmap='gray')
        axes.scatter(centroids[:, 1], centroids[:, 0], c='r', s=size, marker='+')
        axes.set_title('Centroids')
        axes.axis('off')

        plt.show()





class Exporting:
    @staticmethod
    def export_adipocytes(segmentation_image: np.ndarray, 
                        new_segmentation: np.ndarray, 
                        cell_id: str = 'Cell_ID', **kwargs) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Export adipocytes from the segmentation image.
        Add the new objects to the original segmentation image, coloring maintain the same order.
        Export objects numbers and metadata as a pandas dataframe to add the to the single cell object.

        parameters:
        segmentation_image: np.ndarray, segmentation image
        new_segmentation: np.ndarray, new segmentation image

        return:
        np.ndarray, updated segmentation image
        """
        base_image=segmentation_image.copy()
        # Initialize the new object value
        new_value = segmentation_image.max()+1
        # DataFrame to store object properties
        columns = ['area', 'centroid-0', 'centroid-1', 'axis_major_length', 'axis_minor_length', 'eccentricity', cell_id]
        df = pd.DataFrame(columns=columns)

        # Update the base image and extract object properties
        for region in measure.regionprops(measure.label(new_segmentation)):
            base_image[new_segmentation == region.label] = new_value
            new_row=pd.DataFrame.from_dict({
                'area': region.area,
                'centroid-0': region.centroid[0],
                'centroid-1': region.centroid[1],
                'axis_major_length': region.major_axis_length,
                'axis_minor_length': region.minor_axis_length,
                'eccentricity': region.eccentricity,
                cell_id: new_value
            },
                                        orient='index').T
            df = pd.concat([df,new_row], axis=0, ignore_index=True)
            new_value += 1
        
        return base_image, df


    @staticmethod
    def expand_df(adata: AnnData, df: pd.DataFrame, 
                library_id: str, 
                missing_columns: list[str],
                cell_annot_key: str | list[str],
                cell_type: str = 'Adipocyte',
                cell_id: str = 'Cell_ID',
                library_key: str = 'library_id', **kwarg) -> pd.DataFrame:
        """
        Integrate the adipocytes metadata with the AnnData obs.

        parameters:
        adata: AnnData object
        df: pd.DataFrame, adipocytes data
        library_id: str, library id of the image in adata.uns[spatial_key]
        missing_columns: list, missing columns to add to the adata.obs
        cell_id: str, key of the cell id in adata.obs
        library_key: str, key of the library id in adata.obs


        return:
        AnnData, updated AnnData object
        """
        # retrieve info from adata.obs
        missing_data=adata.obs[adata.obs[library_key]==library_id][missing_columns].iloc[0].to_dict()
        df.index='s'+library_id.split('_')[1]+'_'+df[cell_id].astype(int).astype(str)
        df[cell_id]=df[cell_id].astype(np.int64)
        df['ObjectNumber']=df[cell_id].astype(np.int64)
        df[library_key]=library_id
        if cell_annot_key is str:
            df['cell_annot_key']=cell_type
        else:
            for i in cell_annot_key:
                df[i]=cell_type
        for k,v in missing_data.items():
            df[k]=v
        return df

    # function that takes the adata and the df and create new adata with the new df
    @staticmethod
    def create_adipo_adata(adata: AnnData,
                            df: pd.DataFrame, 
                            spatial_key: str = 'spatial',**kwargs) -> AnnData:
        """
        Create a new AnnData object with the adipocytes metadata.

        parameters:
        adata: AnnData object
        df: pd.DataFrame, adipocytes data
        spatial_key: str, key of the spatial coordinates in adata.obsm

        return:
        AnnData, new AnnData object
        """
        # create numpy array with centroids for spatial analysies
        spatial_tmp=df[['centroid-0','centroid-1']].to_numpy()
        # create numpy array of zeros for expression values
        # TODO: mean expression values from the channels?
        counts_tmp=np.zeros((df.shape[0],adata.shape[1]))
        # create adata object, fill it with numpy arra, spatial features, metadata and var_names
        adata_tmp=ad.AnnData(counts_tmp)
        adata_tmp.obs=df
        adata_tmp.obsm[spatial_key]=spatial_tmp
        adata_tmp.obs_names=list(df.index)
        adata_tmp.var_names=adata.var_names
        return adata_tmp
    

    # Function that takes the two adata and the new segmentation image and update the adata with the new segmentation
    def merge_adatas(adata: AnnData, 
                    adata_tmp: AnnData, 
                    new_segmentation: np.ndarray, 
                    library_id: str, 
                    spatial_key: str = 'spatial',
                    image_key: str = 'images',
                    segmentation_key: str = 'segmentation', **kwargs) -> AnnData:
        """
        Merge the new AnnData object with the original AnnData object.

        parameters:
        adata: AnnData object
        adata_tmp: AnnData object
        new_segmentation: np.ndarray, new segmentation image
        library_id: str, library id of the image in adata.uns[spatial_key]

        return:
        AnnData, updated AnnData object
        """
        # merge adata, outer join and keep uns
        adata_new=ad.concat([adata,adata_tmp],join='outer',uns_merge='only')
        # update the segmentation image
        adata_new.uns[spatial_key][library_id][image_key][segmentation_key]=new_segmentation
        return adata_new