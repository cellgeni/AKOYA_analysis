import spatialproteomics as sp
from skimage.io import imread
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
import pandas as pd
import yaml
import os
from typing import Tuple, Union
from plots_AKOYA import plot_marker_celltype_pairs_from_spobject
import fire
import gc
import anndata as ad


def ReadConfFile(FilePath):
    with open(FilePath, 'r') as file:
        data = yaml.safe_load(file)
    image_path = data['image_path']
    crop_x = data['crop_x']
    crop_y = data['crop_y']
    channel_segment = data['channel_for_segmentation']
    list_of_channels = data['list_of_channels']
    list_of_markers = data['list_of_markers']
    label_expansion = data['segmentation_label_expansion']
    min_area = data['min_area']
    max_area = data['max_area']
    save_intermediate_plots = data['save_intermediate_plots']
    N_intermediate_plots = data['number_intermediate_plots']
    S_intermediate_plots = data['size_intermediate_plots']
    list_of_genes_intermediate_plots = data['list_of_genes_intermediate_plots']
    save_binary_plots = data['save_individual_marker_presence_plots']
    threshold_binary = data['fraction_of_positive_pixels']
    output_dir = data['output_dir']
    normalise_intensity = data['normalise_intensity']
    save_intermediate_zarr = data['save_intermediate_zarr']
    list_output_formats = data['list_output_formats'] #can be ['zarr', 'h5ad', 'csv']
    pixelsize = data.get('pixelsize', None)
    return (image_path, crop_x, crop_y, channel_segment, list_of_channels, list_of_markers, 
            label_expansion, min_area, max_area, save_intermediate_plots, N_intermediate_plots, 
            S_intermediate_plots, save_binary_plots, threshold_binary, output_dir, 
            list_of_genes_intermediate_plots, normalise_intensity, save_intermediate_zarr, 
            list_output_formats, pixelsize)




def read_crop_cyx(path, crop_x, crop_y):
    """
    Read tif/qptiff, try to determine axes, return cropped image in CYX order.

    - Uses OME/series axes metadata if available (tifffile series.axes).
    - If metadata missing and image is 3D, assumes the smallest dim is channels (warns).
    - Supports 2D (YX) and 3D (CYX/YXC) inputs.
    """
    with tifffile.TiffFile(path) as tf:
        s = tf.series[0]
        arr = s.asarray()
        axes = getattr(s, "axes", None)
    # --- determine axes + put into CYX ---
    if axes is not None:
        axes = axes.upper()

        if axes == "YX":
            arr = arr[np.newaxis, ...]  # -> CYX (C=1)
        elif set(axes) == set("CYX") and len(axes) == 3:
            # reorder to CYX
            arr = np.transpose(arr, [axes.index("C"), axes.index("Y"), axes.index("X")])
        else:
            raise ValueError(f"Unsupported axes from metadata: {axes} (only YX, CYX, YXC supported).")

    else:
        # No metadata: support 2D/3D only
        if arr.ndim == 2:
            raise ValueError(
                f"The image seems ot have only one channel! Please make sure there are 3 axis: CYX"
            )
        elif arr.ndim == 3:
            c_axis = int(np.argmin(arr.shape))
            print(f"WARNING: No axes metadata. Assuming channel axis={c_axis} (shape={arr.shape}).")
            # move assumed channels to front => C??, then assume remaining are Y,X in their order
            arr = np.moveaxis(arr, c_axis, 0)
            if arr.shape[1] < 2 or arr.shape[2] < 2:
                raise ValueError(f"After channel inference, remaining dims don't look like YX: {arr.shape}")
        else:
            raise ValueError(
                f"No axes metadata and unsupported ndim={arr.ndim}. Only 2D or 3D supported without metadata."
            )
    # --- crop in XY (on CYX array) ---
    if crop_x or crop_y:
        if isinstance(crop_x, str):
            crop_x = [0, arr.shape[2]]
        if isinstance(crop_y, str):
            crop_y = [0, arr.shape[1]]
        cropped = arr[:, crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
    else:
        cropped = arr
    del arr    
    return cropped


def random_subimage_positions(
    image_size: Tuple[int, int],
    subimage_size: Union[int, Tuple[int, int]],
    n: int,
    rng: Union[int, np.random.Generator, None] = None,
) -> np.ndarray:
    """
    Generate random top-left corners of subimages fully contained in an image.

    Parameters
    ----------
    image_size : (sy, sx)
        Full image size (height, width)
    subimage_size : int or (sh, sw)
        Subimage size (square if int)
    n : int
        Number of subimages
    rng : int | np.random.Generator | None
        Random seed or generator

    Returns
    -------
    coords : (n, 2) ndarray
        Top-left corners as (y, x)
    """
    sy, sx = image_size
    if isinstance(subimage_size, int):
        sh, sw = subimage_size, subimage_size
    else:
        sh, sw = subimage_size
    if sh > sy or sw > sx:
        raise ValueError("Subimage size must be smaller than image size.")
    rng = np.random.default_rng(rng)
    max_y = sy - sh
    max_x = sx - sw
    ys = rng.integers(0, max_y + 1, size=n)
    xs = rng.integers(0, max_x + 1, size=n)

    return np.column_stack([ys, xs])

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _prepare_image_for_tiff(img_channel):
    """
    Ensure image is a 2D ndarray with integer dtype suitable for tif viewers.
    Accepts float arrays (assumed 0..1 or arbitrary) or integer arrays.
    Returns uint16.
    """
    arr = np.asarray(img_channel)
    if arr.ndim != 2:
        raise ValueError("Expected 2D channel image.")
    # If floats, try to scale either 0..1 -> 0..65535 or rescale from min/max
    if np.issubdtype(arr.dtype, np.floating):
        mn, mx = np.nanmin(arr), np.nanmax(arr)
        if mn == mx:
            arr_out = np.zeros_like(arr, dtype=np.uint16)
        else:
            # assume meaningful dynamic range, scale to uint16
            arr_norm = (arr - mn) / (mx - mn)
            arr_out = (arr_norm * 65535.0).round().astype(np.uint16)
    else:
        # integer type: upcast to uint16 safely (preserve values if <=65535)
        arr_out = arr.astype(np.uint16, copy=False)
    return arr_out


def save_whole_tissue_dapi(img, out_path, downscale_factor=None):
    """
    Save the DAPI channel (img is CYX numpy array).
    downscale_factor: None or integer. If integer, uses slicing (fast) but choose sensibly.
    """
    _ensure_dir(out_path)
    dapi = img[0]  # C,Y,X -> single channel
    if downscale_factor is not None and downscale_factor > 1:
        dapi = dapi[::downscale_factor, ::downscale_factor]
    dapi_uint16 = _prepare_image_for_tiff(dapi)
    # write using tifffile (uint16) - viewers can read
    tifffile.imwrite(out_path, dapi_uint16)
    return out_path


def plot_save_area_hist(sp_object, save_dir):
    areas = sp_object.pp.add_observations("area").pp.get_layer_as_df()["area"]
    path = os.path.join(save_dir, "area_histogram.png")
    _ensure_dir(path)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(areas, bins=100)
    ax.set_title("segmented cell area distribution after expansion and before filtering")
    ax.set_xlabel("area")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_save_preprocessing_genes(sp_object, subplots_pos, S_intermediate_plots, list_of_genes_intermediate_plots, save_dir):
    for i in range(subplots_pos.shape[0]):
        x0 = subplots_pos[i][1]; y0 = subplots_pos[i][0]
        path = os.path.join(save_dir, f"preprocessing_{i}_ROI.png")
        _ensure_dir(path)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        # create axes explicitly and pass them
        _ = sp_object.pp[["DAPI"], x0:x0+S_intermediate_plots, y0:y0+S_intermediate_plots].pl.show(ax=ax[0])
        _ = sp_object.pp[list_of_genes_intermediate_plots, x0:x0+S_intermediate_plots, y0:y0+S_intermediate_plots].pl.show(ax=ax[1])
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_save_segmentation_masks(sp_object, subplots_pos, S_intermediate_plots, save_dir):
    for i in range(subplots_pos.shape[0]):
        x0 = subplots_pos[i][1]; y0 = subplots_pos[i][0]
        roi = sp_object.sel(x=slice(x0, x0+S_intermediate_plots), y=slice(y0, y0+S_intermediate_plots))
        path = os.path.join(save_dir, f"segmentation_{i}_ROI.png")
        _ensure_dir(path)

        dapi = roi["_image"].sel(channels="DAPI").compute().values
        seg  = roi["_segmentation"].compute().values
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        ax[0].imshow(dapi, cmap='gray')
        ax[0].set_title("DAPI (subplot " + str(i) + ")")
        ax[0].axis("off")
        # draw segmentation boundaries on top
        bound = (seg != np.roll(seg, 1, axis=0)) | (seg != np.roll(seg, 1, axis=1))
        ax[1].imshow(dapi, alpha=0.5, cmap='Purples')
        ax[1].imshow(bound, alpha=0.7, cmap='gray')
        ax[1].set_title("DAPI + seg boundaries")
        ax[1].axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)

def normalize_intensities(sp_object):
    X = sp_object["_intensity"]  # dims: (cells, channels)
    mu = X.mean("cells")
    sd = X.std("cells")
    X_norm = (X - mu) / (sd + 1e-8)
    # I save norm intensiy in _intensity and original one in _intensity_raw
    sp_object = sp_object.assign(_intensity_raw=sp_object["_intensity"])
    sp_object = sp_object.assign(_intensity=X_norm)
    return sp_object

def write_tables_csv(sp_object, outdir):
    os.makedirs(outdir, exist_ok=True)
    SKIP = {"_image", "_segmentation"}  # huge
    for var in sp_object.data_vars:
        if var in SKIP:
            print(f"Skipping {var} (huge image/mask)")
            continue
        da = sp_object[var]
        # We only export "tables"
        if da.ndim != 2:
            print(f"Skipping {var} (ndim={da.ndim})")
            continue
        # Convert to pandas (dask -> compute)
        df = da.compute().to_pandas()
        # Write CSV
        path = os.path.join(outdir, f"{var}.csv")
        df.to_csv(path)
        print(f"Saved {path}  shape={df.shape}")


def spobject_to_anndata(
    sp_object,
    out_dir: str,
    sample_id: str = "sample",
    image_channels=None,          # e.g. ["DAPI"]
    image_downsample: int = 4,    # downsample for storing image in AnnData
    spot_diameter_fullres: float = 10.0,  # "point diameter" in FULLRES PIXELS
    store_lowres: bool = True,
    lowres_factor: int = 4,       # lowres relative to hires image
    pixel_size_um: float | None = None,   # microns per fullres pixel (optional)
):
    """
    Export spatialproteomics xarray Dataset -> AnnData ready for squidpy.pl.spatial_scatter.

    - adata.X := sp_object["_intensity_raw"] (cells x channels)
    - adata.obsm["spatial"] := cell centroids in fullres pixel coords (x, y)
    - adata.obsm["spatial_um"] := same coords in microns (if pixel_size_um is given)
    - adata.uns["spatial"][library_id]["images"]["hires"] := downsampled image (H, W, 3)
    - adata.uns["spatial"][library_id]["scalefactors"] includes:
        tissue_hires_scalef, tissue_lowres_scalef, spot_diameter_fullres,
        and (if pixel_size_um) pixel_size_um, spot_diameter_um
    """

    os.makedirs(out_dir, exist_ok=True)

    # -------------------------
    # 1) X matrix: intensity_raw
    # -------------------------
    if "_intensity_raw" in sp_object:
        X_da = sp_object["_intensity_raw"]
    elif "_intensity" in sp_object:
        X_da = sp_object["_intensity"]
    else:
        raise ValueError("Neither '_intensity_raw' nor '_intensity' exists in sp_object.")

    if tuple(X_da.dims) != ("cells", "channels"):
        X_da = X_da.transpose("cells", "channels")

    X = X_da.data
    var_names = [str(c) for c in sp_object.coords["channels"].values]

    # -------------------------
    # 2) obs from _obs
    # -------------------------
    if "_obs" in sp_object:
        obs_da = sp_object["_obs"]
        if tuple(obs_da.dims) != ("cells", "features"):
            obs_da = obs_da.transpose("cells", "features")
        feats = [str(f) for f in sp_object.coords["features"].values]
        obs_df = pd.DataFrame(obs_da.compute().values, columns=feats)
        obs_df.index = [str(c) for c in sp_object.coords["cells"].values]
    else:
        obs_df = pd.DataFrame(index=[str(c) for c in sp_object.coords["cells"].values])

    # -------------------------
    # 3) spatial coordinates (pixels)
    # -------------------------
    if "centroid-0" not in obs_df.columns or "centroid-1" not in obs_df.columns:
        raise ValueError("Couldn't find 'centroid-0' and 'centroid-1' in sp_object['_obs'].features.")

    # typical convention: centroid-0 = y, centroid-1 = x
    y = obs_df["centroid-0"].to_numpy()
    x = obs_df["centroid-1"].to_numpy()

    spatial_pix = np.c_[x, y].astype(np.float32)  # (x, y) in fullres pixels

    adata = ad.AnnData(
        X=X,
        obs=obs_df,
        var=pd.DataFrame(index=var_names),
    )
    adata.obsm["spatial"] = spatial_pix  # keep in pixels for Squidpy

    if pixel_size_um is not None:
        adata.obsm["spatial_um"] = spatial_pix * float(pixel_size_um)
    all_channels = [str(c) for c in sp_object.coords["channels"].values]
    if image_channels is None:
        image_channels = [all_channels[0]]
    ds_img = sp_object.pp[image_channels] if hasattr(sp_object, "pp") else sp_object.sel(channels=image_channels)
    ds_img = ds_img.isel(y=slice(None, None, image_downsample),
                         x=slice(None, None, image_downsample))
    img = ds_img["_image"]
    if tuple(img.dims) != ("channels", "y", "x"):
        img = img.transpose("channels", "y", "x")
    img_np = img.compute().values  # (C, H, W)
    C, H, W = img_np.shape
    if C == 1:
        rgb = np.repeat(img_np[0][..., None], 3, axis=2)
    else:
        take = min(3, C)
        rgb = np.moveaxis(img_np[:take], 0, 2)  # (H, W, take)
        if take < 3:
            rgb = np.concatenate(
                [rgb, np.repeat(rgb[..., -1:], 3 - take, axis=2)],
                axis=2
            )
    if rgb.dtype != np.uint8:
        arr = rgb.astype(np.float32)
        arr -= np.nanmin(arr)
        mx = np.nanmax(arr)
        if mx > 0:
            arr = arr / mx
        rgb = (255 * arr).clip(0, 255).astype(np.uint8)
    if store_lowres:
        step = max(1, int(lowres_factor))
        rgb_low = rgb[::step, ::step, :]
    else:
        rgb_low = None
    # scalefactors: same convention as before (coords are fullres pixels)
    tissue_hires_scalef = 1.0 / float(image_downsample)
    tissue_lowres_scalef = 1.0 / float(
        image_downsample * (lowres_factor if store_lowres else 1)
    )
    scalefactors = {
        "tissue_hires_scalef": tissue_hires_scalef,
        "spot_diameter_fullres": float(spot_diameter_fullres),
    }
    if store_lowres:
        scalefactors["tissue_lowres_scalef"] = tissue_lowres_scalef
    # If pixel size is given, add physical-size info
    if pixel_size_um is not None:
        scalefactors["pixel_size_um"] = float(pixel_size_um)
        scalefactors["spot_diameter_um"] = float(spot_diameter_fullres) * float(pixel_size_um)
    adata.uns["spatial"] = {
        sample_id: {
            "images": {
                "hires": rgb,
                **({"lowres": rgb_low} if store_lowres else {}),
            },
            "scalefactors": scalefactors,
            "metadata": {
                "source": "spatialproteomics",
                "image_channels": list(image_channels),
                "image_downsample": int(image_downsample),
                "note": (
                    "obsm['spatial'] is in fullres pixels (x,y). "
                    "If pixel_size_um is provided, obsm['spatial_um'] is in microns."
                ),
            },
        }
    }
    out_path = os.path.join(out_dir, f"{sample_id}.h5ad")
    adata.write_h5ad(out_path)
    print(f"Saved anndata object at {out_path}")
    return adata, out_path

def main(ConfFilePath):
    (image_path, crop_x, crop_y, channel_segment, list_of_channels, list_of_markers, label_expansion, min_area, 
    max_area, save_intermediate_plots, N_intermediate_plots, S_intermediate_plots, save_binary_plots, threshold_binary, 
    output_dir, list_of_genes_intermediate_plots, normalise_intensity, save_intermediate_zarr, 
    list_output_formats, pixelsize) = ReadConfFile(ConfFilePath)

    print('Output dir' + str(output_dir))
    path_zarr = os.path.join(output_dir,"sp_object.zarr")
    
    print('Reading the image')
    img = read_crop_cyx(image_path, crop_x, crop_y)
    image_size = img[0].shape
    print(image_size)
    sp_object = sp.load_image_data(img, channel_coords=list_of_channels)
    print(sp_object)
    if save_intermediate_plots:
        os.makedirs(output_dir, exist_ok=True)
        path1 = os.path.join(output_dir, 'whole_tissue_DAPI.png')
        save_whole_tissue_dapi(img, path1, downscale_factor=10)
    gc.collect()


    print('Image preprocesing')
    thrs_list = sp.pp.otsu_per_channel(img, channel_axis=0)
    del img
    sp_object = sp_object.pp.threshold(intensity=thrs_list).pp.apply(medfilt2d, kernel_size=3)
    if save_intermediate_zarr:
        sp_object.to_zarr(path_zarr, mode="w", zarr_version=2, consolidated=True)
    if save_intermediate_plots:       
        #firstly generate list of random subplot locations
        H = sp_object["_image"].sizes["y"]
        W = sp_object["_image"].sizes["x"]
        image_size_sp = (H, W)
        subplots_pos = random_subimage_positions(image_size_sp, S_intermediate_plots, N_intermediate_plots)
        print(subplots_pos)
        plot_save_preprocessing_genes(sp_object, subplots_pos, S_intermediate_plots, list_of_genes_intermediate_plots, output_dir)
    gc.collect()

    print("Segmentation and area filtering")
    sp_object = sp_object.tl.stardist(channel=channel_segment)
    if label_expansion:
        sp_object = sp_object.pp.expand_segmentation(radius = label_expansion)
    sp_object = sp_object.pp.add_observations("area")
    plot_save_area_hist(sp_object, output_dir)
    if min_area: 
        sp_object = sp_object.pp.filter_by_obs("area", lambda x: x > min_area)
    if max_area: 
        sp_object = sp_object.pp.filter_by_obs("area", lambda x: x < max_area)
    if save_intermediate_plots:
        plot_save_segmentation_masks(sp_object, subplots_pos, S_intermediate_plots, output_dir)
    gc.collect()
    if save_intermediate_zarr:
        sp_object.to_zarr(path_zarr, mode="w", zarr_version=2, consolidated=True)
    
    print("Constructing binary matrix of gene markers")
    sp_object = sp_object.pp.add_quantification(func="intensity_mean").pp.transform_expression_matrix(method="arcsinh")
    if normalise_intensity:
        sp_object = normalize_intensities(sp_object)
    sp_object = sp_object.pp.add_quantification(func=sp.percentage_positive, key_added="_percentage_positive")
    threshold_dict  = {k: threshold_binary for k in list_of_markers}
    sp_object = sp_object.la.threshold_labels(threshold_dict, layer_key="_percentage_positive")
    mapping_dict = {k: f"{k}_ct" for k in list_of_markers}
    if save_binary_plots:
        save_folder = os.path.join(output_dir, 'marker_genes vs celltype - whole tissue')
        plot_marker_celltype_pairs_from_spobject(sp_object, mapping_dict, out_dir=save_folder, dpi=300, expr_cmap="Reds",bbox=None,downscale=4)
        if save_intermediate_plots:
            mapping_dict2 = {k: f"{k}_ct" for k in list_of_genes_intermediate_plots}
            save_folder2 = os.path.join(output_dir, 'marker_genes vs celltype - ROIs')
            for i in range(subplots_pos.shape[0]):
                x0 = subplots_pos[i][1]; y0 = subplots_pos[i][0];
                roi_name = "subplot_" + str(i)
                plot_marker_celltype_pairs_from_spobject(sp_object, mapping_dict2, out_dir=save_folder2, dpi=300, expr_cmap="Reds",bbox=[x0, y0, x0+S_intermediate_plots, y0+S_intermediate_plots],downscale=1,roi_name = roi_name)

    print("Saving")
    if 'zarr' in list_output_formats:
        sp_object.to_zarr(path_zarr, mode="w", zarr_version=2, consolidated=True)
        print(f"Saved spatialproteomics object at {path_zarr}")
    if 'csv' in list_output_formats:
        save_folder_csv = os.path.join(output_dir, 'tables_csv')
        write_tables_csv(sp_object, save_folder_csv)
    if 'h5ad' in list_output_formats:
        spobject_to_anndata(sp_object, out_dir=output_dir, sample_id="anndata", image_downsample=10,
                            image_channels=["DAPI"],spot_diameter_fullres=20.0, pixel_size_um = pixelsize)
 
        
if __name__ == "__main__":
    fire.Fire(main) 



