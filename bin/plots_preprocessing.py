import os
import tifffile
import matplotlib.pyplot as plt
import numpy as np

def _ensure_dir(path):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def plot_save_preprocessing_genes(
    sp_object,
    subplots_pos,
    S_intermediate_plots,
    list_of_genes_intermediate_plots,
    save_dir
):
    for i in range(subplots_pos.shape[0]):
        y0 = subplots_pos[i][0]
        x0 = subplots_pos[i][1]

        roi = sp_object.sel(
            x=slice(x0, x0 + S_intermediate_plots),
            y=slice(y0, y0 + S_intermediate_plots)
        )

        path = os.path.join(save_dir, f"preprocessing_{i}_ROI.png")
        _ensure_dir(path)

        dapi = roi["_image"].sel(channels="DAPI").compute().values

        # simple multi-marker projection for visualization
        marker_img = roi["_image"].sel(channels=list_of_genes_intermediate_plots).compute().values
        if marker_img.ndim == 3:
            marker_img = np.max(marker_img, axis=0)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(dapi, cmap="gray", origin="upper")
        ax[0].set_title("DAPI")
        ax[0].axis("off")

        ax[1].imshow(marker_img, cmap="gray", origin="upper")
        ax[1].set_title("Markers")
        ax[1].axis("off")

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)



def plot_save_segmentation_masks(sp_object, subplots_pos, S_intermediate_plots, save_dir):
    for i in range(subplots_pos.shape[0]):
        y0 = subplots_pos[i][0]
        x0 = subplots_pos[i][1]

        roi = sp_object.sel(
            x=slice(x0, x0 + S_intermediate_plots),
            y=slice(y0, y0 + S_intermediate_plots)
        )

        path = os.path.join(save_dir, f"segmentation_{i}_ROI.png")
        _ensure_dir(path)

        dapi = roi["_image"].sel(channels="DAPI").compute().values
        seg = roi["_segmentation"].compute().values

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(dapi, cmap="gray", origin="upper")
        ax[0].set_title(f"DAPI (subplot {i})")
        ax[0].axis("off")

        bound = (seg != np.roll(seg, 1, axis=0)) | (seg != np.roll(seg, 1, axis=1))

        ax[1].imshow(dapi, alpha=0.5, cmap="Purples", origin="upper")
        ax[1].imshow(bound, alpha=0.7, cmap="gray", origin="upper")
        ax[1].set_title("DAPI + seg boundaries")
        ax[1].axis("off")

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)


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