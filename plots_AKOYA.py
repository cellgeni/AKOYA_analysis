import os
import re
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.transform import downscale_local_mean


def _downscale_image_mean(img2d: np.ndarray, factor: int) -> np.ndarray:
    """Downscale intensity image by local mean (fast, stable)."""
    if factor <= 1:
        return img2d
    return downscale_local_mean(img2d, (factor, factor))


def _downscale_mask_max(mask2d: np.ndarray, factor: int) -> np.ndarray:
    """Downscale binary mask by block max (preserves any positive pixel)."""
    if factor <= 1:
        return mask2d.astype(np.uint8)

    H, W = mask2d.shape
    Hc = (H // factor) * factor
    Wc = (W // factor) * factor
    if Hc == 0 or Wc == 0:
        return mask2d.astype(np.uint8)

    m = mask2d[:Hc, :Wc]
    m = m.reshape(Hc // factor, factor, Wc // factor, factor)
    return (m.max(axis=(1, 3)) > 0).astype(np.uint8)


def _norm_name(s: str) -> str:
    """Normalize marker/channel names for fuzzy matching (remove non-alnum, lower)."""
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).lower()


def _get_numeric_label_ids(layer_df: pd.DataFrame) -> pd.Series:
    """
    Return integer segmentation label ids as a pandas Series aligned with layer_df rows.
    Works whether ids are stored in the index or in a column.
    """
    # 1) numeric index
    if pd.api.types.is_integer_dtype(layer_df.index):
        return pd.Series(layer_df.index.astype(int), index=layer_df.index, name="label_id")

    # 2) try common columns
    for col in ["labels", "label", "cell", "cell_id", "cellid", "_label", "seg_label", "segmentation_label"]:
        if col in layer_df.columns:
            s2 = pd.to_numeric(layer_df[col], errors="coerce")
            if s2.notna().any():
                return pd.Series(s2.astype(int).values, index=layer_df.index, name="label_id")

    raise ValueError(
        "Could not find numeric segmentation label IDs in layer_df. "
        "Your '_labels' column appears to be categorical labels (e.g. 'TCF-1_ct'). "
        "Ensure the dataframe index or a column like 'label'/'cell_id' contains integer IDs "
        "that match sp_object['_segmentation']."
    )

def plot_marker_celltype_pairs_from_spobject(
    sp_object,
    marker2celltype: Dict[str, str],
    out_dir: str,
    dpi: int = 200,
    expr_cmap: str = "Reds",
    bbox: Optional[Tuple[int, int, int, int]] = None,  # (x0, y0, x1, y1)
    downscale: int = 4,
    grey_level: float = 0.85,
    vmax_percentile: float = 99.5,
    add_colorbar: bool = True,
    match_names_fuzzily: bool = True,
    verbose: bool = True,
    roi_name = None
) -> List[str]:
    """
    For each marker->celltype:
      left: expression image for marker from sp_object['_image'].sel(channels=marker)
      right: segmentation-based mask of cells positive for marker (from get_layer_as_df: '{marker}_binarized')
             positive cells colored with the chosen colormap; negatives are grey.

    Features:
      - Saves one PNG per pair into out_dir
      - dpi selectable
      - expression cmap selectable; same cmap provides the "positive" color on the mask panel
      - bbox crop support: (x0,y0,x1,y1) in full-res image coordinates
      - downscale (integer) to reduce memory + speed up plotting (default 4)
      - robust channel name matching (optional): handles case/hyphens etc.

    Requirements (matches your sp_object layout):
      - sp_object['_image'] dims: ('channels','y','x')
      - sp_object['_segmentation'] dims: ('y','x') integer labels
      - sp_object.pp.get_layer_as_df() provides marker bin columns like '{marker}_binarized'
        and has numeric segmentation ids either as the dataframe index (recommended) or a column
        like 'label'/'cell_id'.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Get cell-level table
    layer_df = sp_object.pp.get_layer_as_df()

    # Numeric segmentation IDs aligned with rows
    label_ids = _get_numeric_label_ids(layer_df)

    # Image cube and segmentation
    img_cube = sp_object["_image"]           # dims: ('channels','y','x')
    seg = sp_object["_segmentation"]         # dims: ('y','x')

    # Channel coordinate values
    channel_vals = [str(x) for x in img_cube.coords["channels"].values]
    channel_map_exact = {c: c for c in channel_vals}
    channel_map_lower = {c.lower(): c for c in channel_vals}
    channel_map_norm = {_norm_name(c): c for c in channel_vals}

    # For binarized columns, make a normalized map too
    df_cols = list(layer_df.columns)
    df_col_norm_map = {_norm_name(c): c for c in df_cols}

    # Color choices
    cmap_obj = plt.get_cmap(expr_cmap)
    pos_rgb = np.array(cmap_obj(0.75)[:3], dtype=np.float32)   # "strong-ish" cmap color
    neg_rgb = np.array([grey_level, grey_level, grey_level], dtype=np.float32)

    saved_paths: List[str] = []

    for marker, celltype in marker2celltype.items():
        # --- Resolve channel name in '_image' ---
        ch = None
        if marker in channel_map_exact:
            ch = marker
        elif marker.lower() in channel_map_lower:
            ch = channel_map_lower[marker.lower()]
        elif match_names_fuzzily and _norm_name(marker) in channel_map_norm:
            ch = channel_map_norm[_norm_name(marker)]

        if ch is None:
            if verbose:
                print(f"Skipping {marker}->{celltype}: marker not found in sp_object['_image'].coords['channels']")
            continue

        # --- Resolve binarized column in layer_df ---
        #wanted_bin = f"{marker}_binarized"
        wanted_bin = celltype

        bin_col = None

        if wanted_bin in layer_df.columns:
            bin_col = wanted_bin
        else:
            # try channel name version
            alt = f"{ch}_binarized"
            if alt in layer_df.columns:
                bin_col = alt
            elif match_names_fuzzily:
                # fuzzy: normalize and look up
                norm_wanted = _norm_name(wanted_bin)
                if norm_wanted in df_col_norm_map:
                    bin_col = df_col_norm_map[norm_wanted]
                else:
                    # maybe the df used normalized channel name
                    norm_alt = _norm_name(alt)
                    if norm_alt in df_col_norm_map:
                        bin_col = df_col_norm_map[norm_alt]

        if bin_col is None:
            if verbose:
                print(f"Skipping {marker}->{celltype}: no '{marker}_binarized' column found in get_layer_as_df()")
            continue

        # --- Load 2D expression image for this channel (pull only one channel) ---
        expr_img = np.asarray(img_cube.sel(channels=ch).values)   # (y,x) uint8
        seg_img = np.asarray(seg.values)                          # (y,x) int32

        # --- Optional crop ---
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            expr_img = expr_img[y0:y1, x0:x1]
            seg_img = seg_img[y0:y1, x0:x1]

        # --- Downscale expression and normalize for display ---
        expr_ds = _downscale_image_mean(expr_img.astype(np.float32), downscale)
        vmin = 0.0
        vmax = float(np.percentile(expr_ds, vmax_percentile)) if expr_ds.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0

        # --- Build positive-cell mask using segmentation IDs ---
        pos_rows = layer_df[layer_df[bin_col].astype(bool)]
        pos_label_ids = label_ids.loc[pos_rows.index].unique()

        if pos_label_ids.size == 0:
            mask = np.zeros_like(seg_img, dtype=np.uint8)
        else:
            mask = np.isin(seg_img, pos_label_ids).astype(np.uint8)

        mask_ds = _downscale_mask_max(mask, downscale)

        # --- Turn mask into a 2-color RGB image (grey vs cmap color) ---
        mask_rgb = np.empty((mask_ds.shape[0], mask_ds.shape[1], 3), dtype=np.float32)
        mask_rgb[:] = neg_rgb
        mask_rgb[mask_ds.astype(bool)] = pos_rgb

        # --- Plot ---
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        ax1, ax2 = axes

        im = ax1.imshow(expr_ds, cmap=expr_cmap, vmin=vmin, vmax=vmax)
        ax1.set_title(f"{ch} expression")
        ax1.axis("off")
        if add_colorbar:
            fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        ax2.imshow(mask_rgb)
        ax2.set_title(f"{celltype} positives ({marker})")
        ax2.axis("off")

        if roi_name:
            ax1.set_title(f"{roi_name}_{ch} expression")
            ax2.set_title(f"{roi_name}_{celltype} positives ({marker})")

        fig.tight_layout()
        if roi_name:
            out_path = os.path.join(out_dir, f"{roi_name}_{marker}__{celltype}.png")
        else:
            out_path = os.path.join(out_dir, f"{marker}__{celltype}.png")
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)

        saved_paths.append(out_path)

    return saved_paths
