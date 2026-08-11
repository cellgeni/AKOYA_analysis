import spatialproteomics as sp
from skimage.io import imread
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
from scipy.ndimage import find_objects
import pandas as pd
import yaml
import os
from typing import Any, Dict, List, Optional, Tuple, Union
import fire
import gc
import anndata as ad
import xarray as xr
from plots_AKOYA import plot_marker_celltype_pairs_from_spobject
from plots_preprocessing import plot_save_preprocessing_genes, plot_save_segmentation_masks, plot_save_area_hist, save_whole_tissue_dapi
from pathlib import Path
import nd2

RAW_IMAGE_KEY = "_image_raw"
NUCLEI_SEGMENTATION_KEY = "_segmentation_nuclei"
CYTOPLASM_SEGMENTATION_KEY = "_segmentation_cytoplasm"


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
    manual_marker_thresholds = data.get('manual_marker_thresholds', data.get('manual_thresholds', {}))
    split_signal_nuclei_cytoplasm = data.get(
        'split_signal_nuclei_cytoplasm',
        data.get('split_signal_on_nuclei_cytoplasm', False)
    )
    intensity_quantiles = validate_intensity_quantiles(
        data.get('intensity_quantiles', [0.5, 0.75, 0.9, 0.95])
    )
    positive_cell_rules = _optional_mapping(
        data.get('positive_cell_rules', {}),
        "positive_cell_rules",
    )
    stardist_scale = float(data.get('stardist_scale', 3))
    if stardist_scale <= 0:
        raise ValueError("stardist_scale must be greater than 0.")
    save_omero_segmentation_csv = bool(data.get('save_omero_segmentation_csv', False))
    return (image_path, crop_x, crop_y, channel_segment, list_of_channels, list_of_markers, 
            label_expansion, min_area, max_area, save_intermediate_plots, N_intermediate_plots, 
            S_intermediate_plots, save_binary_plots, threshold_binary, output_dir, 
            list_of_genes_intermediate_plots, normalise_intensity, save_intermediate_zarr, 
            list_output_formats, pixelsize, manual_marker_thresholds, split_signal_nuclei_cytoplasm,
            intensity_quantiles, positive_cell_rules, stardist_scale,
            save_omero_segmentation_csv)


def normalize_intensities(sp_object):
    X = sp_object["_intensity"]  # dims: (cells, channels)
    mu = X.mean("cells")
    sd = X.std("cells")
    X_norm = (X - mu) / (sd + 1e-8)
    # I save norm intensiy in _intensity and original one in _intensity_raw
    sp_object = sp_object.assign(_intensity_raw=sp_object["_intensity"])
    sp_object = sp_object.assign(_intensity=X_norm)
    return sp_object


def _optional_mapping(value, name):
    if value is None or value is False:
        return {}
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "false"}:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a dictionary mapping marker/channel names to threshold values.")
    return value


def validate_intensity_quantiles(values):
    """Validate and de-duplicate configured per-cell pixel quantiles."""
    if values is None:
        values = [0.5, 0.75, 0.9, 0.95]
    if not isinstance(values, (list, tuple)):
        raise TypeError("intensity_quantiles must be a list of numbers between 0 and 1.")

    quantiles = []
    for raw_value in values:
        value = float(raw_value)
        if not 0 <= value <= 1:
            raise ValueError(f"Intensity quantile {value} is outside the allowed range [0, 1].")
        if value not in quantiles:
            quantiles.append(value)
    return quantiles


def _quantile_label(value):
    return format(float(value), ".12g")


def _stat_column(channel, compartment, metric):
    """Canonical AnnData obs column name used by metrics and positivity rules."""
    return f"{channel}__{compartment}__{metric}"


def resolve_preprocessing_thresholds(img, list_of_channels, manual_marker_thresholds=None):
    """
    Build per-channel preprocessing thresholds.

    Channels absent from manual_marker_thresholds keep the default Otsu threshold.
    Manual values < 1 are interpreted as channel quantiles; values >= 1 are
    interpreted as absolute image intensities.
    """
    manual_marker_thresholds = _optional_mapping(manual_marker_thresholds, "manual_marker_thresholds")
    thresholds = np.asarray(sp.pp.otsu_per_channel(img, channel_axis=0), dtype=float)
    if not manual_marker_thresholds:
        return thresholds

    channel_to_idx = {str(channel): idx for idx, channel in enumerate(list_of_channels)}
    missing = sorted(set(manual_marker_thresholds) - set(channel_to_idx))
    if missing:
        raise KeyError(
            "Manual threshold channel(s) not found in list_of_channels: "
            + ", ".join(str(x) for x in missing)
        )

    for marker, raw_threshold in manual_marker_thresholds.items():
        threshold = float(raw_threshold)
        if threshold < 0:
            raise ValueError(f"Manual threshold for {marker} must be >= 0.")

        channel_idx = channel_to_idx[str(marker)]
        channel_img = img[channel_idx]
        if threshold < 1:
            thresholds[channel_idx] = float(np.quantile(channel_img, threshold))
            print(f"Using manual relative threshold for {marker}: quantile {threshold} -> {thresholds[channel_idx]}")
        else:
            channel_max = float(np.nanmax(channel_img))
            if threshold > channel_max:
                raise ValueError(
                    f"Manual absolute threshold for {marker} ({threshold}) is above the channel max ({channel_max})."
                )
            thresholds[channel_idx] = threshold
            print(f"Using manual absolute threshold for {marker}: {threshold}")

    return thresholds


def _filter_segmentation_to_cells(segmentation, cells):
    segmentation = np.asarray(segmentation).copy()
    segmentation[~np.isin(segmentation, cells)] = 0
    return segmentation


def _assign_segmentation_layer(sp_object, key, segmentation):
    da = xr.DataArray(
        segmentation.astype(sp_object["_segmentation"].dtype, copy=False),
        coords=[sp_object.coords["y"], sp_object.coords["x"]],
        dims=["y", "x"],
        name=key,
    )
    if key in sp_object:
        sp_object = sp_object.drop_vars(key)
    return xr.merge([sp_object, da], join="outer", compat="no_conflicts")


def add_compartment_segmentations(sp_object, nuclei_segmentation, include_cytoplasm):
    cells = sp_object.coords["cells"].values
    nuclei_segmentation = _filter_segmentation_to_cells(nuclei_segmentation, cells)
    sp_object = _assign_segmentation_layer(sp_object, NUCLEI_SEGMENTATION_KEY, nuclei_segmentation)

    if include_cytoplasm:
        cytoplasm_segmentation = sp_object["_segmentation"].values.copy()
        cytoplasm_segmentation[nuclei_segmentation > 0] = 0
        cytoplasm_segmentation = _filter_segmentation_to_cells(cytoplasm_segmentation, cells)
        sp_object = _assign_segmentation_layer(sp_object, CYTOPLASM_SEGMENTATION_KEY, cytoplasm_segmentation)

    return sp_object


def compute_cell_statistics(image, segmentation, channels, cells, quantiles):
    """Compute raw processed-pixel summaries for every cell and channel in one pass."""
    index = pd.Index(cells, name="cells")
    metric_names = ["mean", "median", "std", "variance", "percentage_positive"]
    metric_names.extend(f"quantile_{_quantile_label(q)}" for q in quantiles)
    values = {
        metric: np.full((len(index), len(channels)), np.nan, dtype=np.float64)
        for metric in metric_names
    }
    if not np.any(segmentation > 0):
        return {
            metric: pd.DataFrame(array, index=index, columns=channels)
            for metric, array in values.items()
        }

    label_to_row = {int(label): row for row, label in enumerate(index)}
    image_yxc = np.moveaxis(np.asarray(image), 0, -1)
    segmentation = np.asarray(segmentation)
    object_slices = find_objects(segmentation)
    for label, row in label_to_row.items():
        if label <= 0 or label > len(object_slices):
            continue
        region_slice = object_slices[label - 1]
        if region_slice is None:
            continue
        region_labels = segmentation[region_slice]
        pixels = image_yxc[region_slice][region_labels == label]
        if pixels.ndim == 1:
            pixels = pixels[:, None]
        values["mean"][row] = np.mean(pixels, axis=0)
        values["median"][row] = np.median(pixels, axis=0)
        values["std"][row] = np.std(pixels, axis=0, ddof=0)
        values["variance"][row] = np.var(pixels, axis=0, ddof=0)
        values["percentage_positive"][row] = np.mean(pixels > 0, axis=0)
        for quantile in quantiles:
            values[f"quantile_{_quantile_label(quantile)}"][row] = np.quantile(
                pixels,
                quantile,
                axis=0,
            )

    return {
        metric: pd.DataFrame(array, index=index, columns=channels)
        for metric, array in values.items()
    }


def compute_all_compartment_results(sp_object, quantiles, include_nuclei_cytoplasm):
    image = sp_object["_image"].values
    channels = [str(c) for c in sp_object.coords["channels"].values]
    cells = sp_object.coords["cells"].values
    segmentations = {"whole_cell": "_segmentation"}
    if include_nuclei_cytoplasm:
        segmentations.update({
            "nuclei": NUCLEI_SEGMENTATION_KEY,
            "cytoplasm": CYTOPLASM_SEGMENTATION_KEY,
        })

    results = {}
    for compartment, seg_key in segmentations.items():
        if seg_key not in sp_object:
            continue
        results[compartment] = {
            "metrics": compute_cell_statistics(
                image,
                sp_object[seg_key].values,
                channels,
                cells,
                quantiles,
            )
        }
    return results


def statistics_to_obs_dataframe(results):
    frames = []
    for compartment, result in results.items():
        for metric, metric_df in result["metrics"].items():
            frame = metric_df.copy()
            frame.columns = [
                _stat_column(channel, compartment, metric)
                for channel in frame.columns
            ]
            frames.append(frame)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()


def _select_positive_rule(marker, compartment, positive_cell_rules, default_threshold):
    rule = positive_cell_rules.get(marker)
    if rule is None:
        return {"metric": "percentage_positive", "threshold": default_threshold}
    if isinstance(rule, (int, float)):
        return {"metric": "percentage_positive", "threshold": float(rule)}
    if not isinstance(rule, dict):
        raise TypeError(f"positive_cell_rules[{marker!r}] must be a number or dictionary.")

    rule_fields = {"column", "metric", "threshold", "operator"}
    if rule_fields.intersection(rule):
        selected = rule
    else:
        aliases = [compartment]
        if compartment == "whole_cell":
            aliases.extend(["whole", "both"])
        selected = next((rule[key] for key in aliases if key in rule), rule.get("default"))
        if selected is None:
            return {"metric": "percentage_positive", "threshold": default_threshold}
        if isinstance(selected, (int, float)):
            selected = {"metric": "percentage_positive", "threshold": float(selected)}
        if not isinstance(selected, dict):
            raise TypeError(
                f"Rule for marker {marker!r}, compartment {compartment!r} must be a number or dictionary."
            )

    return dict(selected)


def _apply_threshold(series, threshold, operator):
    operators = {
        ">=": lambda x: x >= threshold,
        ">": lambda x: x > threshold,
        "<=": lambda x: x <= threshold,
        "<": lambda x: x < threshold,
    }
    if operator not in operators:
        raise ValueError(f"Unsupported positivity operator {operator!r}; use one of {sorted(operators)}.")
    return operators[operator](series).fillna(False).astype(np.int8)


def apply_positive_cell_rules(
    results,
    source_obs,
    markers,
    positive_cell_rules,
    default_threshold,
):
    """Create compartment-specific binary calls from configured obs columns."""
    unknown = sorted(set(positive_cell_rules) - set(markers))
    if unknown:
        raise KeyError("Positive-cell rule marker(s) not found in list_of_markers: " + ", ".join(unknown))

    metadata = {}
    for compartment, result in results.items():
        binary = pd.DataFrame(index=source_obs.index)
        metadata[compartment] = {}
        for marker in markers:
            rule = _select_positive_rule(
                marker,
                compartment,
                positive_cell_rules,
                default_threshold,
            )
            threshold = float(rule.get("threshold", default_threshold))
            operator = str(rule.get("operator", ">="))
            if "column" in rule:
                column = str(rule["column"]).format(
                    marker=marker,
                    channel=marker,
                    compartment=compartment,
                )
            else:
                metric = str(rule.get("metric", "percentage_positive"))
                aliases = {
                    "intensity_mean": "mean",
                    "intensity_median": "median",
                    "standard_deviation": "std",
                    "intensity_std": "std",
                    "var": "variance",
                    "intensity_variance": "variance",
                }
                metric = aliases.get(metric, metric)
                column = _stat_column(marker, compartment, metric)
            if column not in source_obs.columns:
                raise KeyError(
                    f"Positive-cell rule for {marker!r} ({compartment}) references missing obs column {column!r}."
                )
            binary[f"{marker}_positive"] = _apply_threshold(
                pd.to_numeric(source_obs[column], errors="coerce"),
                threshold,
                operator,
            )
            metadata[compartment][marker] = {
                "column": column,
                "threshold": threshold,
                "operator": operator,
            }
        result["binary"] = binary
    return metadata


def _assign_matrix_layer(sp_object, key, df):
    cells = sp_object.coords["cells"].values
    channels = [str(c) for c in sp_object.coords["channels"].values]
    da = xr.DataArray(
        df.loc[cells, channels].values,
        coords=[sp_object.coords["cells"], sp_object.coords["channels"]],
        dims=["cells", "channels"],
        name=key,
    )
    if key in sp_object:
        sp_object = sp_object.drop_vars(key)
    return xr.merge([sp_object, da], join="outer", compat="no_conflicts")


def add_compartment_layers_to_spobject(sp_object, compartment_results):
    for compartment, result in compartment_results.items():
        suffix = "" if compartment == "whole_cell" else f"_{compartment}"
        for metric, metric_df in result["metrics"].items():
            if metric == "percentage_positive":
                key = f"_percentage_positive{suffix}"
            elif metric == "mean":
                key = f"_intensity_mean_pixels{suffix}"
            else:
                safe_metric = metric.replace(".", "_")
                key = f"_intensity_{safe_metric}{suffix}"
            sp_object = _assign_matrix_layer(sp_object, key, metric_df)

        if compartment != "whole_cell":
            transformed_mean = np.arcsinh(result["metrics"]["mean"] / 5.0)
            sp_object = _assign_matrix_layer(
                sp_object,
                f"_intensity_{compartment}",
                transformed_mean,
            )
    return sp_object


def add_analysis_results_to_anndata(
    adata,
    compartment_results,
    statistics_obs,
    var_names,
    quantiles,
    positivity_metadata,
):
    if not compartment_results:
        return

    aligned_statistics = statistics_obs.copy()
    aligned_statistics.index = aligned_statistics.index.map(str)
    aligned_statistics = aligned_statistics.reindex(adata.obs_names)
    aligned_statistics = aligned_statistics.astype(np.float32)
    adata.obs = pd.concat([adata.obs, aligned_statistics], axis=1)

    compartment_summary = {}
    for compartment, result in compartment_results.items():
        intensity = np.arcsinh(result["metrics"]["mean"] / 5.0).copy()
        intensity.index = intensity.index.map(str)
        intensity = intensity.reindex(index=adata.obs_names, columns=var_names)

        layer_key = None
        if compartment != "whole_cell":
            layer_key = f"{compartment}_intensity"
            adata.layers[layer_key] = intensity.to_numpy(dtype=np.float32)

        binary = result["binary"].copy()
        binary.index = binary.index.map(str)
        binary = binary.reindex(index=adata.obs_names).fillna(0).astype(np.int8)

        positive_obs = binary.rename(columns={
            column: f"{column.removesuffix('_positive')}__{compartment}__positive"
            for column in binary.columns
        })
        adata.obs = pd.concat([adata.obs, positive_obs], axis=1)

        obsm_key = f"marker_presence_{compartment}"
        adata.obsm[obsm_key] = binary

        compartment_summary[compartment] = {
            "intensity_layer": layer_key,
            "marker_presence_obsm": obsm_key,
            "marker_presence_columns": list(binary.columns),
        }

    adata.uns["akoya_compartments"] = compartment_summary
    adata.uns["akoya_intensity_quantiles"] = np.asarray(quantiles, dtype=float)
    adata.uns["akoya_positive_cell_rules"] = positivity_metadata


def read_crop_cyx(path, crop_x, crop_y):
    """
    Read tif/qptiff/nd2, try to determine axes, return cropped image in CYX order.

    - Uses OME/series axes metadata if available for tif/qptiff.
    - Uses nd2 axes metadata for nd2 files.
    - If metadata missing and image is 3D, assumes the smallest dim is channels (warns).
    - Supports 2D (YX) and 3D (CYX/YXC) inputs.
    """

    path = Path(path)

    # --- read image ---
    if path.suffix.lower() == ".nd2":
        with nd2.ND2File(path) as f:
            arr = f.asarray()
            axes = "".join(f.sizes.keys())
    else:
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
            raise ValueError(
                f"Unsupported axes from metadata: {axes} "
                f"(only YX, CYX, YXC supported)."
            )

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


def write_tables_csv(sp_object, outdir):
    os.makedirs(outdir, exist_ok=True)
    SKIP = {"_image", RAW_IMAGE_KEY, "_segmentation", NUCLEI_SEGMENTATION_KEY, CYTOPLASM_SEGMENTATION_KEY}  # huge
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


def _polygon_wkt_from_label(segmentation, label, region_slice=None):
    """Return the outer boundary of one label as OMERO-compatible polygon WKT."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "OMERO segmentation CSV export requires opencv-python (cv2)."
        ) from exc

    if region_slice is None:
        region_slice = (slice(0, segmentation.shape[0]), slice(0, segmentation.shape[1]))
    mask = (segmentation[region_slice] == label).astype(np.uint8)
    contour_result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contour_result[-2]  # compatible with OpenCV 3 and 4
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        return None
    contour[:, 0] += region_slice[1].start  # x offset
    contour[:, 1] += region_slice[0].start  # y offset
    contour = np.vstack([contour, contour[0]])  # WKT polygon rings are closed
    points = ", ".join(f"{int(x)} {int(y)}" for x, y in contour)
    return f"POLYGON (({points}))"


def write_omero_segmentation_csvs(sp_object, adata, outdir, markers):
    """Write OMERO polygon CSVs for whole-cell and available compartment masks."""
    os.makedirs(outdir, exist_ok=True)
    reserved_columns = {"object", "label", "score", "confidence_score", "polygon"}
    conflicting_obs_columns = reserved_columns.intersection(adata.obs.columns)
    if conflicting_obs_columns:
        raise ValueError(
            "Cannot create OMERO CSV because adata.obs already contains reserved OMERO column(s): "
            + ", ".join(sorted(conflicting_obs_columns))
        )

    segmentation_layers = {
        "whole_cell": "_segmentation",
        "nuclei": NUCLEI_SEGMENTATION_KEY,
        "cytoplasm": CYTOPLASM_SEGMENTATION_KEY,
    }
    obs = adata.obs.copy()
    obs.index = obs.index.map(str)
    saved_paths = []

    for compartment, segmentation_key in segmentation_layers.items():
        if segmentation_key not in sp_object:
            continue
        segmentation = np.asarray(sp_object[segmentation_key].values)
        labels = np.unique(segmentation)
        labels = labels[labels > 0]
        object_slices = find_objects(segmentation)

        rows = []
        for label_id in labels:
            object_id = int(label_id)
            object_key = str(object_id)
            if object_key not in obs.index:
                continue
            if object_id > len(object_slices) or object_slices[object_id - 1] is None:
                continue
            polygon = _polygon_wkt_from_label(
                segmentation,
                object_id,
                region_slice=object_slices[object_id - 1],
            )
            if polygon is None:
                continue

            row = {
                "object": object_id,
                "label": compartment,
                "score": 1.0,
                "confidence_score": 1.0,
                "polygon": polygon,
            }
            row.update(obs.loc[object_key].to_dict())
            for marker in markers:
                positive_column = f"{marker}__{compartment}__positive"
                is_positive = bool(row.get(positive_column, False))
                row[f"label-{marker}"] = "positive" if is_positive else "negative"
            rows.append(row)

        columns = ["object", "label", "score", "confidence_score", "polygon"]
        columns.extend(adata.obs.columns.tolist())
        columns.extend(f"label-{marker}" for marker in markers)
        omero_df = pd.DataFrame(rows, columns=columns)
        path = os.path.join(outdir, f"omero_segmentation_{compartment}.csv")
        omero_df.to_csv(path, index=False)
        print(f"Saved OMERO segmentation CSV at {path}  shape={omero_df.shape}")
        saved_paths.append(path)

    if not saved_paths:
        print("No segmentation masks were available for OMERO CSV export.")
    return saved_paths


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
    compartment_results: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    statistics_obs: Optional[pd.DataFrame] = None,
    intensity_quantiles: Optional[List[float]] = None,
    positivity_metadata: Optional[Dict[str, Any]] = None,
    write_h5ad: bool = True,
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

    add_analysis_results_to_anndata(
        adata,
        compartment_results or {},
        statistics_obs if statistics_obs is not None else pd.DataFrame(index=adata.obs_names),
        var_names,
        intensity_quantiles or [],
        positivity_metadata or {},
    )

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
    if write_h5ad:
        adata.write_h5ad(out_path)
        print(f"Saved anndata object at {out_path}")
    return adata, out_path

def main(ConfFilePath):
    (image_path, crop_x, crop_y, channel_segment, list_of_channels, list_of_markers, label_expansion, min_area, 
    max_area, save_intermediate_plots, N_intermediate_plots, S_intermediate_plots, save_binary_plots, threshold_binary, 
    output_dir, list_of_genes_intermediate_plots, normalise_intensity, save_intermediate_zarr, 
    list_output_formats, pixelsize, manual_marker_thresholds, split_signal_nuclei_cytoplasm,
    intensity_quantiles, positive_cell_rules, stardist_scale,
    save_omero_segmentation_csv) = ReadConfFile(ConfFilePath)

    print('Output dir' + str(output_dir))
    path_zarr = os.path.join(output_dir,"sp_object.zarr")
    
    print('Reading the image')
    img = read_crop_cyx(image_path, crop_x, crop_y)
    image_size = img[0].shape
    print(image_size)
    sp_object = sp.load_image_data(img, channel_coords=list_of_channels)
    raw_image_layer = sp_object["_image"] if save_binary_plots else None
    print(sp_object)
    if save_intermediate_plots:
        os.makedirs(output_dir, exist_ok=True)
        path1 = os.path.join(output_dir, 'whole_tissue_DAPI.png')
        save_whole_tissue_dapi(img, path1, downscale_factor=10)
    gc.collect()


    print('Image preprocesing')
    thrs_list = resolve_preprocessing_thresholds(img, list_of_channels, manual_marker_thresholds)
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
    print(f"Using StarDist scale={stardist_scale}")
    sp_object = sp_object.tl.stardist(channel=channel_segment, scale=stardist_scale)
    nuclei_segmentation = sp_object["_segmentation"].values.copy() if split_signal_nuclei_cytoplasm else None
    if label_expansion:
        sp_object = sp_object.pp.expand_segmentation(radius = label_expansion)
    sp_object = sp_object.pp.add_observations("area")
    plot_save_area_hist(sp_object, output_dir)
    if min_area: 
        sp_object = sp_object.pp.filter_by_obs("area", lambda x: x > min_area)
    if max_area: 
        sp_object = sp_object.pp.filter_by_obs("area", lambda x: x < max_area)
    if split_signal_nuclei_cytoplasm:
        sp_object = add_compartment_segmentations(
            sp_object,
            nuclei_segmentation,
            include_cytoplasm=bool(label_expansion),
        )
        if not label_expansion:
            print("Cytoplasm signal was requested but segmentation_label_expansion is not set; saving nuclei only.")
    if save_intermediate_plots:
        plot_save_segmentation_masks(sp_object, subplots_pos, S_intermediate_plots, output_dir)
    gc.collect()
    if save_intermediate_zarr:
        sp_object.to_zarr(path_zarr, mode="w", zarr_version=2, consolidated=True)
    
    print("Computing per-cell intensity statistics and marker presence")
    sp_object = sp_object.pp.add_quantification(func="intensity_mean").pp.transform_expression_matrix(method="arcsinh")
    if normalise_intensity:
        sp_object = normalize_intensities(sp_object)
    compartment_results = compute_all_compartment_results(
        sp_object,
        intensity_quantiles,
        split_signal_nuclei_cytoplasm,
    )
    statistics_obs = statistics_to_obs_dataframe(compartment_results)
    base_obs = sp_object.pp.get_layer_as_df().reindex(statistics_obs.index)
    positivity_source_obs = pd.concat([base_obs, statistics_obs], axis=1)
    positivity_metadata = apply_positive_cell_rules(
        compartment_results,
        positivity_source_obs,
        list_of_markers,
        positive_cell_rules,
        threshold_binary,
    )
    sp_object = add_compartment_layers_to_spobject(sp_object, compartment_results)

    # Preserve the historical *_ct columns when the legacy default rules are used.
    if not positive_cell_rules:
        threshold_dict = {k: threshold_binary for k in list_of_markers}
        sp_object = sp_object.la.threshold_labels(threshold_dict, layer_key="_percentage_positive")
    else:
        print("Custom positive_cell_rules are active; use the *__positive AnnData obs columns for calls.")
    mapping_dict = {k: f"{k}_positive" for k in list_of_markers}
    if save_binary_plots:
        if raw_image_layer is not None:
            sp_object = sp_object.assign({RAW_IMAGE_KEY: raw_image_layer})
        save_folder = os.path.join(output_dir, 'marker_genes vs celltype - whole tissue')
        plot_marker_celltype_pairs_from_spobject(
            sp_object,
            mapping_dict,
            out_dir=save_folder,
            dpi=300,
            expr_cmap="Reds",
            binary_df=compartment_results["whole_cell"]["binary"],
            positive_source_label="configured whole-cell rule",
            bbox=None,
            downscale=4,
        )
        if save_intermediate_plots:
            mapping_dict2 = {k: f"{k}_positive" for k in list_of_genes_intermediate_plots}
            save_folder2 = os.path.join(output_dir, 'marker_genes vs celltype - ROIs')
            for i in range(subplots_pos.shape[0]):
                x0 = subplots_pos[i][1]; y0 = subplots_pos[i][0];
                roi_name = "subplot_" + str(i)
                plot_marker_celltype_pairs_from_spobject(
                    sp_object,
                    mapping_dict2,
                    out_dir=save_folder2,
                    dpi=300,
                    expr_cmap="Reds",
                    binary_df=compartment_results["whole_cell"]["binary"],
                    positive_source_label="configured whole-cell rule",
                    bbox=[x0, y0, x0+S_intermediate_plots, y0+S_intermediate_plots],
                    downscale=1,
                    roi_name=roi_name,
                )
        if split_signal_nuclei_cytoplasm and compartment_results:
            compartment_plot_specs = {
                "nuclei": (NUCLEI_SEGMENTATION_KEY, "nuclei signal"),
                "cytoplasm": (CYTOPLASM_SEGMENTATION_KEY, "cytoplasmic signal"),
            }
            for compartment, (segmentation_key, source_label) in compartment_plot_specs.items():
                if compartment not in compartment_results or segmentation_key not in sp_object:
                    continue

                save_folder = os.path.join(
                    output_dir,
                    f"marker_genes vs celltype - {compartment} whole tissue",
                )
                plot_marker_celltype_pairs_from_spobject(
                    sp_object,
                    mapping_dict,
                    out_dir=save_folder,
                    dpi=300,
                    expr_cmap="Reds",
                    segmentation_key=segmentation_key,
                    binary_df=compartment_results[compartment]["binary"],
                    positive_source_label=source_label,
                    bbox=None,
                    downscale=4,
                )

                if save_intermediate_plots:
                    save_folder2 = os.path.join(
                        output_dir,
                        f"marker_genes vs celltype - {compartment} ROIs",
                    )
                    for i in range(subplots_pos.shape[0]):
                        x0 = subplots_pos[i][1]; y0 = subplots_pos[i][0];
                        roi_name = "subplot_" + str(i)
                        plot_marker_celltype_pairs_from_spobject(
                            sp_object,
                            mapping_dict2,
                            out_dir=save_folder2,
                            dpi=300,
                            expr_cmap="Reds",
                            segmentation_key=segmentation_key,
                            binary_df=compartment_results[compartment]["binary"],
                            positive_source_label=source_label,
                            bbox=[x0, y0, x0+S_intermediate_plots, y0+S_intermediate_plots],
                            downscale=1,
                            roi_name=roi_name,
                        )
        if RAW_IMAGE_KEY in sp_object:
            sp_object = sp_object.drop_vars(RAW_IMAGE_KEY)

    print("Saving")
    if 'zarr' in list_output_formats:
        sp_object.to_zarr(path_zarr, mode="w", zarr_version=2, consolidated=True)
        print(f"Saved spatialproteomics object at {path_zarr}")
    if 'csv' in list_output_formats:
        save_folder_csv = os.path.join(output_dir, 'tables_csv')
        write_tables_csv(sp_object, save_folder_csv)
    if 'h5ad' in list_output_formats or save_omero_segmentation_csv:
        adata, _ = spobject_to_anndata(
            sp_object,
            out_dir=output_dir,
            sample_id="anndata",
            image_downsample=10,
            image_channels=["DAPI"],
            spot_diameter_fullres=20.0,
            pixel_size_um=pixelsize,
            compartment_results=compartment_results,
            statistics_obs=statistics_obs,
            intensity_quantiles=intensity_quantiles,
            positivity_metadata=positivity_metadata,
            write_h5ad='h5ad' in list_output_formats,
        )
        if save_omero_segmentation_csv:
            write_omero_segmentation_csvs(
                sp_object,
                adata,
                os.path.join(output_dir, "omero_segmentation_csv"),
                list_of_markers,
            )
 
        
if __name__ == "__main__":
    fire.Fire(main) 
