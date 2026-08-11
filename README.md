# Introduction 

AKOYA pipeline automates low-level spatial proteomics processing from a multichannel image: preprocessing, segmentation, per-cell intensity extraction, and positive/negative marker calls. It is built mostly using [spatialproteomics_cellgeni](https://github.com/cellgeni/spatialproteomics), which was adapted for large images (original package: [spatialproteomics](https://github.com/sagar87/spatialproteomics)). Outputs can include (i) a spatialproteomics object in Zarr format; (ii) CSV tables containing cell positions, per-cell intensity summaries, and marker-presence calls; and (iii) an AnnData file. `adata.X` remains the transformed mean-intensity matrix for backwards compatibility, while the additional per-cell statistics are stored in `adata.obs`. See **open_output_files.ipynb** for examples.


# Environment
 Create Docker container using [Dockerfile](https://github.com/cellgeni/AKOYA_analysis/blob/main/container/Dockerfile). If you run it on farm, please use singularity container : `/nfs/cellgeni/singularity/images/spatialproteomics.sif`

# Run whole pipeline
## Prepare configuration file
All parameters used in pipeline together with input/output paths should be specified in one configuration file (see as example **conf_AKOYA.yaml**). Below is description of all par-s for configuration file are described below


### Input image and channels

`image_path (str)` - Path to the multiplex image file (e.g. .tif / .qptiff) to be processed

'list_of_channels (list[str])' - Channel names in the same order as the channel axis of the loaded image

`channel_for_segmentation (str)` - Which channel to use for StarDist segmentation. Typically the DAPI channel name

`list_of_markers (list[str])` - Marker channels used to compute % positive pixels and generate binary labels / cell-type-like labels via thresholding.

### Cropping

`crop_x (list[int, int])` - X-range to crop as [x_start, x_end].

`crop_y (list[int, int])` - Y-range to crop as [y_start, y_end].

### Segmentation + filtering

`segmentation_label_expansion (int or falsy)` - If truthy, expands segmentation labels by this radius (pixels) using expand_segmentation

`stardist_scale (float, optional)` - Image scale passed to StarDist (default: `3`, preserving the original pipeline behaviour). For cells that are split into multiple objects, try a smaller value such as `1`, `0.75`, or `0.5`; this reduces the apparent cell size seen by the model. This option changes segmentation itself, whereas `segmentation_label_expansion` only expands masks after segmentation.

`min_area (int or falsy)` - If truthy, filters out segmented objects with area <= min_area.

`max_area (int or falsy)` - If truthy, filters out segmented objects with area >= max_area.

### Intermediate plots

`save_intermediate_plots (bool)` - Whether to save intermediate QC plots and ROI snapshots.

`number_intermediate_plots (int)` - Number of random ROIs (subimages) to sample for intermediate plotting.

`size_intermediate_plots (int)` - ROI size in pixels (square). Each ROI is size_intermediate_plots × size_intermediate_plots.

`list_of_genes_intermediate_plots (list[str])` - Channel names to display in intermediate ROI plots (alongside DAPI).

Marker-vs-celltype plots saved by `save_individual_marker_presence_plots` include raw marker expression before preprocessing, processed marker expression after thresholding/filtering, and the called positive cells.

### Binary marker presence / label thresholding

`save_individual_marker_presence_plots (bool)` - Whether to plot marker-vs-celltype (binary label) maps.

`fraction_of_positive_pixels (float)` - Backwards-compatible default threshold applied to the fraction of non-zero pixels. It is used for markers/compartments without a matching entry in `positive_cell_rules`.

`manual_marker_thresholds (dict[str, float], optional)` - Per-channel overrides for image preprocessing thresholds, for example `{CD8: 1200, "PD-L1": 0.995}`. Channels not listed here keep the automatic Otsu threshold. Values below 1 are interpreted as relative channel quantiles; values greater than or equal to 1 are interpreted as absolute intensity thresholds.

`split_signal_nuclei_cytoplasm (bool, optional)` - If True, also quantifies nuclei and, when `segmentation_label_expansion` is set, cytoplasm. `whole_cell` means the expanded/combined cell mask (nucleus plus cytoplasm). Whole-cell mean intensity remains in `adata.X`; backwards-compatible transformed compartment means are written to `adata.layers["nuclei_intensity"]` and `adata.layers["cytoplasm_intensity"]`.

### Per-cell intensity statistics

Statistics are computed from processed pixels after image thresholding and median filtering. For every image channel and each available compartment (`whole_cell`, plus `nuclei` and `cytoplasm` when enabled), the pipeline writes these raw processed-pixel statistics to `adata.obs`:

- `mean`
- `median`
- `std` (population standard deviation, `ddof=0`)
- `variance` (population variance, `ddof=0`)
- `percentage_positive` (fraction of pixels greater than zero)
- every configured quantile

`intensity_quantiles (list[float], optional)` - Quantiles to calculate. Values must be between 0 and 1. The default is `[0.5, 0.75, 0.9, 0.95]`.

Columns use the unambiguous form `<channel>__<compartment>__<metric>`. For example:

```text
CD8__whole_cell__mean
CD8__whole_cell__quantile_0.9
CD8__nuclei__median
CD8__cytoplasm__variance
```

The same matrices are exported to Zarr/CSV with names such as `_intensity_median`, `_intensity_median_nuclei`, and `_intensity_quantile_0_9_cytoplasm`.

### Configurable positive/negative calls

`positive_cell_rules (dict, optional)` - Selects the source statistic and threshold independently for every marker. With `{}` (the default), all calls use `fraction_of_positive_pixels`, preserving the previous behaviour. Supported operators are `>=` (default), `>`, `<=`, and `<`.

A rule containing `metric` applies to all available compartments:

```yaml
positive_cell_rules:
  CD8:
    metric: quantile_0.9
    threshold: 25
  CD163:
    metric: median
    operator: ">"
    threshold: 12
```

Rules can differ between whole cell, nucleus, and cytoplasm:

```yaml
positive_cell_rules:
  PD-L1:
    whole_cell: {metric: median, threshold: 12}
    nuclei: {metric: mean, threshold: 8}
    cytoplasm: {metric: percentage_positive, threshold: 0.3}
```

You may also reference an exact `adata.obs` source column. `{marker}`, `{channel}`, and `{compartment}` placeholders are expanded automatically. Existing observation columns such as `area` can also be used:

```yaml
positive_cell_rules:
  CD8:
    column: "CD8__{compartment}__std"
    threshold: 4
  PD-L1:
    column: area
    operator: ">="
    threshold: 150
```

Calls are stored as integer 0/1 columns named `<marker>__<compartment>__positive` in `adata.obs`. They are also available as DataFrames in `adata.obsm["marker_presence_whole_cell"]`, `adata.obsm["marker_presence_nuclei"]`, and `adata.obsm["marker_presence_cytoplasm"]`. The resolved source column, operator, and threshold are recorded in `adata.uns["akoya_positive_cell_rules"]`. When no custom rules are supplied, the historical `*_ct` fields are retained as well.


### Output control

`output_dir (str)` - Path to directory where results and plots are written.

`list_output_formats (list[str])` - Which outputs to save. Supported values in this script: ["zarr", "h5ad", "csv"]

`save_intermediate_zarr (bool)` - If True, saves intermediate sp_object snapshots to `output_dir/sp_object.zarr` after key steps.

`save_omero_segmentation_csv (bool, optional)` - If True, writes OMERO-compatible polygon tables to `output_dir/omero_segmentation_csv/`. One CSV is generated for `whole_cell` and, when their masks are available, `nuclei` and `cytoplasm`. Each file begins with the OMERO fields `object`, `label`, `score`, `confidence_score`, and `polygon`; `polygon` contains a closed WKT string in the form `POLYGON ((x y, ...))`. All `adata.obs` columns are appended, together with one categorical `label-<marker>` column per marker containing `positive` or `negative` for that compartment.

### Optional metadata

`normalise_intensity (bool)` - If True, performs z-score normalization per channel across cells and stores:

`pixelsize (float or null, optional)` - Microns per full-resolution pixel. If provided, it is used only to create anndata h5ad object

## Run pipeline

The pipeline depending on image size requires signigicant amount of memory, so it is recommended for full-tissue crop (with ~(20k x 20k) pixels image and 60 channels) to use 200 Gb of memory or more. Example of submission code can be found in [submit_AKOYA_pipeline.sh](https://github.com/cellgeni/AKOYA_analysis/blob/main/templates/submit_AKOYA_pipeline.sh). Then one can submit a job simply as:

`bsub < submit_AKOYA_pipeline.sh`

## Run pipeline for many samples

In case you want to run the pipeline for many samples, the most annoying part is to prepare all separate configuration files. If you want to keep all parameters the same, you literally need to change in each conf file only "image_path" and "output_dir". You can do it automatically (if you have one example of conf file with all parameters tuned) using notebook [prepare_all_conf](https://github.com/cellgeni/AKOYA_analysis/blob/main/templates/prepare_all_conf.ipynb). And then you can run all of them by submitting number of jobs with [submit_all_AKOYA_jobs.sh](https://github.com/cellgeni/AKOYA_analysis/blob/main/templates/submit_all_AKOYA_jobs.sh)


# Run separate steps of the pipeline

To run separetely steps from the pipeline (such as image preprocessing, segmentation or intensity extraction) please use as an example notebook [AKOYA_analysis_steps](https://github.com/cellgeni/AKOYA_analysis/blob/main/AKOYA_analysis_steps.ipynb). Please note, that there we use only some of all available from [spatialproteomics](https://github.com/sagar87/spatialproteomics), if you find to find out more about other options of image preprocessing, segmentation, plottig and celltyping please visit [spatialproteomics documentation](https://sagar87.github.io/spatialproteomics/)
