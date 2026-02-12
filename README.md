# Prepare environment
 Create the conda environment from yaml file and activate:
 
` conda env create -f environment.yml `

` conda activate sp_env `

 Install spatialproteomics package from cellgeni fork:
 
` python -m pip install "git+https://github.com/cellgeni/spatialproteomics.git" `

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

`min_area (int or falsy)` - If truthy, filters out segmented objects with area <= min_area.

`max_area (int or falsy)` - If truthy, filters out segmented objects with area >= max_area.

### Intermediate plots

`save_intermediate_plots (bool)` - Whether to save intermediate QC plots and ROI snapshots.

`number_intermediate_plots (int)` - Number of random ROIs (subimages) to sample for intermediate plotting.

`size_intermediate_plots (int)` - ROI size in pixels (square). Each ROI is size_intermediate_plots × size_intermediate_plots.

`list_of_genes_intermediate_plots (list[str])` - Channel names to display in intermediate ROI plots (alongside DAPI).

### Binary marker presence / label thresholding

`save_individual_marker_presence_plots (bool)` - Whether to plot marker-vs-celltype (binary label) maps.

`fraction_of_positive_pixels (float)` - Threshold applied to the `_percentage_positive` layer for each marker. Applied the same percentage for all channels


### Output control

`output_dir (str)` - Path to directory where results and plots are written.

`list_output_formats (list[str])` - Which outputs to save. Supported values in this script: ["zarr", "h5ad", "csv"]

`save_intermediate_zarr (bool)` - If True, saves intermediate sp_object snapshots to `output_dir/sp_object.zarr` after key steps.

Optional metadata
`normalise_intensity (bool)` - If True, performs z-score normalization per channel across cells and stores:

`pixelsize (float or null, optional)` - Microns per full-resolution pixel. If provided, it is used only to create anndata h5ad object

## Run pipeline
