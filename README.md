# Introduction 

AKOYA pipeline is supposed to automate spatial proteomics low-level data processing. It uses only multichannel image as input file to perform image preprocessing, segmentation and intensity extraction to define presence/absence of marker genes for each segmented cell. It is build moslty using [spatialproteomics_cellgeni](https://github.com/cellgeni/spatialproteomics) package which was edited to work with large images (original package - [spatialproteomics](https://github.com/sagar87/spatialproteomics) As output pipline can produce (i) original spatialproteomics file in zarr xarray format; (ii) set of csv files with analysis tables (like cell position, average intensity per cell and binary marker presence tables); (iii) AnnData file, where X matrix is average intensity of each marker gene per cell. You can find examples of output files and how to open them in notebook **open_output_files.ipynb** 


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


