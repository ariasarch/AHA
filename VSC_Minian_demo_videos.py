#!/usr/bin/env python
# coding: utf-8

# %%

# Step 1: Import Necessary Modules 

import time
start_time_total = time.time()

import os # Import portable OS-dependent functionality 
import sys # Provides Direct Interaction with Python Variables 
import itertools as itt # For Efficient Looping
import holoviews as hv # Used for Visualization 
from bokeh.plotting import show # For in-line plotting
import numpy as np # For numerical computations
import multiprocessing as mp # Allow for multiproccessing 
from dask.distributed import Client, LocalCluster # Parallel Computing Library - for faster computation
from holoviews.operation.datashader import datashade, regrid # Dynamically Shade Large Datasets
from holoviews.util import Dynamic # Create Dynamic Objects
from IPython.display import display # Display Objects

print("\033[1;32mStep 1 Complete\033[0m")

# %%

# Step 2: Set up Initial Basic Parameters

# Define variables
minian_path = "." # Define the path where the Minan Module is located, a "." means where this notebook is running
dpath = "./demo_movies/" # Define the path where the Demo movies are located
minian_ds_path = os.path.join(dpath, "minian") # Define the path where the Demo movies are stored
intpath = "./minian_intermediate" # Define path for intermediate results 
subset = dict(frame=slice(0, None)) # Define subset of frames to process (here, all frames are included)
subset_mc = None # Motion correction subset (initialized to None)
interactive = True # Flag for enabling interactive mode 
output_size = 100 # Output size, possibly used for visualization
n_workers = int(os.getenv("MINIAN_NWORKERS", 4)) # Number of workers for parallel computation (default set to 4), this could just be replaced by n_workers = 4

# Parameters for saving Minian dataset
param_save_minian = {
    "dpath": minian_ds_path,
    "meta_dict": dict(session=-1, animal=-2), #rat1_session2.avi for example
    "overwrite": True,
}

# Parameters for Preprocessing to load video data
param_load_videos = {
    "pattern": "msCam[0-9]+\.avi$", # Match file names that start with msCam by one or more digits with files that end in .avi
    "dtype": np.uint8, # 8-bit integers (0 - 255)
    "downsample": dict(frame=1, height=1, width=1),
    "downsample_strategy": "subset",
}

# Parameters to denoise the video and for background removal 
param_denoise = {"method": "median", "ksize": 7}
param_background_removal = {"method": "tophat", "wnd": 15}

# Motion Correction Parameters
subset_mc = None # Subset for motion correction
param_estimate_motion = {"dim": "frame"} # Parameters to estimate motion in the video

# Initialization Parameters
param_seeds_init = {
    "wnd_size": 1000,
    "method": "rolling",
    "stp_size": 500,
    "max_wnd": 15,
    "diff_thres": 3,
}
param_pnr_refine = {"noise_freq": 0.06, "thres": 1} # Parameters for refining seeds using Peak-to-Noise ratio
param_ks_refine = {"sig": 0.05} # Parameters for refining seeds using K-S test
param_seeds_merge = {"thres_dist": 10, "thres_corr": 0.8, "noise_freq": 0.06} # Parameters to merge seeds
param_initialize = {"thres_corr": 0.8, "wnd": 10, "noise_freq": 0.06} # Parameters for initializing neuronal components
param_init_merge = {"thres_corr": 0.8} # Parameters for merging initial components

# CNMF (Constrained Non-negative Matrix Factorization) Parameters
param_get_noise = {"noise_range": (0.06, 0.5)} # Parameters to estimate noise level

# Parameters for the first spatial update
param_first_spatial = {
    "dl_wnd": 10,
    "sparse_penal": 0.01,
    "size_thres": (25, None),
}

# Parameters for the first temporal update
param_first_temporal = {
    "noise_freq": 0.06,
    "sparse_penal": 1,
    "p": 1,
    "add_lag": 20,
    "jac_thres": 0.2,
}

param_first_merge = {"thres_corr": 0.8} # Parameters for merging components after the first update

# Parameters for the second spatial update
param_second_spatial = {
    "dl_wnd": 10,
    "sparse_penal": 0.01,
    "size_thres": (25, None),
}

# Parameters for the second temporal update
param_second_temporal = {
    "noise_freq": 0.06,
    "sparse_penal": 1,
    "p": 1,
    "add_lag": 20,
    "jac_thres": 0.4,
}

# Set the number of threads for various libraries to 1
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MINIAN_INTERMEDIATE"] = intpath

print("\033[1;32mStep 2 Complete\033[0m")

# %%

# Step 3: Import related functions from Minian

sys.path.append(minian_path) # Append the Minian path to the system path to ensure Python can locate the Minian module

# Import functions related to the CNMF process from Minian
from minian.cnmf import (
    compute_AtC,             # Function computing product of spatial & temporal matrices
    compute_trace,           # Function to compute a trace (likely a temporal trace of activity)
    get_noise_fft,           # Function to estimate noise using FFT
    smooth_sig,              # Function to smooth signals
    unit_merge,              # Function to merge units (neuronal sources)
    update_spatial,          # Function to update spatial footprints of sources
    update_temporal,         # Function to update temporal activity of sources
    update_background,       # Function to update the estimated background
)

# Import functions related to initialization of sources and seeds in Minian
from minian.initialization import (
    gmm_refine,              # Refine initial estimates using Gaussian Mixture Model
    initA,                   # Initialize spatial footprints
    initC,                   # Initialize temporal activity
    intensity_refine,        # Refine initial estimates based on intensity
    ks_refine,               # Refine initial estimates using K-S test
    pnr_refine,              # Refine initial estimates using Peak-to-Noise Ratio
    seeds_init,              # Initialize seeds or starting points for source detection
    seeds_merge,             # Merge seeds that are close or similar
)

# Import functions related to motion correction in Minian
from minian.motion_correction import (
    apply_transform,         # Apply estimated motion transformations to videos
    estimate_motion,         # Estimate motion in the videos
    xr,                      # 
)

# Import pre-processing functions from Minian
from minian.preprocessing import (
    denoise,                 # Denoise video data
    remove_background,       # Remove background from video data
)

# Import utility functions from Minian
from minian.utilities import (
    TaskAnnotation,          # Likely a tool to annotate tasks or steps
    get_optimal_chk,         # Function to determine optimal chunk sizes for processing
    load_videos,             # Function to load video datasets
    open_minian,             # Open a Minian dataset
    save_minian,             # Save a Minian dataset
)

# Import visualization tools from Minian
from minian.visualization import (
    CNMFViewer,              # Viewer tool to inspect results of CNMF
    VArrayViewer,            # Viewer tool to inspect array-like data
    generate_videos,         # Function to generate videos, possibly of processed data
    visualize_gmm_fit,       # Visualize fit of Gaussian Mixture Model
    visualize_motion,        # Visualize estimated motion
    visualize_preprocess,    # Visualize results of preprocessing
    visualize_seeds,         # Visualize initial seeds or starting points
    visualize_spatial_update,# Visualize updates to spatial footprints
    visualize_temporal_update,# Visualize updates to temporal activity
    write_video,             # Function to write videos, possibly for saving processed data
)

print("\033[1;32mStep 3 Complete\033[0m")

# %%

# Step 4: Module Initialization

# The following cell handles initialization of modules and parameters necessary for minian to be run and usually should not be modified.
dpath = os.path.abspath(dpath)
hv.extension("bokeh", width=100)

print("\033[1;32mStep 4 Complete\033[0m")

# In[5]:

# Step 5: Begin Cluster

# Ensure that all spawned processes are frozen and behaves as expected
mp.freeze_support()

# Allow for cross compatabilty 
if os.name == 'posix': 
    mp.set_start_method('fork', force=True)  # Unix-like operating system
else:
    mp.set_start_method('spawn', force=True)  # default for Windows

# Initialize a local Dask cluster and return a client to interact with it
def initialize_dask():

    from dask.distributed import LocalCluster, Client

    # Set up a local cluster with custom configuration
    cluster = LocalCluster(
        n_workers=n_workers,
        memory_limit="4GB",
        resources={"MEM": 1},
        threads_per_worker=2,
        dashboard_address=":8787",
    )

    # Add custom task annotations to the cluster's scheduler
    annt_plugin = TaskAnnotation()
    cluster.scheduler.add_plugin(annt_plugin)

    # Create a client to interact with the local cluster
    client = Client(cluster)

    return client, cluster

# Ensure this block of code is only executed when this script is run directly (not imported)
if __name__ == '__main__': 

    client, cluster = initialize_dask() 

    print("Dashboard is available at:", client.dashboard_link)
    print("\033[1;32mStep 5 Complete\033[0m")

    # %%

    # Step 6: Preprocessing 

    start_time = time.time() 

    varr = load_videos(dpath, **param_load_videos)
    chk, _ = get_optimal_chk(varr, dtype=float) # Estimate optimal chunk size for computations

    # Re-chunk the video array (change its internal block division for efficient computation)
    varr = save_minian(
        varr.chunk({"frame": chk["frame"], "height": -1, "width": -1}).rename("varr"),
        intpath,
        overwrite=True,
    )

    # Visualization of Raw Data and optionally set roi for motion correction
    hv.output(size=output_size)
    if interactive:
        vaviewer = VArrayViewer(varr, framerate=5, summary=["mean", "max"])
        display(vaviewer.show())

    if interactive:
        try:
            subset_mc = list(vaviewer.mask.values())[0]
        except IndexError:
            pass

    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 6 Complete\033[0m")

    # %%

    # Step 7: Perform Glow Removal 

    start_time = time.time() 

    varr_ref = varr.sel(subset) 
    varr_min = varr_ref.min("frame").compute() 
    varr_ref = varr_ref - varr_min 

    if interactive: 
        vaviewer = VArrayViewer( 
            [varr.rename("original"), varr_ref.rename("glow_removed")], 
            framerate=5, 
            summary=None, 
            layout=True, 
        )

        display(vaviewer.show())
    
    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 7 Complete\033[0m")

    # %%

    # Step 8: Perform Denoising

    start_time = time.time() 

    if interactive:
        display(
            visualize_preprocess(
                varr_ref.isel(frame=0).compute(),
                denoise,
                method=["median"],
                ksize=[5, 7, 9],
            )
        )

    varr_ref = denoise(varr_ref, **param_denoise)

    if interactive:
        display(
            visualize_preprocess(
                varr_ref.isel(frame=0).compute(),
                remove_background,
                method=["tophat"],
                wnd=[10, 15, 20],
            )
        )
    
    varr_ref = remove_background(varr_ref, **param_background_removal)
    varr_ref = save_minian(varr_ref.rename("varr_ref"), dpath=intpath, overwrite=True)
   
    elapsed_time = time.time() - start_time     
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 8 Complete\033[0m")

    # %%

    # Step 9: Perform Motion Correction   
    
    start_time = time.time() 
    
    # Estimate Motion
    motion = estimate_motion(varr_ref.sel(subset_mc), **param_estimate_motion)
    motion = save_minian(motion.rename("motion").chunk({"frame": chk["frame"]}), **param_save_minian)
    general_motion_plot = visualize_motion(motion)
    show(hv.render(general_motion_plot))
   
    # Update our transformation as Y_hw_chk
    Y = apply_transform(varr_ref, motion, fill=0) 
    Y_fm_chk = save_minian(Y.astype(float).rename("Y_fm_chk"), intpath, overwrite=True)
    Y_hw_chk = save_minian(Y_fm_chk.rename("Y_hw_chk"), intpath, overwrite=True, chunks={"frame": -1, "height": chk["height"], "width": chk["width"]})

    if interactive:
        vaviewer = VArrayViewer(
            [varr_ref.rename("before_mc"), Y_fm_chk.rename("after_mc")],
            framerate=5,
            summary=None,
            layout=True,
        )
        display(vaviewer.show())


    im_opts = dict(
        frame_width=500,
        aspect=varr_ref.sizes["width"] / varr_ref.sizes["height"],
        cmap="Viridis",
        colorbar=True,
    )
    
    max_projection_motion_correction = (
        regrid(
            hv.Image(
                varr_ref.max("frame").compute().astype(np.float32),
                ["width", "height"],
                label="before_mc",
            ).opts(**im_opts)
        )
        + regrid(
            hv.Image(
                Y_hw_chk.max("frame").compute().astype(np.float32),
                ["width", "height"],
                label="after_mc",
            ).opts(**im_opts)
        )
    )
    
    show(hv.render(max_projection_motion_correction))
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 9 Complete\033[0m")

    # %%

    # Step 10: Generating Single ROIs 

    start_time = time.time() 

    # Generate side-by-side Comparison 
    vid_arr = xr.concat([varr_ref, Y_fm_chk], "width").chunk({"width": -1})
    write_video(vid_arr, "minian_mc.mp4", dpath)

    # Save Max Projection 
    max_proj = save_minian(Y_fm_chk.max("frame").rename("max_proj"), **param_save_minian).compute()
    
    # Set Seed Initializtion
    seeds = seeds_init(Y_fm_chk, **param_seeds_init)
    # hv.output(size=output_size)
    seed_intialization_plot = visualize_seeds(max_proj, seeds)
    show(hv.render(seed_intialization_plot))
    
    # Peak to Noise Refining
    if interactive:
        noise_freq_list = [0.005, 0.01, 0.02, 0.06, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
        example_seeds = seeds.sample(6, axis="rows")
        example_trace = Y_hw_chk.sel(height=example_seeds["height"].to_xarray(), width=example_seeds["width"].to_xarray()).rename(**{"index": "seed"})
        smooth_dict = dict()
        for freq in noise_freq_list:
            trace_smth_low = smooth_sig(example_trace, freq)
            trace_smth_high = smooth_sig(example_trace, freq, btype="high")
            trace_smth_low = trace_smth_low.compute()
            trace_smth_high = trace_smth_high.compute()
            hv_trace = hv.HoloMap({"signal": (hv.Dataset(trace_smth_low).to(hv.Curve, kdims=["frame"]).opts(frame_width=300, aspect=2, ylabel="Signal (A.U.)")),
                                   "noise": (hv.Dataset(trace_smth_high).to(hv.Curve, kdims=["frame"]).opts(frame_width=300, aspect=2, ylabel="Signal (A.U.)")),
                                   }, kdims="trace",).collate()
            smooth_dict[freq] = hv_trace

    # hv.output(size=int(output_size * 0.7))
    if interactive:
        hv_res = (
            hv.HoloMap(smooth_dict, kdims=["noise_freq"])
            .collate()
            .opts(aspect=2)
            .overlay("trace")
            .layout("seed")
            .cols(3)
        )
        display(hv_res)

    # Update via pnr_refine
    seeds, pnr, gmm = pnr_refine(Y_hw_chk, seeds, **param_pnr_refine) # thres = auto assumes a Guassian Mixture Model

    if gmm:
        display(visualize_gmm_fit(pnr, gmm, 100))
    else:
        print("nothing to show")

    # hv.output(size=output_size)
    max_projection_seed_plot = visualize_seeds(max_proj, seeds, "mask_pnr")
    show(hv.render(max_projection_seed_plot))

    # Refine via a Kolmogorov-Smirnov test
    seeds = ks_refine(Y_hw_chk, seeds, **param_ks_refine)
    # hv.output(size=output_size)
    ks_mask_update = visualize_seeds(max_proj, seeds, "mask_ks")
    show(hv.render(ks_mask_update))

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 10 Complete\033[0m")

    # %%

    # Step 11: Initializing CNMF

    start_time = time.time() 

    seeds_final = seeds[seeds["mask_ks"] & seeds["mask_pnr"]].reset_index(drop=True)
    seeds_final = seeds_merge(Y_hw_chk, max_proj, seeds_final, **param_seeds_merge)

    # hv.output(size=output_size)
    single_roi_plot = visualize_seeds(max_proj, seeds_final, "mask_mrg") 
    show(hv.render(single_roi_plot))

    A_init = initA(Y_hw_chk, seeds_final[seeds_final["mask_mrg"]], **param_initialize)
    A_init = save_minian(A_init.rename("A_init"), intpath, overwrite=True)

    C_init = initC(Y_fm_chk, A_init)
    C_init = save_minian(C_init.rename("C_init"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})

    A, C = unit_merge(A_init, C_init, **param_init_merge)
    A = save_minian(A.rename("A"), intpath, overwrite=True)
    C = save_minian(C.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"),intpath,overwrite=True,chunks={"unit_id": -1, "frame": chk["frame"]})

    b, f = update_background(Y_fm_chk, A, C_chk)
    f = save_minian(f.rename("f"), intpath, overwrite=True)
    b = save_minian(b.rename("b"), intpath, overwrite=True)

    # hv.output(size=int(output_size * 0.55))
    im_opts = dict(
        frame_width=500,
        aspect=A.sizes["width"] / A.sizes["height"],
        cmap="Viridis",
        colorbar=True,
    )
    cr_opts = dict(frame_width=750, aspect=1.5 * A.sizes["width"] / A.sizes["height"])
    merged_rois_plot = (
        regrid(
            hv.Image(
                A.max("unit_id").rename("A").compute().astype(np.float32),
                kdims=["width", "height"],
            ).opts(**im_opts)
        ).relabel("Initial Spatial Footprints")
        + regrid(
            hv.Image(
                C.rename("C").compute().astype(np.float32), kdims=["frame", "unit_id"]
            ).opts(cmap="viridis", colorbar=True, **cr_opts)
        ).relabel("Initial Temporal Components")
        + regrid(
            hv.Image(
                b.rename("b").compute().astype(np.float32), kdims=["width", "height"]
            ).opts(**im_opts)
        ).relabel("Initial Background Sptial")
        + datashade(hv.Curve(f.rename("f").compute(), kdims=["frame"]), min_alpha=200)
        .opts(**cr_opts)
        .relabel("Initial Background Temporal")
    ).cols(2)

    show(hv.render(merged_rois_plot))
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 11 Complete\033[0m")

    # %%

    # Step 12: Estimating Spatial Noise for CNMF

    start_time = time.time() 

    sn_spatial = get_noise_fft(Y_hw_chk, **param_get_noise)
    sn_spatial = save_minian(sn_spatial.rename("sn_spatial"), intpath, overwrite=True)

    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C.sel(unit_id=units).persist()

    if interactive:
        sprs_ls = [0.005, 0.01, 0.05]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_mask, cur_norm = update_spatial(
                Y_hw_chk,
                A_sub,
                C_sub,
                sn_spatial,
                in_memory=True,
                dl_wnd=param_first_spatial["dl_wnd"],
                sparse_penal=cur_sprs,
            )
            if cur_A.sizes["unit_id"]:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])

    if interactive:
        display(hv_res)
    
    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_first_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)
    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    opts = dict(
        plot=dict(height=A.sizes["height"], width=A.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    
    spatial_footprint_plot = (
        regrid(
            hv.Image(
                A.max("unit_id").compute().astype(np.float32).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Spatial Footprints Initial")
        + regrid(
            hv.Image(
                (A.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Binary Spatial Footprints Initial")
        + regrid(
            hv.Image(
                A_new.max("unit_id").compute().astype(np.float32).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Spatial Footprints First Update")
        + regrid(
            hv.Image(
                (A_new > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Binary Spatial Footprints First Update")
    ).cols(2)

    show(hv.render(spatial_footprint_plot))

    opts_im = dict(
        plot=dict(height=b.sizes["height"], width=b.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    opts_cr = dict(plot=dict(height=b.sizes["height"], width=b.sizes["height"] * 2))
    
    background_spatial_plot = (
        regrid(
            hv.Image(b.compute().astype(np.float32), kdims=["width", "height"]).opts(
                **opts_im
            )
        ).relabel("Background Spatial Initial")
        + hv.Curve(f.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal Initial")
        + regrid(
            hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"]).opts(
                **opts_im
            )
        ).relabel("Background Spatial First Update")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal First Update")
    ).cols(2)

    show(hv.render(background_spatial_plot))

    A = save_minian(
        A_new.rename("A"),
        intpath,
        overwrite=True,
        chunks={"unit_id": 1, "height": -1, "width": -1},
    )
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(
        f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True
    )
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)

    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 12 Complete\033[0m")
    
    # %%

    # Step 13: Estimating Temporal Activity for CNMF
    
    start_time = time.time() 

    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C_chk.sel(unit_id=units).persist()

    if interactive:
        p_ls = [1]
        sprs_ls = [0.1, 0.5, 1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = (
            compute_trace(Y_fm_chk, A_sub, b, C_sub, f)
            .persist()
            .chunk({"unit_id": 1, "frame": -1})
        )
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(
            p_ls, sprs_ls, add_ls, noise_ls
        ):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print(
                "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(
                    cur_p, cur_sprs, cur_add, cur_noise
                )
            )
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=cur_sprs,
                p=cur_p,
                use_smooth=True,
                add_lag=cur_add,
                noise_freq=cur_noise,
            )
            YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (
                YrA.compute(),
                cur_C.compute(),
                cur_S.compute(),
                cur_g.compute(),
                (cur_C + cur_b0 + cur_c0).compute(),
                A_sub.compute(),
            )
        hv_res = visualize_temporal_update(
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
            sig_dict,
            A_dict,
            kdims=["p", "sparse penalty", "additional lag", "noise frequency"],
        )

    if interactive:
        display(hv_res)

    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)
    
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
    
    temproal_components_plot = (
        regrid(
            hv.Image(
                C.compute().astype(np.float32).rename("ci"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Temporal Trace Initial")
        + hv.Div("")
        + regrid(
            hv.Image(
                C_new.compute().astype(np.float32).rename("c1"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Temporal Trace First Update")
        + regrid(
            hv.Image(
                S_new.compute().astype(np.float32).rename("s1"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Spikes First Update")
    ).cols(2)
    
    show(hv.render(temproal_components_plot))

    # Visualization of Dropped Units
    if interactive:
        h, w = A.sizes["height"], A.sizes["width"]
        im_opts = dict(aspect=w / h, frame_width=500, cmap="Viridis")
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = mask.where(mask == False, drop=True).coords["unit_id"].values
        if len(bad_units) > 0:
            hv_res = (
                hv.NdLayout(
                    {
                        "Spatial Footprint": Dynamic(
                            hv.Dataset(A.sel(unit_id=bad_units).compute().rename("A"))
                            .to(hv.Image, kdims=["width", "height"])
                            .opts(**im_opts)
                        ),
                        "Spatial Footprints of Accepted Units": Dynamic(
                            hv.Image(
                                A.sel(unit_id=mask).sum("unit_id").compute().rename("A"),
                                kdims=["width", "height"],
                            ).opts(**im_opts)
                        ),
                    }
                )
                + datashade(
                    hv.Dataset(YrA.sel(unit_id=bad_units).rename("raw")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                .opts(**cr_opts)
                .relabel("Temporal Trace")
            ).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

    # Visualization of Accepted Units
    if interactive:
        sig = C_new + b0_new + c0_new
        display(
            visualize_temporal_update(
                YrA.sel(unit_id=mask),
                C_new,
                S_new,
                g,
                sig,
                A.sel(unit_id=mask),
            )
        )

    # Save Units
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    A_mrg, C_mrg, [sig_mrg] = unit_merge(A, C, [C + b0 + c0], **param_first_merge)

    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
    merged_units_plot = (
        regrid(
            hv.Image(
                C.compute().astype(np.float32).rename("c1"), kdims=["frame", "unit_id"]
            )
            .relabel("Temporal Signals Before Merge")
            .opts(**opts_im)
        )
        + regrid(
            hv.Image(
                C_mrg.compute().astype(np.float32).rename("c2"), kdims=["frame", "unit_id"]
            )
            .relabel("Temporal Signals After Merge")
            .opts(**opts_im)
        )
    )
    
    show(hv.render(merged_units_plot))

    # Save Merged Units
    A = save_minian(A_mrg.rename("A_mrg"), intpath, overwrite=True)
    C = save_minian(C_mrg.rename("C_mrg"), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_mrg_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]},)
    sig = save_minian(sig_mrg.rename("sig_mrg"), intpath, overwrite=True)

    # Parameter Exploration 
    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = sig.sel(unit_id=units).persist()

    if interactive:
        sprs_ls = [5e-3, 1e-2, 5e-2]
        A_dict = dict()
        C_dict = dict()
        for cur_sprs in sprs_ls:
            cur_A, cur_mask, cur_norm = update_spatial(
                Y_hw_chk,
                A_sub,
                C_sub,
                sn_spatial,
                in_memory=True,
                dl_wnd=param_second_spatial["dl_wnd"],
                sparse_penal=cur_sprs,
            )
            if cur_A.sizes["unit_id"]:
                A_dict[cur_sprs] = cur_A.compute()
                C_dict[cur_sprs] = C_sub.sel(unit_id=cur_mask).compute()
        hv_res = visualize_spatial_update(A_dict, C_dict, kdims=["sparse penalty"])

    if interactive:
        display(hv_res)
        
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 13 Complete\033[0m")

    # %%

    # Step 14: Performing Second Spatial Update for CNMF
    
    start_time = time.time() 

    A_new, mask, norm_fac = update_spatial(Y_hw_chk, A, C, sn_spatial, **param_second_spatial)
    C_new = save_minian((C.sel(unit_id=mask) * norm_fac).rename("C_new"), intpath, overwrite=True)
    C_chk_new = save_minian((C_chk.sel(unit_id=mask) * norm_fac).rename("C_chk_new"), intpath, overwrite=True)

    b_new, f_new = update_background(Y_fm_chk, A_new, C_chk_new)

    opts = dict(
        plot=dict(height=A.sizes["height"], width=A.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    spatial_footprint_updated_plot = (
        regrid(
            hv.Image(
                A.max("unit_id").compute().astype(np.float32).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Spatial Footprints Last")
        + regrid(
            hv.Image(
                (A.fillna(0) > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Binary Spatial Footprints Last")
        + regrid(
            hv.Image(
                A_new.max("unit_id").compute().astype(np.float32).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Spatial Footprints New")
        + regrid(
            hv.Image(
                (A_new > 0).sum("unit_id").compute().astype(np.uint8).rename("A"),
                kdims=["width", "height"],
            ).opts(**opts)
        ).relabel("Binary Spatial Footprints New")
    ).cols(2)

    show(hv.render(spatial_footprint_updated_plot))
    
    opts_im = dict(
        plot=dict(height=b.sizes["height"], width=b.sizes["width"], colorbar=True),
        style=dict(cmap="Viridis"),
    )
    opts_cr = dict(plot=dict(height=b.sizes["height"], width=b.sizes["height"] * 2))
    background_visualization = (
        regrid(
            hv.Image(b.compute().astype(np.float32), kdims=["width", "height"]).opts(
                **opts_im
            )
        ).relabel("Background Spatial Last")
        + hv.Curve(f.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal Last")
        + regrid(
            hv.Image(b_new.compute().astype(np.float32), kdims=["width", "height"]).opts(
                **opts_im
            )
        ).relabel("Background Spatial New")
        + hv.Curve(f_new.compute().rename("f").astype(np.float16), kdims=["frame"])
        .opts(**opts_cr)
        .relabel("Background Temporal New")
    ).cols(2)
    
    show(hv.render(background_visualization))

    # Save Results
    A = save_minian(A_new.rename("A"), intpath, overwrite=True, chunks={"unit_id": 1, "height": -1, "width": -1},)
    b = save_minian(b_new.rename("b"), intpath, overwrite=True)
    f = save_minian(f_new.chunk({"frame": chk["frame"]}).rename("f"), intpath, overwrite=True)
    C = save_minian(C_new.rename("C"), intpath, overwrite=True)
    C_chk = save_minian(C_chk_new.rename("C_chk"), intpath, overwrite=True)
    
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 14 Complete\033[0m")

    # %%

    # Step 15: Performing Second Temporal Update for CNMF
    
    start_time = time.time() 

    # Paramter Exploration
    if interactive:
        units = np.random.choice(A.coords["unit_id"], 10, replace=False)
        units.sort()
        A_sub = A.sel(unit_id=units).persist()
        C_sub = C_chk.sel(unit_id=units).persist()

    if interactive:
        p_ls = [1]
        sprs_ls = [0.1, 0.5, 1, 2]
        add_ls = [20]
        noise_ls = [0.06]
        YA_dict, C_dict, S_dict, g_dict, sig_dict, A_dict = [dict() for _ in range(6)]
        YrA = (
            compute_trace(Y_fm_chk, A_sub, b, C_sub, f)
            .persist()
            .chunk({"unit_id": 1, "frame": -1})
        )
        for cur_p, cur_sprs, cur_add, cur_noise in itt.product(
            p_ls, sprs_ls, add_ls, noise_ls
        ):
            ks = (cur_p, cur_sprs, cur_add, cur_noise)
            print(
                "p:{}, sparse penalty:{}, additional lag:{}, noise frequency:{}".format(
                    cur_p, cur_sprs, cur_add, cur_noise
                )
            )
            cur_C, cur_S, cur_b0, cur_c0, cur_g, cur_mask = update_temporal(
                A_sub,
                C_sub,
                YrA=YrA,
                sparse_penal=cur_sprs,
                p=cur_p,
                use_smooth=True,
                add_lag=cur_add,
                noise_freq=cur_noise,
            )
            YA_dict[ks], C_dict[ks], S_dict[ks], g_dict[ks], sig_dict[ks], A_dict[ks] = (
                YrA.compute(),
                cur_C.compute(),
                cur_S.compute(),
                cur_g.compute(),
                (cur_C + cur_b0 + cur_c0).compute(),
                A_sub.compute(),
            )
        hv_res = visualize_temporal_update(
            YA_dict,
            C_dict,
            S_dict,
            g_dict,
            sig_dict,
            A_dict,
            kdims=["p", "sparse penalty", "additional lag", "noise frequency"],
        )

    if interactive:
        display(hv_res)


    YrA = save_minian(compute_trace(Y_fm_chk, A, b, C_chk, f).rename("YrA"), intpath, overwrite=True, chunks={"unit_id": 1, "frame": -1})
    C_new, S_new, b0_new, c0_new, g, mask = update_temporal(A, C, YrA=YrA, **param_second_temporal)
    
    opts_im = dict(frame_width=500, aspect=2, colorbar=True, cmap="Viridis")
    second_temporal_update = (
        regrid(
            hv.Image(
                C.compute().astype(np.float32).rename("c1"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Temporal Trace Last")
        + regrid(
            hv.Image(
                S.compute().astype(np.float32).rename("s1"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Spikes Last")
        + regrid(
            hv.Image(
                C_new.compute().astype(np.float32).rename("c2"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Temporal Trace New")
        + regrid(
            hv.Image(
                S_new.compute().astype(np.float32).rename("s2"), kdims=["frame", "unit_id"]
            ).opts(**opts_im)
        ).relabel("Spikes New")
    ).cols(2)

    show(hv.render(second_temporal_update))

    # Visualization of Dropped Units
    if interactive:
        h, w = A.sizes["height"], A.sizes["width"]
        im_opts = dict(aspect=w / h, frame_width=500, cmap="Viridis")
        cr_opts = dict(aspect=3, frame_width=1000)
        bad_units = mask.where(mask == False, drop=True).coords["unit_id"].values
        if len(bad_units) > 0:
            hv_res = (
                hv.NdLayout(
                    {
                        "Spatial Footprint": Dynamic(
                            hv.Dataset(A.sel(unit_id=bad_units).compute().rename("A"))
                            .to(hv.Image, kdims=["width", "height"])
                            .opts(**im_opts)
                        ),
                        "Spatial Footprints of Accepted Units": Dynamic(
                            hv.Image(
                                A.sel(unit_id=mask).sum("unit_id").compute().rename("A"),
                                kdims=["width", "height"],
                            ).opts(**im_opts)
                        ),
                    }
                )
                + datashade(
                    hv.Dataset(YrA.sel(unit_id=bad_units).rename("raw")).to(
                        hv.Curve, kdims=["frame"]
                    )
                )
                .opts(**cr_opts)
                .relabel("Temporal Trace")
            ).cols(1)
            display(hv_res)
        else:
            print("No rejected units to display")

    # Visualization of Accepted Units
    if interactive:
        sig = C_new + b0_new + c0_new
        display(
            visualize_temporal_update(
                YrA.sel(unit_id=mask),
                C_new,
                S_new,
                g,
                sig,
                A.sel(unit_id=mask),
            )
        )

    # Save Results 
    C = save_minian(C_new.rename("C").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    C_chk = save_minian(C.rename("C_chk"), intpath, overwrite=True, chunks={"unit_id": -1, "frame": chk["frame"]})
    S = save_minian(S_new.rename("S").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    b0 = save_minian(b0_new.rename("b0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    c0 = save_minian(c0_new.rename("c0").chunk({"unit_id": 1, "frame": -1}), intpath, overwrite=True)
    A = A.sel(unit_id=C.coords["unit_id"].values)

    # Visualization 
    generate_videos(varr.sel(subset), Y_fm_chk, A=A, C=C_chk, vpath=dpath)

    if interactive:
        cnmfviewer = CNMFViewer(A=A, C=C, S=S, org=Y_fm_chk)

    if interactive:
        display(cnmfviewer.show())
        
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")

    print("\033[1;32mStep 15 Complete\033[0m")

    # %%

    # Step 16: Saving Results
    
    start_time = time.time() 
    
    # Save Unit Labels
    if interactive:
        A = A.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        C = C.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        S = S.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        c0 = c0.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))
        b0 = b0.assign_coords(unit_labels=("unit_id", cnmfviewer.unit_labels))


    A = save_minian(A.rename("A"), **param_save_minian)
    C = save_minian(C.rename("C"), **param_save_minian)
    S = save_minian(S.rename("S"), **param_save_minian)
    c0 = save_minian(c0.rename("c0"), **param_save_minian)
    b0 = save_minian(b0.rename("b0"), **param_save_minian)
    b = save_minian(b.rename("b"), **param_save_minian)
    f = save_minian(f.rename("f"), **param_save_minian)
    
    client.close()
    cluster.close()

    elapsed_time = time.time() - start_time_total     # End the timer and print the elapsed time
    print(f"Total Time taken: {elapsed_time:.2f} seconds")
    
    print("\033[1;32mStep 16 Complete\033[0m")

    print('Analysis Complete')