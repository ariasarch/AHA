#Core video processing functions

import cv2
import numpy as np
import xarray as xr
import dask.array as da
from typing import Optional
from scipy.stats import kstest
from scipy.spatial.distance import cdist
from dask.array import Array as DaskArray
import networkx as nx
from numpy.fft import fft2, fftshift

def factors(n):
    """Placeholder for a function that returns factors of n."""
    return [factor for factor in range(1, n+1) if n % factor == 0]

def get_chunksize(arr_chk: DaskArray) -> dict:
    """
    Compute chunk sizes from a Dask array.

    Parameters:
    - arr_chk (DaskArray): A Dask array.

    Returns:
    - dict: A dictionary mapping dimension names to chunk sizes.
    """
    # Ensure arr_chk is a Dask array
    if not isinstance(arr_chk, DaskArray):
        raise ValueError("arr_chk must be a Dask array.")
    
    # Mapping dimension names to chunk sizes
    return {dim: size for dim, size in zip(arr_chk.dims, arr_chk.chunksize)}

def get_optimal_chk(
    arr: xr.DataArray,
    dim_grp=[("frame",), ("height", "width")],
    csize=256,
    dtype: Optional[type] = None,
) -> dict:
    """
    Compute the optimal chunk size across all dimensions of the input array.
    
    This function uses `dask` autochunking mechanism to determine the optimal
    chunk size of an array. The difference between this and directly using
    "auto" as chunksize is that it understands which dimensions are usually
    chunked together with the help of `dim_grp`. It also support computing
    chunks for custom `dtype` and explicit requirement of chunk size.
    """
    if dtype is not None:
        arr = arr.astype(dtype)
    dims = arr.dims
    if not dim_grp:
        dim_grp = [(d,) for d in dims]
    chk_compute = dict()
    for dg in dim_grp:
        d_rest = set(dims) - set(dg)
        dg_dict = {d: "auto" for d in dg}
        dr_dict = {d: -1 for d in d_rest}
        dg_dict.update(dr_dict)
        with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
            arr_chk = arr.chunk(dg_dict)
        chk = get_chunksize(arr_chk)
        chk_compute.update({d: chk[d] for d in dg})
    with da.config.set({"array.chunk-size": "{}MiB".format(csize)}):
        arr_chk = arr.chunk({d: "auto" for d in dims})
    chk_store_da = get_chunksize(arr_chk)
    chk_store = dict()
    for d in dims:
        ncomp = int(arr.sizes[d] / chk_compute[d])
        sz = np.array(factors(ncomp)) * chk_compute[d]
        chk_store[d] = sz[np.argmin(np.abs(sz - chk_store_da[d]))]
    return chk_compute, chk_store_da

def apply_denoise(self):
    if self.video_editor_ui.data_array is not None:
        denoise_method = self.video_editor_ui.denoise_method_combo.currentText()
        denoise_param = self.video_editor_ui.denoise_param_slider.value()
        
        # Assume data_array contains grayscale images for simplicity
        # Apply denoising to each frame
        denoised_data = []
        for frame in self.video_editor_ui.data_array:
            denoised_frame = vp.denoise(frame, method=denoise_method, kernel_size=denoise_param)
            denoised_data.append(denoised_frame)

        # Convert the list of frames back to DataArray if needed
        # Update the video display or data storage accordingly
        # This is a placeholder to indicate where you might convert and display the denoised video
        print("Denoising completed.")
    else:
        print("Data array is not loaded.")


def denoise(frame, method='gaussian', kernel_size=5):
    """
    Apply denoising to a single frame.

    Args:
        frame (np.array): The input frame.
        method (str): Denoising method ('gaussian', 'median', or 'bilateral').
        kernel_size (int): Size of the kernel used for denoising.

    Returns:
        np.array: The denoised frame.
    """
    kernel_size = int(kernel_size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    if method == 'gaussian':
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    elif method == 'median':
        return cv2.medianBlur(frame, kernel_size)
    elif method == 'bilateral':
        return cv2.bilateralFilter(frame, kernel_size, 75, 75)
    else:
        raise ValueError(f"Denoise method {method} not understood")
    


def remove_background(frame, method="uniform", kernel_size=5):
    """
    Remove background from a single frame using specified method.

    Args:
        frame (np.array): The input frame.
        method (str): Background removal method ('uniform' or 'tophat').
        kernel_size (int): Size of the kernel used for background removal.

    Returns:
        np.array: Frame with background removed.
    """
    if method == "uniform":
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        background = cv2.filter2D(frame, -1, kernel)
        return frame - background
    elif method == "tophat":
        selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(frame, cv2.MORPH_TOPHAT, selem)
    else:
        raise ValueError("Unknown background removal method: " + method)


def estimate_motion(current_frame, previous_frame):
    """
    Estimate motion between two consecutive frames.

    Args:
        current_frame (np.array): The current frame.
        previous_frame (np.array): The previous frame.

    Returns:
        np.array: The estimated motion vector field.
    """
    # Parameters for optical flow
    flow_params = {
        'pyr_scale': 0.5, 
        'levels': 3, 
        'winsize': 15, 
        'iterations': 3, 
        'poly_n': 5, 
        'poly_sigma': 1.2, 
        'flags': 0
    }

    # Convert frames to grayscale
    prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, **flow_params)
    
    return flow


def apply_transform(frame, motion_vector, border_mode=cv2.BORDER_REFLECT):
    """
    Apply a transformation to a frame based on the motion vector.

    Args:
        frame (np.array): The input frame.
        motion_vector (np.array): The motion vector field.
        border_mode (int): Pixel extrapolation method.

    Returns:
        np.array: The transformed frame.
    """
    h, w = frame.shape[:2]

    # Create a grid of coordinates and add the motion vector
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    coords = np.float32(np.dstack([x, y]) + motion_vector)

    # Apply the warp
    transformed_frame = cv2.remap(frame, coords, None, interpolation=cv2.INTER_LINEAR, borderMode=border_mode)

    return transformed_frame


def seeds_init(frame, threshold=100, min_distance=10):
    """
    Initialize seeds for tracking or segmentation in a frame.

    Args:
        frame (np.array): The input frame.
        threshold (int): The threshold value for seed initialization.
        min_distance (int): The minimum distance between seeds.

    Returns:
        List[Tuple[int, int]]: A list of seed coordinates.
    """
    # Convert frame to grayscale if it's not already
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold
    ret, thresh_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize seeds
    seeds = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            if all(cy > min_distance and cx > min_distance for x, y in seeds):
                seeds.append((cx, cy))

    return seeds


def pnr_refine(frame, seeds, noise_freq=0.25, thres=1.5):
    """
    Refine seeds based on peak-to-noise ratio.

    Args:
        frame (np.array): The input frame.
        seeds (List[Tuple[int, int]]): Initial seeds to refine.
        noise_freq (float): Frequency threshold for noise.
        thres (float): Threshold for PNR to keep a seed.

    Returns:
        List[Tuple[int, int]]: Refined list of seed coordinates.
    """
    refined_seeds = []

    for seed in seeds:
        x, y = seed

        # Extract a small region around the seed
        region = frame[max(y - 10, 0):min(y + 10, frame.shape[0]), max(x - 10, 0):min(x + 10, frame.shape[1])]
        
        # Calculate peak-to-noise ratio in the region
        peak = np.max(region)
        noise = np.std(region)
        pnr = peak / noise if noise != 0 else 0

        # Add seed to refined list if it meets the threshold
        if pnr >= thres:
            refined_seeds.append(seed)

    return refined_seeds



def ks_refine(frame, seeds, significance_level=0.05):
    """
    Refine seeds based on the Kolmogorov-Smirnov test.

    Args:
        frame (np.array): The input frame.
        seeds (List[Tuple[int, int]]): Initial seeds to refine.
        significance_level (float): Significance level for the KS test.

    Returns:
        List[Tuple[int, int]]: Refined list of seed coordinates.
    """
    refined_seeds = []

    for seed in seeds:
        x, y = seed

        # Extract a small region around the seed
        region = frame[max(y - 10, 0):min(y + 10, frame.shape[0]), max(x - 10, 0):min(x + 10, frame.shape[1])].flatten()

        # Perform the Kolmogorov-Smirnov test
        ks_statistic, p_value = kstest(region, 'norm')

        # Add seed to refined list if the test passes the significance level
        if p_value > significance_level:
            refined_seeds.append(seed)

    return refined_seeds


def seeds_merge(seeds, distance_threshold=5):
    """
    Merge seeds that are close to each other.

    Args:
        seeds (List[Tuple[int, int]]): List of seed coordinates.
        distance_threshold (int): Maximum distance to consider seeds for merging.

    Returns:
        List[Tuple[int, int]]: Merged list of seed coordinates.
    """
    if not seeds:
        return []

    # Convert list of tuples to numpy array for easier manipulation
    seeds_array = np.array(seeds)

    # Compute the distance matrix between seeds
    dist_matrix = np.sqrt(np.sum((seeds_array[:, np.newaxis, :] - seeds_array[np.newaxis, :, :]) ** 2, axis=2))

    # Merge seeds
    merged_seeds = []
    while seeds_array.size > 0:
        # Start with the first seed
        seed = seeds_array[0]
        merged_seeds.append(tuple(seed))

        # Find seeds within the distance threshold
        close_seeds = np.where(dist_matrix[0] <= distance_threshold)[0]

        # Remove the seeds that are close from the list
        seeds_array = np.delete(seeds_array, close_seeds, axis=0)
        dist_matrix = np.delete(dist_matrix, close_seeds, axis=0)
        dist_matrix = np.delete(dist_matrix, close_seeds, axis=1)

    return merged_seeds


def initA(frame, seeds, spatial_radius=5):
    """
    Initialize spatial footprints based on seeds.

    Args:
        frame (np.array): The input frame.
        seeds (List[Tuple[int, int]]): List of seed coordinates.
        spatial_radius (int): Radius around each seed to define the spatial footprint.

    Returns:
        List[np.array]: List of spatial footprints.
    """
    footprints = []

    for seed in seeds:
        x, y = seed

        # Define the bounding box for the spatial footprint
        x_min, x_max = max(x - spatial_radius, 0), min(x + spatial_radius, frame.shape[1])
        y_min, y_max = max(y - spatial_radius, 0), min(y + spatial_radius, frame.shape[0])

        # Extract the spatial footprint
        footprint = frame[y_min:y_max, x_min:x_max]

        # Add the extracted footprint to the list
        footprints.append(footprint)

    return footprints


def initC(frames, footprints):
    """
    Initialize temporal components for each spatial footprint.

    Args:
        frames (List[np.array]): List of video frames.
        footprints (List[np.array]): List of spatial footprints.

    Returns:
        List[np.array]: List of temporal components for each footprint.
    """
    temporal_components = []

    for footprint in footprints:
        # Initialize a temporal component for the footprint
        temporal_component = np.zeros(len(frames))

        for i, frame in enumerate(frames):
            # Extract the same region as the footprint from the current frame
            region = frame[:footprint.shape[0], :footprint.shape[1]]

            # Calculate the temporal component as the mean intensity
            temporal_component[i] = np.mean(region * footprint)

        temporal_components.append(temporal_component)

    return temporal_components


def unit_merge(units, similarity_threshold=0.8):
    """
    Merge units that are similar to each other.

    Args:
        units (List[np.array]): List of units (e.g., spatial footprints or temporal components).
        similarity_threshold (float): Threshold for merging based on similarity.

    Returns:
        List[np.array]: Merged list of units.
    """
    if not units:
        return []

    # Placeholder for similarity calculation
    def calculate_similarity(unit1, unit2):
        # Implement a method to calculate similarity (e.g., correlation) between two units
        return np.corrcoef(unit1.flatten(), unit2.flatten())[0, 1]

    merged_units = []
    merged = [False] * len(units)

    for i, unit in enumerate(units):
        if merged[i]:
            continue

        merge_candidate = unit
        for j in range(i + 1, len(units)):
            if merged[j]:
                continue

            similarity = calculate_similarity(unit, units[j])
            if similarity >= similarity_threshold:
                merge_candidate += units[j]  # Merge the units
                merged[j] = True

        merged_units.append(merge_candidate)
    
    return merged_units


def get_noise_fft(frame):
    """
    Estimate noise frequency using FFT.

    Args:
        frame (np.array): The input frame.

    Returns:
        np.array: FFT of the frame.
    """
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Fast Fourier Transform
    f_transform = fft2(gray_frame)
    f_shift = fftshift(f_transform)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shift))

    return magnitude_spectrum


def update_spatial(frames, current_spatial, update_factor=0.1):
    """
    Update spatial components based on current frames.

    Args:
        frames (List[np.array]): List of video frames.
        current_spatial (List[np.array]): Current spatial components.
        update_factor (float): Factor for updating the spatial components.

    Returns:
        List[np.array]: Updated spatial components.
    """
    updated_spatial = []

    for spatial in current_spatial:
        updated_component = spatial.copy()

        for frame in frames:
            # Update the spatial component based on the frame
            updated_component += update_factor * (frame - spatial)

        updated_spatial.append(updated_component)

    return updated_spatial


def update_temporal(frames, temporal_components, update_factor=0.1):
    """
    Update temporal components based on video frames.

    Args:
        frames (List[np.array]): List of video frames.
        temporal_components (List[np.array]): Current temporal components.
        update_factor (float): Factor controlling the update magnitude.

    Returns:
        List[np.array]: Updated list of temporal components.
    """
    updated_components = []

    for component in temporal_components:
        updated_component = np.zeros_like(component)

        for i, frame in enumerate(frames):
            # Placeholder for a specific update rule
            # Here we update the temporal component based on the frame's intensity
            updated_component[i] = (1 - update_factor) * component[i] + update_factor * np.mean(frame)

        updated_components.append(updated_component)

    return updated_components


def generate_videos(frames, transformations):
    """
    Generate new videos after applying transformations to frames.

    Args:
        frames (List[np.array]): List of original video frames.
        transformations (List[callable]): List of functions to apply to each frame.

    Returns:
        List[np.array]: List of transformed video frames.
    """
    transformed_videos = []

    for transform in transformations:
        transformed_video = [transform(frame) for frame in frames]
        transformed_videos.append(transformed_video)

    return transformed_videos


def save_minian(data, filename, format='avi'):
    """
    Save processed data into a file.

    Args:
        data (List[np.array] or np.array): Processed data to be saved.
        filename (str): Name of the file to save the data.
        format (str): Format of the file ('avi', 'mp4', etc.).

    Returns:
        bool: True if saving was successful, False otherwise.
    """
    if format == 'avi':
        # Assuming 'data' is a list of frames (video)
        frame_height, frame_width = data[0].shape[:2]
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 30, (frame_width, frame_height))

        for frame in data:
            out.write(frame)

        out.release()
        return True
    else:
        # Implement saving for other formats or data types
        raise NotImplementedError(f"Saving format '{format}' is not implemented.")

    return False

