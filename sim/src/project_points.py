import numpy as np
from logutils import setup_logging
import h5py
from output_conf import DATA, delete_if_exists, HDF5_NAME

MAX_GB_DISK = 60.
MAX_GB_RAM = 1.

# camera constants for the iphone
f  = 4.11/1000.
W = 4.8/1000.
H = 3.6 / 1000.   # http://photoseek.com/2013/compare-digital-camera-sensor-sizes-full-frame-35mm-aps-c-micro-four-thirds-1-inch-type/
c_dim = np.array([W, H])
resolution = np.array([1280, 720],dtype='float64')

divider = c_dim * resolution


LOG = setup_logging()
def xy_to_xyz_homogenous(arr):
    """input an array of x, y coordinates (last dimension must be of length two), and output x, y, z, 1 coordinates"""
    to_concat = np.zeros(arr.shape[0:-1] + (2,), dtype=arr.dtype)
    to_concat[:,:,1] = 1.
    return np.concatenate([arr, to_concat], axis=arr.ndim-1)


if __name__ == "__main__":

    extrinsic_matrix = DATA['extrinsic_matricies']
    results = DATA['trajectory_1']['ball_trajectories']

    # find how much data we are dealing with
    n_pitches, n_frames = results.shape[0:2]
    n_lookat_targets, n_cam_positions = extrinsic_matrix.shape[0:2]
    n_extrinsic_matricies = n_cam_positions*n_lookat_targets
    n_results = n_pitches*n_frames*n_cam_positions*n_lookat_targets

    output_size_gb = n_results * 2 * 2 * 1e-9

    LOG.info("There are %s different pitches" % "{:,}".format(n_pitches))
    LOG.info("There are %s different frames per pitch" % "{:,}".format(n_frames))
    LOG.info("There are %s different cam positions and %s lookat targets" %\
     ("{:,}".format(n_cam_positions), "{:,}".format(n_lookat_targets)))
    LOG.info("There are a total %s extrinsic matricies" % "{:,}".format(n_extrinsic_matricies))
    LOG.info("We have %s results to be output." % "{:,}".format(n_results))

    LOG.info("The output data (typed uint16) will be of size %s GB uncompressed." % output_size_gb)

    if output_size_gb > MAX_GB_DISK:
        raise IOError("The size of the output data (%s GB) exceeds the max limit on disk writes (%s GB)." %\
        (output_size_gb, MAX_GB_DISK))
    else:
        ds_name = "final_projections"
        # prepare output dataset
        delete_if_exists(ds_name)
        dset = DATA.create_dataset(ds_name, 
            shape = (n_pitches, n_cam_positions, n_lookat_targets, n_frames, 2), 
            dtype=np.uint16)

        LOG.info("Dataset %s created in h5file %s. Approximately %s GB." % (ds_name, HDF5_NAME, output_size_gb))

    # size of an intermediate matrix for 1 pitch created by tensordot below
    # size for every pitch-cam-chunk -- 8 bytes per entry
    # tensordot copies the data, so we need twice as much memory to be under the cap
    # three entries in camera coordinates
    intermediate_size_per_pitch_cam = 8 * (n_results / (n_pitches*n_cam_positions)) * 2 * 3
    intermediate_size_per_pitch = n_cam_positions*intermediate_size_per_pitch_cam
    
    max_bytes = MAX_GB_RAM*1e+9

    if max_bytes > intermediate_size_per_pitch:
        chunksize_pitch = max_bytes // intermediate_size_per_pitch
        n_chunks_pitch =  (n_pitches // chunksize_pitch) + 1
        n_chunks_cam = 1
    elif max_bytes > intermediate_size_per_pitch_cam:
        n_chunks_pitch = n_pitches
        chunksize_cam = max_bytes // intermediate_size_per_pitch_cam
        n_chunks_cam = (n_cam_positions // chunksize_cam) + 1
    else:
        raise MemoryError("NOT ENOUGH MEMORY TO PROCESS ONE PITCH AND ONE CAMERA POSTIION. MAX RAM CURRENTLY SET AT %s GB" %\
         MAX_GB_RAM)

    chunk_bounds_pitch = np.array([int(x) for x in np.linspace(0, n_pitches, n_chunks_pitch+1)])
    chunk_bounds_cam = np.array([int(x) for x in np.linspace(0, n_cam_positions, n_chunks_cam+1)])

    LOG.info("Breaking the calculation into %s chunk(s)" % int(n_chunks_cam*n_chunks_pitch))
    LOG.info("There are %s pitch chunk(s) and %s cam chunk(s)" % (n_chunks_pitch, n_chunks_cam))

    for lbound_p, ubound_p in zip(chunk_bounds_pitch[:-1], chunk_bounds_pitch[1:]):

        LOG.info("Now processing pitches %s to %s" % (lbound_p, ubound_p))
        pitch_points = xy_to_xyz_homogenous(results[lbound_p:ubound_p,:,1:3])

        for lbound_c, ubound_c in zip(chunk_bounds_cam[:-1], chunk_bounds_cam[1:]):

            # dot product of the pitch point and extrinsic matrix for frame and lookat in the selected cam_pos and pitches
            camera_coordinates = np.tensordot(pitch_points, extrinsic_matrix[:,lbound_c:ubound_c,:,:], axes=([2],[3]))

            # actual length in meters on the camera sensor
            xy = camera_coordinates[:,:,:,:,0:2] * (f / camera_coordinates[:,:,:,:,2, np.newaxis])

            # length in terms of percent of screen
            pct_scr = xy / c_dim

            relative_pixels_from_center = (pct_scr * resolution).astype('uint16')

            final = relative_pixels_from_center + resolution/2

            # combine into one
            # dset[lbound_p:ubound_p,lbound_c:ubound_c,:,:] =\
            #  (((camera_coordinates[:,:,:,:,0:2] * (f / camera_coordinates[:,:,:,:,2, np.newaxis]) )\
            #  / divider).astype('uint16') + (resolution/2)).swapaxes(1,3)


            dset[lbound_p:ubound_p,lbound_c:ubound_c,:,:] = final.swapaxes(1,3)

    LOG.info("Script Finished")
