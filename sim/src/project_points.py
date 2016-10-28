import numpy as np
from logutils import setup_logging
import h5py

# camera constants for the iphone
f  = 4.11/1000.
W = 4.8/1000.
H = 3.6 / 1000.   # http://photoseek.com/2013/compare-digital-camera-sensor-sizes-full-frame-35mm-aps-c-micro-four-thirds-1-inch-type/
c_dim = np.array([W, H])
resolution = np.array([1280, 720],dtype='uint16')


LOG = setup_logging()
def xy_to_xyz_homogenous(arr):
    """input an array of x, y coordinates (last dimension must be of length two), and output x, y, z, 1 coordinates"""
    to_concat = np.zeros(arr.shape[0:-1] + (2,), dtype=arr.dtype)
    to_concat[:,:,1] = 1.
    return np.concatenate([arr, to_concat], axis=arr.ndim-1)


if __name__ == "__main__":

    # load up matricies
    with open("../src/extrinsic_matrix.npy", "r") as fil:
        openfil = np.load(fil)
        #camera_points = openfil['camera_points']
        #lookat_points = openfil['lookat_points']
        extrinsic_matrix = openfil['extrinsic_matrix']
        
    with open("../src/results.npz", "r") as fil2:
        openfil = np.load(fil2)
        results = openfil['results']
        #info = openfil['info']

    LOG.info("Imported extrinsic_matrix of shape %s" % str(extrinsic_matrix.shape))
    LOG.info("Imported results matrix of shape %s" % str(results.shape))

    pitch_points = xy_to_xyz_homogenous(results[:,:,1:3])
    LOG.info("Pitch points shape is %s" % str(pitch_points.shape))

    n_pitches, n_frames = pitch_points.shape[0:2]
    n_lookat_targets, n_cam_positions = extrinsic_matrix.shape[0:2]

    n_results = n_pitches*n_frames*n_cam_positions*n_lookat_targets

    LOG.info("There are %s different pitches" % "{:,}".format(n_pitches))
    LOG.info("There are %s different frames per pitch" % "{:,}".format(n_frames))
    LOG.info("There are %s different cam positions" % "{:,}".format(n_cam_positions))
    LOG.info("There are %s different lookat targets" % "{:,}".format(n_lookat_targets))
    LOG.info("We have %s results to be output." % "{:,}".format(n_results))

    output_size_gb = n_results * 2 * 2 * 1e-9
    LOG.info("The output data (typed uint16) will be of size %s GB uncompressed." % output_size_gb)

    h5_file = 'projected_points.hdf5'
    fh5 = h5py.File(h5_file)

    ds_name = "projected_points_results"
    try:
        del fh5[ds_name]
    except KeyError:
        pass
    else:
        LOG.info("Deleted dataset %s from %s because it will be replaced by this run." % (ds_name, h5_file))

    dset = fh5.create_dataset(ds_name, 
        shape = (n_pitches, n_cam_positions, n_lookat_targets, n_frames, 2), 
        dtype=np.uint16)

    LOG.info("Dataset %s created in h5file %s" % (ds_name, h5_file))

    for pitch in range(n_pitches):
        if pitch % 100 == 0:
            LOG.info("Now processing pitch %s" % pitch)
        for cam_pos in range(n_cam_positions):

            camera_coordinates = np.tensordot(pitch_points[pitch,:,:], extrinsic_matrix[:,cam_pos,:,:], axes=([1],[2]))
            
            # actual length in meters on the camera sensor
            xy = camera_coordinates[:,:,0:2] * (f / camera_coordinates[:,:,2, np.newaxis])

            # length in terms of percent of screen
            pct_scr = xy / c_dim

            relative_pixels_from_center = (pct_scr * resolution).astype('uint16')

            final = relative_pixels_from_center + resolution/2


            dset[pitch,cam_pos,:,:] = final.swapaxes(0,1)
    
    LOG.info("Script Finished")
    fh5.close()