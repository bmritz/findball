"""Make camera coordinates

Find all possible vlaues for the projection matrix



plan:

Look at camera model:
1. get all camera positions on one side of the pitch -- 3d box
2. get all points on the x-y axis near the pitcher -- these are the points we will "look at"
3. calculate the rotation matrix for each combination of these points

variable names
XYZ -- coordinate axes -- origin is pitchers mound
suffix c is for camera, p is for lookat point

ex:
Zc -- Z coordinate of camera
Yp -- Y coordinate of the target point for the lookat

distances specified in ft. -- will be conveted

"""

import numpy as np 
import json, argparse, sys, os
from gen_grid_points import gen_grid_points
from logutils import setup_logging

LOG = setup_logging()
# rules get all camera positions in a 3-d box on one side of pitcher
# Box (mesh grid) that the camera will be in
# distances in ft (will be converted)

# convert your specified ranges to another measurement system (ex. from ft. to meters)
# set to 0.3048 to specify in ft, but have matrix in meters
# set to 1 for no conversion

def get_extrinsic_matrix(C, P):
    """get the rotation matricies for all points in C looking at all points in P

    INPUTS
    ------
    C: numpy array of shape (num camera points, 3)
    P: numpy array of shape (num target points, 3)

    OUTPUTS
    -------

    extrinsic_camera_matrix : numpy array of shape (num target points, num camera points, 3, 3)

    algorithm for look-at camera rotation matrix
    1. Compute L = p - C.
    2. Normalize L.
    3. Compute s = L x u. (cross product)
    4. Normalize s.
    5. Compute u_ = s x L.
    6. Then Extrinsic rotation matrix given by:

            s1,  s2,  s3
    R =     u_1, u_2, u_3
            -L1, -L2, -L3

    u is the y-axis -- vary this to get roll

    This takes advantage of broadcasting 
        -- see https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    """
    u = np.array([0,1,0], dtype='float32')

    # 1. Compute L = p - C.
    L = np.stack([p - C for p in P])

    # 2. Normalize L.
    norm_L = np.linalg.norm(L, axis=2)
    L = L / norm_L[:,:, np.newaxis]

    # 3. Compute s = L x u. (cross product)
    s = np.cross(L, u)

    # 4. Normalize s. 
    s = s / np.linalg.norm(s, axis=2)[:,:,np.newaxis]

    # 5. Compute u' = s x L.
    u_ = np.cross(s, L)

    # 6
    # negatives here are correct--but maybe different from what you read because 
    # we are using different handed coordinates -- see understand 3d projection
    R = np.stack([-s, u_, L], axis=2)
    del L
    del s
    del u_

    # transformation vector
    # I CHANGED THIS to conserve memory, concatenate zeros here and set t in immediately,
    # this prevents us from copying R after t is allocated, and prevents us from having to 
    # hold R, T, and EX in memory at the same time

    #t = np.zeros(shape=(R.shape[0], ) + C.shape )
    # EX is the extrinsic matrix
    EX = np.concatenate([R, np.zeros(shape=R.shape[0:3]+(1,), dtype=R.dtype)], axis=3)
    del R # R is held in EX already
    for num_p in range(EX.shape[0]):
        for num_c in range(EX.shape[1]):
            #t[num_p,num_c] = -np.dot(R[num_p,num_c], C[num_c])
            EX[num_p,num_c,:,3] = -np.dot(EX[num_p,num_c,:,0:3], C[num_c])
    # extrinsic matrix
    # changed to create above to conserve memory
    #EX = np.concatenate([R, t[:,:,:,np.newaxis]], axis=3)

    return EX

def gen_C_gen_P(args_c, args_p, dtype="float64", conversion_factor = 1.):
    """ returns C and P matricies """

    C, ranges_C = gen_grid_points(args_c, dtype=dtype) * conversion_factor
    P, ranges_P = gen_grid_points(args_p, dtype=dtype) * conversion_factor
    return C, P, ranges_C, ranges_P

def load_conf(filname):
    LOG.info("Loading configuration file: %s" % filname)
    with open(filname, 'r') as fil:
        conf = json.load(fil)
        args_c = [
            (conf["Xc_min"], conf["Xc_max"], conf["Xc_step"]),
            (conf["Yc_min"], conf["Yc_max"], conf["Yc_step"]),
            (conf["Zc_min"], conf["Zc_max"], conf["Zc_step"]),
        ]

        args_p = [
            (conf["Xp_min"], conf["Xp_max"], conf["Xp_step"]),
            (conf["Yp_min"], conf["Yp_max"], conf["Yp_step"]),
            (conf["Zp_min"], conf["Zp_max"], conf["Zp_step"]),
        ]
        CONVERSION_FACTOR = conf["CONVERSION_FACTOR"]
    return args_c,  args_p, CONVERSION_FACTOR

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="configuration file")
    args = parser.parse_args()

    conf_filname = args.config or "config.json"
    output_filename = "extrinsic_matrix.npy"
    
    args_c, args_p, CONVERSION_FACTOR = load_conf(conf_filname)
    C, P, ranges_C, ranges_P = gen_C_gen_P(args_c, args_p, conversion_factor=CONVERSION_FACTOR, dtype="float32")

    LOG.info("Shape of camera matrix %s" % str(C.shape))
    LOG.info("Shape of target point matrix: %s" % str(P.shape))
    LOG.info("Creating %s extrinsic matricies." % (C.shape[0]*P.shape[0],))
    EX = get_extrinsic_matrix(C,P)
    LOG.info("Shape of extrinsic matrix: %s" % str(EX.shape))
    with open(output_filename, "w") as outfile:
        np.savez(outfile, extrinsic_matrix = EX, camera_points=C, lookat_points=P)

    LOG.info("SCRIPT FINISHED")
