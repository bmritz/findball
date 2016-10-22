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
from cartesian import cartesian

FOV = 63.54  #degrees

# rules get all camera positions in a 3-d box on one side of pitcher
# Box (mesh grid) that the camera will be in
# distances in ft (will be converted)

Xc_min = -20  # -20
Xc_max = 80 # 80
Xc_n = (Xc_max - Xc_min) + 1  # get whole feet
Xc = np.linspace(Xc_min, Xc_max, Xc_n, dtype='float32')

Yc_min = 3
Yc_max = 5
Yc_n = (Yc_max - Yc_min) + 1  # get whole feet
Yc = np.linspace(Yc_min, Yc_max, Yc_n, dtype='float32')

Zc_min = 50
Zc_max = 150 # 200
Zc_n = (Zc_max - Zc_min) + 1  # get whole feet
Zc = np.linspace(Zc_min, Zc_max, Zc_n, dtype='float32')

points_c = cartesian([Xc, Yc, Zc])

# bounded plane for lookat target point
Xp_min = -5
Xp_max = 40 #40
Xp_n = (Xp_max - Xp_min) + 1   # get whole feet
Xp = np.linspace(Xp_min, Xp_max, Xp_n, dtype='float32')

Yp_min = -2
Yp_max = 10  #10
Yp_n = (Yp_max - Yp_min) + 1   # get whole feet
Yp = np.linspace(Yp_min, Yp_max, Yp_n, dtype='float32')

# just looking at x-y axis -- varying z would bring duplication

points_p = cartesian([Xp, Yp, np.array([0], dtype='float32')])

# metric conversion
m_per_ft = 0.3048
points_c = points_c * m_per_ft
points_p = points_p * m_per_ft


"""
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
L = np.stack([p - points_c for p in points_p])

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
R = np.stack([s, u_, -L], axis=2)

# transformation vector
t = np.zeros(shape=(R.shape[0], ) + points_c.shape)
for num_p in range(t.shape[0]):
    for num_c in range(t.shape[1]):
        t[num_p,num_c] = np.dot(R[num_p,num_c], points_c[num_c])


