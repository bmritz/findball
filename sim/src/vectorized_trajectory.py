import numpy as np

ACCELERATION_DUE_TO_GRAVITY = 9.8

def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out

# translate altitude to air density (Taken right from PHet)
def air_density(altitude=0):
    """ takes altitude in meters"""
    if altitude < 11000:
        # troposphere
        temperature = 15.04 - 0.00649 * altitude
        pressure = 101.29 * math.pow( ( temperature + 273.1 ) / 288.08, 5.256 )
    elif altitude < 25000:
        # lower stratosphere
        temperature = -56.46
        pressure = 22.65 * math.exp( 1.73 - 0.000157 * altitude )
    else:
        #upper stratosphere (altitude >= 25000 meters)
        temperature = -131.21 + 0.00299 * altitude
        pressure = 2.488 * math.pow( ( temperature + 273.1 ) / 216.6, -11.388 )
    
    return pressure / ( 0.2869 * ( temperature + 273.1 ) ) 

"""Baseball Coefficients:

diameter: 74.68 mm
area: 4380.2459820395015 mm
mass: between 141.75 to 148.83 g
"""

#initial conditions
class Trajectory(object):
    def __init__(self,x0, y0, velocity_t0, launch_angle_deg_t0,
        air_density, area, drag_coefficient, mass, x_acceleration_t0=None, y_acceleration_t0=None):

        self.x0 = x0
        self.y0 = y0
        self.velocity_t0 = velocity_t0
        self.launch_angle_deg_t0 = launch_angle_deg_t0
        self.launch_angle_t0 = np.deg2rad(self.launch_angle_deg_t0)

        self.x_velocity_t0 = self.velocity_t0 * np.cos(self.launch_angle_t0)
        self.y_velocity_t0 = self.velocity_t0 * np.sin(self.launch_angle_t0)

        self.x_acceleration_t0 = np.zeros_like(self.x0) if x_acceleration_t0 is None else x_acceleration_t0
        self.y_acceleration_t0 = np.full(self.x0.shape, -ACCELERATION_DUE_TO_GRAVITY) if y_acceleration_t0 is None else y_acceleration_t0

        self.air_density = air_density
        self.area = area
        self.drag_coefficient = drag_coefficient
        self.mass = mass

    def solve_n_steps(self, n, dt):
        """
        approx. time:
        tried on set of 10:
        ((245 microseconds) / 10) * 1 000 000 000 =
        6.80555556 hours

        returns an array of pitches

        so arr[:,:,0] is the data for pitch 0
        arr[0,:,0] is the x coordinate across time for pitch 0
        arr[1,:,0] is the y coordinate across time for pitch 0

        """

        # axis 1: time
        # axis 2: which pitch

        # initalize output array
        res_time = np.arange(n)*dt
        x = np.vstack([self.x0, np.zeros((n-1, self.x0.shape[0]))])
        y = np.vstack([self.y0, np.zeros((n-1, self.y0.shape[0]))])
        v = np.vstack([self.velocity_t0, np.zeros((n-1, self.velocity_t0.shape[0]))])
        vx = np.vstack([self.x_velocity_t0, np.zeros((n-1, self.x_velocity_t0.shape[0]))])
        vy = np.vstack([self.y_velocity_t0, np.zeros((n-1, self.y_velocity_t0.shape[0]))])
        ax = np.vstack([self.x_acceleration_t0, np.zeros((n-1, self.x_acceleration_t0.shape[0]))])
        ay = np.vstack([self.y_acceleration_t0, np.zeros((n-1, self.y_acceleration_t0.shape[0]))])

        for t in range(1, n):
            p = t-1
            # project new x & ys given previous conditions
            x[t] = x[p] + vx[p] * dt + 0.5 * ax[p] * dt * dt
            y[t] = y[p] + vy[p] * dt + 0.5 * ay[p] * dt * dt
            # new conditions
            vx[t] = vx[p] + ax[p] * dt
            vy[t] = vy[p] + ay[p] * dt
            v[t] = np.sqrt(vx[t]**2 + vy[t]**2)
            dragForceX = 0.5 * self.air_density * self.area * self.drag_coefficient * v[p] * vx[p]
            dragForceY = 0.5 * self.air_density * self.area * self.drag_coefficient * v[p] * vx[p]
            ax[t] = -dragForceX / self.mass
            ay[t] = -ACCELERATION_DUE_TO_GRAVITY - dragForceY / self.mass

        return np.stack([x, y, v, vx, vy, ax, ay])
    def self_x_dist(max_x):
        pass





x_min = 0
x_max=0
x_n = 1
possible_x = np.linspace(x_min, x_max, x_n, dtype='float32')

y_min = 4
y_max=7
y_n = 7
possible_y = np.linspace(y_min, y_max, y_n, dtype='float32')
print "Ys are: %s" % possible_y

launch_angle_min = -2.
launch_angle_max = 2.
launch_angle_n = 21
possible_launch_angle = np.linspace(launch_angle_min, launch_angle_max, launch_angle_n, dtype='float32')

velo_min = 30
velo_max = 105
velo_n = (velo_max - velo_min)*2
possible_velo = np.linspace(velo_min, velo_max, velo_n, dtype='float32')

# this takes cartesian product and outputs same values in same positions that go in, multiplied out for every combo
x, y, launch_angle, velo = cartesian([possible_x, possible_y, possible_launch_angle, possible_velo]).T

"""Baseball Coefficients:

diameter: 74.68 mm
area: 4380.2459820395015 mm
mass: between 141.75 to 148.83 g
"""
mass = np.mean([141.75, 148.83]) / 1000. # kg
area = ((74.68 / 1000.)**2) * (np.pi / 4) # m^2
drag_coefficient = 0.3
adensity = air_density(181.051)  # elevation of Chicago


traj = Trajectory(x, y, velo, launch_angle, adensity, area, drag_coefficient, mass)

results = traj.solve_n_steps(240*3, 1/240.)

# floor the y values
# cap the x values
# make these missing values