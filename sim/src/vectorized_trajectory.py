import numpy as np
from gen_grid_points import gen_grid_points
import argparse, json, os, math

ACCELERATION_DUE_TO_GRAVITY = 9.8

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
    def __init__(self,x0, y0, v0, launch_angle,
        air_density, area, drag_coefficient, mass, x_acceleration_t0=None, y_acceleration_t0=None):

        """ x0 and y0 in meters, v0 in m/s, launch_angle in degrees"""
        self.x0 = x0
        self.y0 = y0
        self.velocity_t0 = v0
        self.launch_angle_deg_t0 = launch_angle
        self.launch_angle_t0 = np.deg2rad(self.launch_angle_deg_t0)

        self.x_velocity_t0 = self.velocity_t0 * np.cos(self.launch_angle_t0)
        self.y_velocity_t0 = self.velocity_t0 * np.sin(self.launch_angle_t0)

        self.x_acceleration_t0 = np.zeros_like(self.x0) if x_acceleration_t0 is None else x_acceleration_t0
        self.y_acceleration_t0 = np.full(self.x0.shape, -ACCELERATION_DUE_TO_GRAVITY) if y_acceleration_t0 is None else y_acceleration_t0

        self.air_density = air_density
        self.area = area
        self.drag_coefficient = drag_coefficient
        self.mass = mass
        self.n_runs = self.x0.shape[0]

        print self.y0.shape

    def solve_n_steps(self, n, dt):
        """
        approx. time:
        tried on set of 10:
        ((245 microseconds) / 10) * 1 000 000 000 =
        6.80555556 hours

        returns an array of pitches

        so arr[0,:,:] is the data for pitch 0
        arr[0,:,0] is the time for pitch 0 across time
        arr[0,:,1] is the x coordinate across time for pitch 0
        arr[0,:,2] is the y coordinate across time for pitch 0

        """

        # axis 1: time
        # axis 2: which pitch

        # initalize output array
        out = np.zeros(shape=(self.n_runs, n, 8))
        #res_time = np.arange(n)*dt
        # set initial conditions for time 0

        out[:,:,0] = np.arange(n)*dt
        out[:,0,1] = self.x0
        out[:,0,2] = self.y0
        out[:,0,3] = self.velocity_t0
        out[:,0,4] = self.x_velocity_t0
        out[:,0,5] = self.y_velocity_t0
        out[:,0,6] = self.x_acceleration_t0
        out[:,0,7] = self.y_acceleration_t0

        #x = np.vstack([self.x0, np.zeros((n-1, self.x0.shape[0]))])
        #y = np.vstack([self.y0, np.zeros((n-1, self.y0.shape[0]))])
        #v = np.vstack([self.velocity_t0, np.zeros((n-1, self.velocity_t0.shape[0]))])
        #vx = np.vstack([self.x_velocity_t0, np.zeros((n-1, self.x_velocity_t0.shape[0]))])
        #vy = np.vstack([self.y_velocity_t0, np.zeros((n-1, self.y_velocity_t0.shape[0]))])
        #ax = np.vstack([self.x_acceleration_t0, np.zeros((n-1, self.x_acceleration_t0.shape[0]))])
        #ay = np.vstack([self.y_acceleration_t0, np.zeros((n-1, self.y_acceleration_t0.shape[0]))])

        for t in range(1, n):
            p = t-1
            # project new x & ys given previous conditions
            #x[t] = x[p] + vx[p] * dt + 0.5 * ax[p] * dt * dt
            #y[t] = y[p] + vy[p] * dt + 0.5 * ay[p] * dt * dt

            out[:,t,1] = out[:,p,1] + out[:,p,4] * dt + 0.5 * out[:,p,6] * dt * dt
            out[:,t,2] = out[:,p,2] + out[:,p,5] * dt + 0.5 * out[:,p,7] * dt * dt
            # new conditions
            #vx[t] = vx[p] + ax[p] * dt
            #vy[t] = vy[p] + ay[p] * dt
            #v[t] = np.sqrt(vx[t]**2 + vy[t]**2)
            #dragForceX = 0.5 * self.air_density * self.area * self.drag_coefficient * v[p] * vx[p]
            #dragForceY = 0.5 * self.air_density * self.area * self.drag_coefficient * v[p] * vx[p]
            #ax[t] = -dragForceX / self.mass
            #ay[t] = -ACCELERATION_DUE_TO_GRAVITY - dragForceY / self.mass

            out[:,t,4] = out[:,p,4] + out[:,p,6] * dt
            out[:,t,5] = out[:,p,5] + out[:,p,7] * dt
            dragForceX = 0.5 * self.air_density * self.area * self.drag_coefficient * out[:,p,2] * out[:,p,4]
            dragForceY = 0.5 * self.air_density * self.area * self.drag_coefficient * out[:,p,2] * out[:,p,5]
            out[:,t,6] = -dragForceX / self.mass
            out[:,t,7] = -ACCELERATION_DUE_TO_GRAVITY - dragForceY / self.mass

        return out
        #return np.stack([x, y, v, vx, vy, ax, ay])

def load_conf(filname):
    with open(filname, 'r') as fil:
        conf = json.load(fil)
    return conf

def conditions_from_conf(conf):

    grid_spec = conf['GRID_SPEC']
    # append x axis always 0
    names = ['x0']
    ranges = [[0,0,1]]

    for spec in grid_spec:
        names.append(spec['name'])
        if spec['name'] == 'y0':
            # convert units
            # 1 ft = 0.3048 m
            ranges.append([0.3048 * ft for ft in spec['spec']])
        elif spec['name'] == 'v0':
            #  1 mph = 0.44704 m/s
            ranges.append([0.44704 * mph for mph in spec['spec']])
        else:
            ranges.append(spec['spec'])

    initial_conditions = gen_grid_points(ranges, dtype='float32')

    initial_conditions = dict(zip(names, initial_conditions.T))
    """Baseball Coefficients:

    diameter: 74.68 mm
    area: 4380.2459820395015 mm
    mass: between 141.75 to 148.83 g
    """

    initial_conditions["mass"] = np.mean([141.75, 148.83]) / 1000. # kg
    initial_conditions['area'] = ((74.68 / 1000.)**2) * (np.pi / 4) # m^2
    initial_conditions['drag_coefficient'] = 0.3 # units?
    initial_conditions['air_density'] = air_density(181.051)  # elevation of Chicago
    return initial_conditions


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="configuration file")
    args = parser.parse_args()

    # import config
    conf_filname = args.config or "config_vectorized_trajectory.json"
    
    conf = load_conf(conf_filname)

    initial_conditions = conditions_from_conf(conf)

    traj = Trajectory(**initial_conditions)
    attrs = vars(traj)
    # {'kids': 0, 'name': 'Dog', 'color': 'Spotted', 'age': 10, 'legs': 2, 'smell': 'Alot'}
    # now dump this in some way or another
    print ',\n'.join("%s: %s" % item for item in attrs.items())
    results = traj.solve_n_steps(240*3, 1/240.)

    print results.shape