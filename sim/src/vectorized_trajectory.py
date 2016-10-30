import numpy as np
from logutils import setup_logging
from gen_grid_points import gen_grid_points
from output_conf import DATA, delete_if_exists
import pandas as pd
import argparse, json, os, math, logging


LOG = setup_logging()
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
    def __init__(self,name, x0, y0, v0, launch_angle,
        air_density, area, drag_coefficient, mass, x_acceleration_t0=None, y_acceleration_t0=None):

        # write the initial conditions to a group in our data that is specific for this trajectory
        self.n_runs = x0.shape[0]
        self.name = name
        self.dtype = x0.dtype
        self.itemsize = x0.dtype.itemsize

        try:
            self.DATA = DATA[name]
        except KeyError:
            self.DATA = DATA.create_group(name)

        delete_if_exists('initial_conditions', name)
        self.initial_conditions = self.DATA.create_dataset('initial_conditions', 
            shape = (self.n_runs, 4), 
            dtype=self.dtype)
        self.initial_conditions[:,0] = x0
        self.initial_conditions[:,1] = y0
        self.initial_conditions[:,2] = v0
        self.initial_conditions[:,3] = launch_angle

        # store metadata
        self.DATA.attrs['air_density'] = air_density
        self.DATA.attrs['mass'] = mass
        self.DATA.attrs['area'] = area
        self.DATA.attrs['drag_coefficient'] = drag_coefficient

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

        LOG.info("Trajectory Created with %s pitches to be calculated." % self.n_runs)

    def solve_n_steps(self, n, dt, max_bytes=0.5e+9):
        """
        returns an array of pitches

        so arr[0,:,:] is the data for pitch 0
        arr[0,:,0] is the time for pitch 0 across time
        arr[0,:,1] is the x coordinate across time for pitch 0
        arr[0,:,2] is the y coordinate across time for pitch 0

        """

        # axis 1: time
        # axis 2: which pitch

        # if our number of bytes will be too large, then iterate over chunks
        matrix_size = self.n_runs * n * 8 

        LOG.info("There are %s entries in the solution matrix. It will be %s GB uncompressed." %\
         (matrix_size, matrix_size*self.itemsize*1e-9))

        n_chunks = ((matrix_size * 8) // max_bytes) + int(matrix_size % max_bytes > 1)
        LOG.info("Breaking the calculation into %s chunks" % int(n_chunks))

        chunk_bounds = np.array([int(x) for x in np.linspace(0, self.n_runs, n_chunks+1)])
        LOG.info("Chunk bounds are %s" % str(chunk_bounds))
        # we will insert into and return this solution array
        #self.solution = np.zeros(shape = (self.n_runs, n, 4), dtype = 'float32')
        delete_if_exists("ball_trajectories", self.name)
        self.solution = self.DATA.create_dataset("ball_trajectories", 
            shape = (self.n_runs, n, 8), 
            dtype=self.dtype)

        # NOTE: on-disk way -- much slower but likely less memory
        # NOTE: keeping this here in case we implement dask
        # self.solution[:,:,0] = np.arange(n)*dt
        # self.solution[:,0,1] = self.x0
        # self.solution[:,0,2] = self.y0
        # self.solution[:,0,3] = self.velocity_t0
        # self.solution[:,0,4] = self.x_velocity_t0
        # self.solution[:,0,5] = self.y_velocity_t0
        # self.solution[:,0,6] = self.x_acceleration_t0
        # self.solution[:,0,7] = self.y_acceleration_t0

        # chunking the pitches (first dimension)
        for lbound, ubound in zip(chunk_bounds[:-1], chunk_bounds[1:]):
            LOG.info("now running from %s to %s" % (lbound, ubound))

            # initalize output array
            out = np.zeros(shape=(ubound-lbound, n, 8))

            # set initial conditions for time 0
            out[:,:,0] = np.arange(n)*dt
            out[:,0,1] = self.x0[lbound:ubound]
            out[:,0,2] = self.y0[lbound:ubound]
            out[:,0,3] = self.velocity_t0[lbound:ubound]
            out[:,0,4] = self.x_velocity_t0[lbound:ubound]
            out[:,0,5] = self.y_velocity_t0[lbound:ubound]
            out[:,0,6] = self.x_acceleration_t0[lbound:ubound]
            out[:,0,7] = self.y_acceleration_t0[lbound:ubound]

            # one by one way -- moved on from this 
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
                # one by one way -- moved on from this 
                #x[t] = x[p] + vx[p] * dt + 0.5 * ax[p] * dt * dt
                #y[t] = y[p] + vy[p] * dt + 0.5 * ay[p] * dt * dt

                # new conditions
                #vx[t] = vx[p] + ax[p] * dt
                #vy[t] = vy[p] + ay[p] * dt
                #v[t] = np.sqrt(vx[t]**2 + vy[t]**2)

                #dragForceX = 0.5 * self.air_density * self.area * self.drag_coefficient * v[p] * vx[p]
                #dragForceY = 0.5 * self.air_density * self.area * self.drag_coefficient * v[p] * vx[p]
                #ax[t] = -dragForceX / self.mass
                #ay[t] = -ACCELERATION_DUE_TO_GRAVITY - dragForceY / self.mass

                # iterate over memory way
                out[:,t,1] = out[:,p,1] + out[:,p,4] * dt + 0.5 * out[:,p,6] * dt * dt
                out[:,t,2] = out[:,p,2] + out[:,p,5] * dt + 0.5 * out[:,p,7] * dt * dt

                out[:,t,4] = out[:,p,4] + out[:,p,6] * dt
                out[:,t,5] = out[:,p,5] + out[:,p,7] * dt

                out[:,t,3] =  np.sqrt(out[:,t,4]**2 + out[:,t,5]**2)
                dragForceX = 0.5 * self.air_density * self.area * self.drag_coefficient * out[:,p,2] * out[:,p,4]
                dragForceY = 0.5 * self.air_density * self.area * self.drag_coefficient * out[:,p,2] * out[:,p,5]
                out[:,t,6] = -dragForceX / self.mass
                out[:,t,7] = -ACCELERATION_DUE_TO_GRAVITY - dragForceY / self.mass


                # on-disk way -- keeping this here in case we implement dask
                # self.solution[lbound:ubound,t,1] = self.solution[lbound:ubound,p,1] + self.solution[lbound:ubound,p,4] * dt + 0.5 * self.solution[lbound:ubound,p,6] * dt * dt
                # self.solution[lbound:ubound,t,2] = self.solution[lbound:ubound,p,2] + self.solution[lbound:ubound,p,5] * dt + 0.5 * self.solution[lbound:ubound,p,7] * dt * dt
                
                # self.solution[lbound:ubound,t,4] = self.solution[lbound:ubound,p,4] + self.solution[lbound:ubound,p,6] * dt
                # self.solution[lbound:ubound,t,5] = self.solution[lbound:ubound,p,5] + self.solution[lbound:ubound,p,7] * dt

                # self.solution[lbound:ubound,t,3] =  np.sqrt(self.solution[lbound:ubound,t,4]**2 + self.solution[lbound:ubound,t,5]**2)
                # dragForceX = 0.5 * self.air_density * self.area * self.drag_coefficient * self.solution[lbound:ubound,p,2] * self.solution[lbound:ubound,p,4]
                # dragForceY = 0.5 * self.air_density * self.area * self.drag_coefficient * self.solution[lbound:ubound,p,2] * self.solution[lbound:ubound,p,5]
                # self.solution[lbound:ubound,t,6] = -dragForceX / self.mass
                # self.solution[lbound:ubound,t,7] = -ACCELERATION_DUE_TO_GRAVITY - dragForceY / self.mass

            self.solution[lbound:ubound,:] = out[:,:]
            #del out
        return self.solution
        #return np.stack([x, y, v, vx, vy, ax, ay])

        def dump_result(directory, mode='a'):
            df = pd.DataFrame(self.solution)

def load_conf(filname):
    with open(filname, 'r') as fil:
        conf = json.load(fil)
    return conf

def conditions_from_conf(conf, **kwargs):

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

    initial_conditions, all_ranges  = gen_grid_points(ranges, **kwargs)

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

    parser.add_argument("-o", "--output", type=str, help="output file")
    args = parser.parse_args()

    # import config
    conf_filname = args.config or "config_vectorized_trajectory.json"
    output_filename = args.output or "results.npz"

    conf = load_conf(conf_filname)
    LOG.info("configuration: %s" % conf)
    initial_conditions = conditions_from_conf(conf)

    initial_conditions['name'] = 'trajectory_1'

    traj = Trajectory(**initial_conditions)
    attrs = vars(traj)
    traj.solve_n_steps(int(240.*1.2), 1/240.)
    # with open(output_filename, "w") as outfile:
    #     np.savez(outfile, info=traj.info, results=traj.solution)
    LOG.info("Script Finished")
