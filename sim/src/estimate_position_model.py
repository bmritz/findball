import h5py
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K


f = h5py.File("../data/DATA.hdf5", "r")

n_samples, n_frames, n_pic_dims = f['projection_dat']['final_projections'].shape

velos = f['projection_dat']['velocities']
projections = f['projection_dat']['final_projections']
#reshape data 
BATCH_SIZE = 24
NB_EPOCH = 3

def prep_inputs_positions(lbound, ubound, position_dat, resolution=np.array((1280, 720)), n_frames=288, n_pic_dims=2):

    d = position_dat[lbound:ubound,:,:,:].astype("float32")
    d  = d / resolution.reshape(-1,1)

    return d

def batch_generator(h5_trajectories, h5_velos, batch_size):
    n_samples = h5_trajectories.shape[0]

    assert n_samples % batch_size == 0
    assert h5_trajectories.shape[0] == h5_velos.shape[0]

    while 1:

        for i in range(0,n_samples,batch_size):
            x_dat = prep_inputs_positions(i, i+batch_size, h5_trajectories)
            y_dat = h5_velos[i:i+batch_size]
            yield x_dat, y_dat



n_samples_train = int(((n_samples * TRAIN_SAMPLE)// BATCH_SIZE) * BATCH_SIZE)
n_samples_test = int(n_samples - n_trues)
to_sample = np.array([True]*n_samples_train + [False]*n_samples_test)
mask = np.random.choice(to_sample, size=n_samples, replace=False)

X = projections[:]

X = X.reshape(X.shape[0], n_frames, n_pic_dims, 1)
X = X[:,:,0:2,:]
X_train = X[mask,:,:,:]
X_test = X[np.invert(mask),:,:,:]

y = velos[:]
y_train = y[mask]
y_test = y[np.invert(mask)]


MAX_VELO = y_train.max()
MIN_VELO = y_train.min()
RANGE = MAX_VELO - MIN_VELO
y_mean = y_train.mean()
y_std = y_train.std()

y_train_norm = (y_train - y_mean) / ((MAX_VELO - MIN_VELO)/ 2.)
y_test_norm = (y_test - y_mean) / ((MAX_VELO - MIN_VELO)/ 2.)

#y_train_norm = (y_train - MIN_VELO) / RANGE
#y_test_norm = (y_test- MIN_VELO) / RANGE

y_train_norm2 = (MAX_VELO - y_train)

model = Sequential()
model.add(Convolution2D(1, 2, 2, border_mode='valid', input_shape=(n_frames, 2, 1)))
model.add(Activation('relu'))
#model.add(Flatten(input_shape=(n_frames, 2, 1)))
model.add(Dense(144/2))
#model.add(Dense(40))
model.add(Activation('tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(loss='mean_squared_error', optimizer='adam')

# gens_valid = batch_generator(X_test, y_test, BATCH_SIZE)
# gens = batch_generator(X_train, y_train, BATCH_SIZE)

model.fit(X_train, y_train_norm, batch_size=BATCH_SIZE, nb_epoch=5,
    verbose=1, validation_data=(X_test, y_test_norm))

model.fit_generator(gens, samples_per_epoch=n_samples_train, nb_epoch = NB_EPOCH, 
    #validation_data = (X_test, y_test), nb_val_samples = n_samples_test, 
    verbose=2)

def get_iterator(dim1, dim2, low, high):
    result = [(0,0), (0,0)]
    result.append((low // dim2, high // dim2))

def find_num(i, tup):
    if tup:
        val = (i // np.product(tup),)
        rem = i % np.product(tup)
        val = val + find_num(rem, tup[1:])
        return val
    else:
        return (i,)

def check(tup1, tup2):
    res = 0
    for i, x in enumerate(tup1):
        res = res + (x * np.product(tup2[i+1:]))
    return res


def it(x):

    ret = [0]*len(x)
    for i in range(np.product(x)):
        for j in range(1, len(x)):
            val = i // np.product(x[j:])
            rem = i % np.product(x[j:])