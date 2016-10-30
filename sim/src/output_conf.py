import h5py
import logging

HDF5_NAME = "DATA.hdf5"

DATA = h5py.File(HDF5_NAME)

def delete_if_exists(ds_name):
    try:
        del DATA[ds_name]
    except KeyError:
        pass
    else:
        logging.info("Deleted dataset %s from %s because it will be replaced by this run." % (ds_name, HDF5_NAME))
