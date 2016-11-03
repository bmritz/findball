import h5py, logging, os
from brutils.oututils import ensure_filepath

DATA_FILE = os.path.join(os.path.abspath(__file__), "../../data/DATA.hdf5")

HDF5_NAME = os.path.abspath(DATA_FILE)

ensure_filepath(HDF5_NAME)

DATA = h5py.File(HDF5_NAME)

def delete_if_exists(ds_name, group=None):
    if group is None:
        try:
            del DATA[ds_name]
        except KeyError:
            pass
        else:
            logging.info("Deleted dataset %s from %s because it will be replaced by this run." % (ds_name, HDF5_NAME))
    else:
        grp = DATA[group]
        try: 
            del grp[ds_name]
        except KeyError:
            pass
        else:
            logging.info("Deleted dataset %s from h5file: %s, group: %s because it will be replaced by this run." % (ds_name, HDF5_NAME, group))
