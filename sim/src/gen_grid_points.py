import numpy as np

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
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def gen_linspace(minimum, maximum, step=1, **kwargs):
    """get linspace between min and max inclusive by step"""
    assert maximum >= minimum, "maximum must be >= minimum"
    assert step > 0, "This function only counts upwards"
    n = ((maximum - minimum) / step) + 1
    return np.linspace(minimum, maximum, n, **kwargs)

def gen_grid_points(ranges, **kwargs):
    """ ranges is a list of tuples: (min, max, step)"""
    all_ranges = []
    for tup in ranges:
        rng = gen_linspace(tup[0], tup[1], tup[2], **kwargs)
        all_ranges.append(rng)

    return cartesian(all_ranges), all_ranges
