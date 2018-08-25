import numpy as np


def unique_rows(in_array):
    """
    This (quickly) finds the unique rows in an array

    Parameters
    ----------
    in_array: ndarray
        The array for which the unique rows should be found

    Returns
    -------
    u_return: ndarray
       Array with the unique rows of the original array.

    """
    # Sort input array
    order = np.lexsort(in_array.T)

    # Apply sort and compare neighbors
    x = in_array[order]
    diff_x = np.ones(len(x), dtype=bool)
    diff_x[1:] = (x[1:] != x[:-1]).any(-1)

    # Reverse sort and return unique rows
    un_order = order.argsort()
    diff_in_array = diff_x[un_order]
    return in_array[diff_in_array]
