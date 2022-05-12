# -*- coding: utf-8 -*-
"""Collection of useful functions shared by other classes"""

import glob
import numpy as np


def get_list_of_files(directory,min_year,max_year):
    """
    Get a flattened list of all grib files within a directory within a certain time range
    The time range is read from the .grib file name.
    """
    globs_exprs = [directory+f'*_{i}_*.grib'.format(i) for i in np.arange(min_year, max_year+1)]
    list_of_files = [glob.glob(g) for g in globs_exprs]
    return sorted([item for sublist in list_of_files for item in sublist])