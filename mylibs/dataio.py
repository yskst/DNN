#!/usr/bin/python
# -*- coding: utf-8 -*-

""" The library for data IO. """

import fileinput
import numpy as np

format_dict = {'f4be':'>f4', 'f4le':'<f4', 'f4ne':'=f4', 'npy':None, 'text':None}

def load_data(fname, fmt):

    if fmt not in format_dict.keys():
        raise IOError('The format type %s is unknown.' %s fmt)

    if fmt == 'npy':
        return np.load(fname)
    elif fmt == 'text':
        return np.loadtxt(fname)
    else:
        return np.fromfile(fname, dtype=format_dict[$fmt])
