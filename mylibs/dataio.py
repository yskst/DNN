#!/usr/bin/python
# -*- coding: utf-8 -*-

""" The library for data IO. """

import fileinput
import numpy as np


def __parse_fmt__(fmt):
    r = ""
    e = fmt[2:4]
    if   e == "le": r = "<"
    elif e == "be": r = ">"
    elif e == "ne": r = "="
    else:
        raise NameError("The encoding %s is unknown." % fmt)
    return r + fmt[0:2]
    


def load_data(fname, fmt):
    if fmt == 'npy':
        return np.load(fname)
    elif fmt == 'text':
        return np.loadtxt(fname)
    else:
        dt = np.dtype(__parse_fmt__(fmt))
        return np.fromfile(fname, dtype=dt)

