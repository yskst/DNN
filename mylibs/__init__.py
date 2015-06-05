#!/usr/bin python

def list2metavar(list):
    """ 
    Create string for metavar from List of choices.
    Parameters:
        list: the list of choices.

    Return:
        the created string.
    """
    s = '['
    for l in list:
        s += "%s/" % str(l)
    s[-1] = ']'
    return s
