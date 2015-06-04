#!/usr/bin python

def list2metavar(list):
    s = str(list)
    return s.replace("'","").replace(", ", "/")
