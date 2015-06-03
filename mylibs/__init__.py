#!/usr/bin python

def list2metavar(list):
    s = str(list)
    s.replace("'","").replace(", ", "/")
