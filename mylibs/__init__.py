#!/usr/bin python

list2metavar(list):
    s = str(list)
    s.replace("'","").replace(", ", "/")
