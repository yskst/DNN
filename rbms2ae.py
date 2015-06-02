#!/usr/bin/python
# -*- config: utf-8 -*-

import sys
from optparse import OptionParser

import numpy as np
import theano
import theano.tensor as T
from mylibs import perceptron
from mylibs import mlp

def stderr(s):
    sys.stderr.write(s)

if __name__=='__main__':
    floatX = theano.config.floatX
    # Option analysis.
    usage="%prog [options] rbms"
    desc ="Training Restricted Boltzmann machine."
    op = OptionParser(usage=usage, description=desc)
    
    # Required option.
    op.add_option("-o",  "--of", action="store", dest="output",
            type="string", help="output file name which is npz format")

    # Load RBMs.
    (options, args) = op.parse_args()
    
    rbms = []
    for fname in args:
        rbms.append(perceptron.load(fname))

    args.reverse()
    for fname in args:
        r = perceptron.load(fname)
        rbms.append(r.inverse())

    m = mlp.mlp(rbms)
    m.save(options.output)
