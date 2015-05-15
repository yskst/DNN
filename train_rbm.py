#!/usr/bin/python
# -*- coding: utf-8 -*-

import fileinput
from optparse import OptionParser

import Theano
import numpy as np
import lib.trans_func.*


if __name__=='__main__':

    # Option analysis.
    usage="%prog [options] files..."
    desc ="Training Restricted Boltzmann machine."
    op = OptionParser(usage=usage, decription=desc)
    
    # Required option.
    op.add_option("-of",  "--output", action="store", dest="output"
            type="string", help="output file name which is npz format")
    op.add_option("--hidnum", action="store", dest="hidnum",
            type="int", help="The number of hidden layer's perceptron")
    op.add_option("-lr",  "--learning-rate", action="store", dest="lr",
            type="float", help="learning rate")
    op.add_option("-mm",  "--momentum", action="store", dest="mm",
            type="float", help="momentum")
    op.add_option("-re",  "--regularize", action="store", dest="re",
            type="float", help="L2-reguralizer")
    op.add_option("-ep",  "--epoch", action="store", dest="epoch",
            type="int",   help="Training epoch.")
    op.add_option("-mb",  "--minibatch", action="store", dest="mb",
            type="int",   help="The size of minibatch.")
    op.add_option("--rbmtype", action="store", dest="rbmtype",
            type="choice", choices=["gb", "bb"],
            help="gb:gaussain-bernoulli, bb:bernoulli-bernoulli")

    # Optional option.
   op.add_option("-rs", "--random-seed", action="store", dest="seed",
           type="int", default=1234, 
           help="Random seed.")
   op.add_option("-iw", "--initial-weight", action="store", dest="weight",
           type="string", metavar="File", default=None, 
           help="The initial weight value.")
   op.add_option("-ib", "--initial-bias", action="store", dest="bias",
           type="string", metavar="FILE", default=None,
           help="The initial bias value.")

    (options, args) = op.parse_args()


    # LOad training data.
    data = np.loadtxt(fileinput(args)).astype(theano.config.floatX)
    visnum = data.shape[1]
    hidnum = options.hidnum




