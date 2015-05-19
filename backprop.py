#!/usr/bin/python
# -*- config: utf-8 -*-

import sys
import time

from fileinput import FileInput
from optparse import OptionParser

import numpy as np
import theano
import theano.tensor as T
from mylibs import activate_func

if __name__=='__main__':

    # Option analysis.
    usage="%prog [options] [data label]...\n
            [data label] must be an even number.The Even-numbered is the input and the odd-numbered is the output correspondig to the input."
    desc ="Training DNN with backpropagation."
    op = OptionParser(usage=usage, description=desc)
    
    # Required option.
    op.add_option("-l", "--init-layer", action="append", dest="layers",
            metavar="FILE", type="string",
            help="Initial value of each layer. This option can use several times.
                  The decleared order is mutch to the each layer.")
    op.add_option("-o",  "--output", action="store", dest="output",metavar="FILE"
            type="string",help="output file name which is npz format")
    op.add_option("--lr", action="store", dest="lr",
            type="float", help="learning rate")
    op.add_option("--mm", action="store", dest="mm",
            type="float", help="momentum")
    op.add_option("--re", action="store", dest="re",
            type="float", help="L2-reguralizer")
    op.add_option("-e",  "--epoch", action="store", dest="epoch",
            type="int",   help="Training epoch.")
    op.add_option("--mb", action="store", dest="mb",
            type="int",   help="The size of minibatch.")
    
    # Optional option.
    op.add_option("--seed", "--random-seed", action="store", dest="seed",
            type="int", default=1234,
            help="Random seed. [default=%default]")

    
    (options, args) = op.parse_args()
    mbsize = int(options.mb)
    lr     = float(options.lr)
    mm     = float(options.mm)
    re     = float(options.re)
    seed   = int(options.seed)


    # load data.
    L = []
    for f in options.layers:
        L.append(activate_func.load(f))

    dat = np.loadtxt(FileInput(args[1::2]))
    tar = np.loadtxt(Fileinput(args[2::2]))

    # Formula to calcurate output of DNN.
    x = T.fmatrix("x")
    p = L[0].forward(x)
    for layer in L[1:]:
        p = layer.forward(p)

    # Formula to calcurate cost which is cross entropy.
    y = 
    


