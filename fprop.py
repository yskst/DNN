#!/usr/bin/python
# -*- config: utf-8 -*-

import sys
import fileinput
from optparse import OptionParser

import numpy as np
import theano
import theano.tensor as T
from mylibs import perceptron

def stderr(s):
    sys.stderr.write(s)

def save_le(f, data):
    """ save with little endian """
    data.tofile(f)

def save_be(f, data):
    """ save with big endian """
    data.byteswap()
    data.tofile(f)

def load_data(format, fname, visnum):
    
    if format == 'npy':
        data = load(fname)

        if data.shape()[0] != visnum:
            stderr("Dimension miamatch between training data and visnum.\n")
            sys.exit(1)
        return data
    else:
        if   format == 'f4be': type='>f4'
        elif format == 'f4le': type='<f4'
        elif format == 'f4ne': type='=f4'
        return np.fromfile(fname, dtype=type).reshape(-1, visnum)

if __name__=='__main__':
    floatX = theano.config.floatX
    # Option analysis.
    usage="%prog [options] file"
    desc ="Training Restricted Boltzmann machine."
    op = OptionParser(usage=usage, description=desc)
    
    # Required option.
    op.add_option("--rbm", action="store", dest="rbm", metavar="FILE",
            type="string", help="Trained RBM file.")
    op.add_option("-o",  "--of", action="store", dest="output",
            type="string", help="output file name which is npz format")
    op.add_option("--df", action="store",dest="df",
            type="choice", choices=["f4ne", "f4be", "f4le", "npy"], metavar="[f4ne/f4be/f4le/npy]",
            help="sample data is raw 4byte float format with native/big/little endian or npy(npz) format.")
    op.add_option("--ot", action="store", dest="ot",
            type="choice", choices=["f4ne", "f4be", "f4le"], metavar="[f4ne/f4be/f4le]",
            help="The format of output data.")

    # Optional option.
    op.add_option("--mb", action="store", dest="mb", default=512,
            type="int",   help="The size of minibatch.[default=%default]")

    (options, args) = op.parse_args()
    mbsize = int(options.mb)

    # Load training data.
    nn = perceptron.load(options.rbm)
    data = load_data(options.df, args[0], nn.idim)
    
    x = T.fmatrix("x")
    y = nn.forward(x)
    
    fpropagation = theano.function(inputs=[x], outputs=y)

    mbnum = len(data)/mbsize
    of = open(options.output, 'wb')
    
    if options.ot == 'f4be': save = save_be
    else                   : save = save_le

    for b in range(mbnum):
        o = fpropagation(data[b*mbsize:(b+1)*mbsize])
        save(of, o)

    of.close()
