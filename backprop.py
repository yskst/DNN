#!/usr/bin/python
# -*- config: utf-8 -*-

import sys
import time

from fileinput import FileInput
from optparse import OptionParser

import numpy as np
import theano
import theano.tensor as T
from mylibs import *
from mylibs import mlp
from mylibs import dataio

def shuffle(dat, tar):
    state = np.random.get_state()
    np.random.shuffle(dat)
    np.random.set_state(state)
    np.random.shuffle(tar)
    return

if __name__=='__main__':
    floatX = theano.config.floatX
    # Option analysis.
    usage="""%prog [op] [data tarel]...\n
            [data tarel] must be an even number.The Even-numbered is the input and the odd-numbered is the output correspondig to the input."""
    desc ="Training DNN with backpropagation."
    op = OptionParser(usage=usage, description=desc)
    
    # Required option.
    op.add_option("-l", "--init-layer", action="store", dest="layer",
            metavar="FILE", type="string", help="Initial value of MLP")
    op.add_option("-o",  "--output", action="store", dest="output",
            metavar="FILE", type="string",
            help="output file name which is npz format")
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
    op.add_option("--df", action="store", dest="df",
            type="string", help="the format of training data.")
    op.add_option("--tf", action="store", dest="tf",
            type="string", help="the format of target data.")
    op.add_option("--ot", action="store", dest="ot",
            type="choice", choices=['c','f'], metavar='f/c',
            help="The target type is feature or category.")

    # Optional option.
    op.add_option("--seed", "--random-seed", action="store", dest="seed",
            type="int", default=1234,
            help="Random seed. [default=%default]")

    
    (op, args) = op.parse_args()
    mbsize = int(op.mb)
    lr     = float(op.lr)
    mm     = float(op.mm)
    re     = float(op.re)
    seed   = int(op.seed)


    # load data.
    m   = mlp.load(op.layer)
    dat = dataio.load_data(args[0], op.df).reshape(-1, m[0].idim).astype(theano.config.floatX)
    tar = dataio.load_data(args[1], op.tf).reshape(-1, m[-1].odim)

    # Allocate memory to update
    diffw    = []
    diffbias = []
    for i in m:
        shape = i.w.get_value().shape
        diffw.append(
                theano.shared(value=np.zeros(shape[0]*shape[1], dtype=floatX).reshape(shape)))
        diffbias.append(
                theano.shared(value=np.zeros(shape[1], dtype=floatX)))

    # Formula to calcurate cost which is cross entropy.
    if op.ot == 'c':
        tar  = np.asarray(tar, dtype=np.int32)
        y    = T.ivector("y")
        cost = T.sum(T.nnet.categorical_crossentropy(m.f, y)) / mbsize
    elif op.ot == 'f':
        tar  = tar.astype(np.float32)
        y    = T.fmatrix("y")
        cost = T.sum((m.f-y)*(m.f-y)) / mbsize
    
    # Formula to calcurate gradient.
    temp=[]
    for layer in m:
        temp.append(layer.w)
        temp.append(layer.bias)
    grads = T.grad(cost, temp)


    # Formula to training.
    update_diff=[]
    for i,layer in enumerate(m):
        update_diff.append((diffw[i],    -lr*(grads[i*2]   + re*layer.w  ) + mm*diffw[i]   ))
        update_diff.append((diffbias[i], -lr*(grads[i*2+1] + re*layer.bias)+ mm*diffbias[i]))

    update_update=[]
    for i,layer in enumerate(m):
        update_update.append((layer.w,   layer.w    + diffw[i]   ))
        update_update.append((layer.bias,layer.bias + diffbias[i]))
    
    x = m.x
    trainer_diff  = theano.function(inputs=[x,y], outputs=cost, updates=update_diff)
    trainer_update= theano.function(inputs=[],   outputs=None, updates=update_update)

    # Formula to eval.
    if op.ot == 'c':
        if m[-1].odim == 1: err=T.sum(T.neq(T.ge(m.f, 0.5), y))
        else              : err=T.sum(T.neq(T.argmax(m.f, axis=1),y))
        tester=theano.function(inputs=[x,y], outputs=err)
    elif op.ot == 'f':
        tester = theano.function(intput=[x,y], outputs=cost)

    np.random.seed(seed)
    mbnum = int(dat.shape[0] / mbsize)
    for i in range(seed):
        shuffle(dat, tar)

        c=0.0
        e=0.0
        for b in range(mbnum):
            c += trainer_diff(dat[mbsize*b:mbsize*(b+1)], tar[mbsize*b:mbsize*(b+1)])
            trainer_update()
            e += tester(dat[mbsize*b:mbsize*(b+1)], tar[mbsize*b:mbsize*(b+1)])
        
        c/=mbnum*mbsize
        e/=mbnum*mbsize

        sys.stdout.write("%4d ephoch, cost= %0.8e mse= %0.8e\n" % (i, c, e))
    m.save(op.output)

