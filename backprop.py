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

def shuffle(dat, lab):
    state = np.random.get_state()
    np.random.shuffle(dat)
    np.random.set_state(state)
    np.random.shuffle(lab)
    return

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
    op.add_option("--ot", action="store", dest="ot",
            type="choice", choices=["c", "f"], metavar="c/f",
            help="output layer type is category or feature vector")
    
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
    if options.ot == 'c':
        y    = T.ivector("y")
        cost = T.sum(T.nnet.categorical_crossentropy(p, y)) / mbsize
    else if options.ot == 'f':
        y    = T.fmatrix("y")
        cost = T.sum((p-y)*(p-y)) / mbsize
    
    # Formula to calcurate gradient.
    temp=[]
    for layer in L:
        temp.append(layer.w)
        temp.append(layer.bias)
    grads = T.grad(cost, temp)


    # Formula to training.
    update_diff=[]
    for (i,layer) in enumerate(L):
        update_diff.append((layer.diffw,   -lr*(grads[i*2]  +re*layer.w   )+mm*layer.diffw   ))
        update_diff.append((layer.diffbias, -lr(grads[i*2+1]+re*layer.bais)+mm*layer.diffbias))

    update_update=[]
    for layer in L:
        update_update.append((layer.w,   layer.w    + layer.diffw   ))
        update_update.append((layer.bias,layer.bias + layer.diffbias))

    trainer_diff  = theano.function(inputs=[x,y], outputs=cost, updates=updates_diff)
    trainer_update= theano.function(intputs=[],   outputs=None, update=update_update)

    # Formula to eval.
    if options.ot == 'c':
        err=T.sum(T.neq(T.argmax(p,axis=1),y))
        tester=theano.function(inputs=[x,y], output=err)
    elif options.ot == 'f':
        tester = theano.function(intput=[x,y], output=cost)

    np.random.seed(seed)
    mbnum = dat.shape()[1] / mbsize

    for i in range(seed):
        shuffle(dat, lab)

        c=0.0
        e=0.0
        for b in range(mbsize):
            c += trainer_diff(dat[mbsize*b:(mbsize+1)*b], lab[mbsize*b:(mbsize+1)*b])
            trainer_update()
            e += tester(dat[mbsize*b:(mbsize+1)*b], lab[mbsize*b:(mbsize+1)*b])
        
        c/=mbnum*mbsize
        e/=mbnum*mbsize

        sys.stdout.write("%4d ephoch, cost= %0.8e mse= %0.8e\n" % (i, c, e))
