#!/usr/bin/python
# -*- config: utf-8 -*-

import sys
import time

import fileinput
from optparse import OptionParser

import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from mylibs import activate_func

if __name__=='__main__':

    # Option analysis.
    usage="%prog [options] files..."
    desc ="Training Restricted Boltzmann machine."
    op = OptionParser(usage=usage, description=desc)
    
    # Required option.
    op.add_option("-o",  "--output", action="store", dest="output",
            type="string", help="output file name which is npz format")
    op.add_option("--hidnum", action="store", dest="hidnum",
            type="int", help="The number of hidden layer's perceptron")
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
    op.add_option("--rbmtype", action="store", dest="rbmtype",
            type="choice", choices=["gb", "bb"], metavar="[gb/bb]",
            help="gb:gaussain-bernoulli, bb:bernoulli-bernoulli")
    op.add_option("-a", "--activate-function", action="store",dest="af",
            type="choice", choices=activate_func.activate_functions, 
            help="Activete function.")

    # Optional option.
    op.add_option("--seed", "--random-seed", action="store", dest="seed",
            type="int", default=1234,
            help="Random seed. [default=%default]")
    op.add_option("-w", "--initial-weight", action="store", dest="weight",
            type="string", metavar="File", default=None, 
            help="The initial weight value.(Optional)")
    op.add_option("-b", "--initial-bias", action="store", dest="bias",
            type="string", metavar="FILE", default=None,
            help="The initial bias value.(Optional)")

    (options, args) = op.parse_args()
    mbsize = int(options.mb)
    lr     = float(options.lr)
    mm     = float(options.mm)
    re     = float(options.re)
    seed   = int(options.seed)

    # Load training data.
    data = np.loadtxt(fileinput.FileInput(args), dtype=theano.config.floatX).T
    visnum = data.shape[1]
    hidnum = int(options.hidnum)

    # Load Initial value of weight and bias.
    # If not specified, create it by random value.
    if options.weight:
        w = np.loadtxt(options.weight, dtype=theano.config.floatX)
    else:
        w = np.random.RandomState(seed).uniform(
                low=-4.0*np.sqrt(6.0/(hidnum+visnum)),
                high=4.0*np.sqrt(6.0/(hidnum+visnum)),
                size=(visnum, hidnum)).astype(theano.config.floatX)
    if options.bias:
        b = np.loadtxt(option.bias, dtype=theano.config.floatX)
    else:
        b = np.zeros(hidnum, dtype=theano.config.floatX)

    af = activate_func.activate_generetor(options.af, w, b)
    vbias     = theano.shared(
                    value=np.zeros(visnum, dtype=theano.config.floatX))
    diffvbias = theano.shared(
                    value=np.zeros(visnum,dtype=theano.config.floatX))

    
    # Create a formula of propagation.

    gibbs_rng=RandomStreams(
            np.random.RandomState(seed).randint(2**30))

    v0act=T.fmatrix("v0act")
    h0act=af.forward(v0act)
    h0smp=gibbs_rng.binomial(
        size=(mbsize, hidnum),n=1,p=h0act,dtype=theano.config.floatX)
    if options.rbmtype == 'gb':
        v1act=T.dot(h0smp, af.w.T) + vbias
    elif options.rbmtype == 'bb':
        v1act=T.nnet.sigmoid(T.dot(h0smp, af.w.T) + vbias)
    h1act=af.forward(v1act)

    # Create a formula of update.
    grad_w    = (T.dot(v1act.T,h1act)-T.dot(v0act.T,h0act))/mbsize
    grad_hbias= (T.sum(h1act,axis=0) -T.sum(h0act,axis=0) )/mbsize
    grad_vbias= (T.sum(v1act,axis=0) -T.sum(v0act,axis=0) )/mbsize

    updates_diff=[
            (af.diffw, -lr*grad_w     +mm*af.diffw    -re*af.w),
            (af.bias,  -lr*grad_hbias +mm*af.diffbias         ),
            (diffvbias,-lr*grad_vbias +mm*diffvbias           )]

    updates_update=[
            (af.w,    af.w   +af.diffw    ),
            (af.bias, af.bias+af.diffbias) ,
            (vbias,   vbias  +diffvbias)]

    mse=T.mean((v0act-v1act)**2)

    trainer_diff  =theano.function(inputs=[v0act],
            outputs=mse,
            updates=updates_diff)

    trainer_update=theano.function(inputs=[],
            outputs=None,
            updates=updates_update)

    np.random.seed(seed)
    mbnum=data.shape[0]/mbsize

    for e in range(options.epoch):
        # shuffle of training data
        t1 = time.clock()
        np.random.shuffle(data)

        # estimate by mini-batch
        err = 0.0
        for b in range(mbnum):
            err += trainer_diff(data[mbsize*b:mbsize*(b+1)])
            trainer_update()
        err/=mbnum
        t2 = time.clock()
        
        sys.stdout.write("%3d epoch, %e mse (%f sec)\n" % (e+1, err, t2-t1)) 

    # output model parameter.
    af.save(options.output)
