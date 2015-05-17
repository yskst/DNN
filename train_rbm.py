#!/usr/bin/python
# -*- coding: utf-8 -*-

import fileinput
from optparse import OptionParser

import numpy as np
import Theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStream as RandomStreams
import lib.activate_func
        

if __name__=='__main__':

    # Option analysis.
    usage="%prog [options] files..."
    desc ="Training Restricted Boltzmann machine."
    op = OptionParser(usage=usage, decription=desc)
    
    # Required option.
    op.add_option("-of",  "--output", action="store", dest="output",
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
    op.add_option("-af", "--activate-function", action="store",dest="af",
            type="choice", choices=activate_functions, 
            help="Activete function.")

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
    mbsize = options.mb

    # Load training data.
    data = np.loadtxt(fileinput(args)).astype(theano.config.floatX)
    visnum = data.shape[1]
    hidnum = options.hidnum

    # Load Initial value of weight and bias.
    # If not specified, create it by random value.
    if options.weight:
        w = np.loadtxt(options.weight)
    else:
        w = np.random.RandomState(options.seed).uniform(
                low=-4.0*np.sqrt(6.0/(hidnum+visnum)),
                high=4.0*np.sqrt(6.0/(hidnum+visnum)),
                size=(visnum, hidnum), dtype=theano.config.floatX)
    if options.bias:
        b = np.loadtxt(option.bias)
    else:
        b = np.zeros(hidnum, dtype=theano.config.floatX)

    af = activate_generetor(options.af, w, b)
    vbias     = theano.shared(
                    value=numpy.zeros(visnum, dtype=theano.coding.floatX))
    diffvbias = theano.shared(
                    value=numpy.zeros(visnum,dtype=theano.coding.floatX))

    
    # Create a formula of propagation.

    gibbs_rng=RandomStreams(
            numpy.random.RandomState(options.seed).randint(2**30))

    v0act=T.fmatrix("v0act")
    h0act=af.forward(v0act)
    h0smp=gibbs_rng.binomial(
        size=(mbsize, hidnum),n=1,p=h0act,dtype=theano.coding.floatX)
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
            (af.diffw, -lr*grad_w     +mm*af.diffW    -re*af.w),
            (af.bias,  -lr*grad_hbias +mm*af.diffbias         ),
            (diffvbias,-lr*grad_vbias +mm*diffvbias           )]

    updates_update=[
            (af.w,    af.w   +af.diffw    ),
            (af.bias, af.bias+af.diffhbias) ,
            (vbias,   vbias  +diffvbias)]

    mse=T.mean((v0act-v1act)**2)

    trainer_diff  =theano.function(inputs=[v0act],
            outputs=mse,
            updates=updates_diff)

    trainer_update=theano.function(inputs=[],
            outputs=None,
            updates=updates_update)


