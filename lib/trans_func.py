#!/usr/bin/python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy

# This is the libraries in transfer functions.

class LinearLayer(object):

    # Constructer
    def __init__(self, weight, bias):
        self.idim = weight.shape[0]
        self.odim = weight.shape[1]

        # Allocate the memory of parameter.
        self.w    = theano.shared(value=weight, name="w")
        self.bias = theano.shared(value=bias, name="bias")

        # Allocate the memory of parameter's difference.
        self.diffw    = theano.shared(
                value=numpy.zeros((self.idim,self.odim), dtype=theano.config.floatX),
                name="diffw")
        self.diffbias = theano.shared(
                value=numpy.zeros(self.idim, dtype=theano.config.floatX),
                name="diffbias")

    # Transfer function.
    def forward(self, x):
        return T.dot(x, self.w) + self.bias


# Softmax class
def SoftmaxLayer(LinearLayer):
    def forward(self, x):
        return T.nnet.softmax(super(SoftmaxLayer, self).forward(x))

# Sigmoid class
def SigmoidLayer(LinearLayer):
    def forward(self, x):
        return T.nnet.sigmoid(super(SigmoidLayer, self).forward(x))

# Rectified Linear class.
def ReLULayer(LinearLayer):
    def forward(self, x):
        return T.max(super(ReLULayer, self).forward(x))
