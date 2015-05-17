#!/usr/bin/python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy

# This is the libraries in activatefer functions.

activate_functions = ["linear", "sigmoid", "softmax", "relu"]

def activate_generetor(func, weight, bias):
    if func == "linear":
        return LinearLayer(weight, bias)
    elif func == "sigmoid":
        return SigmoidLayer(weight, bias)
    elif func == "softmax":
        return SoftmaxLayer(weight, bias)
    elif func == "relu":
        return ReLULayer(weight, bias)
    else:
        raise NameError(func + ' is not defines as activatefer function.')


class LinearLayer(object):

    # Constructer
    def __init__(self, weight, bias):
        self.idim = weight.shape[0]
        self.odim = weight.shape[1]

        # Allocate the memory of parameter.
        self.w    = theano.shared(value=weight)
        self.bias = theano.shared(value=bias)

        # Allocate the memory of parameter's difference.
        self.diffw    = theano.shared(
                value=numpy.zeros((self.idim,self.odim), dtype=theano.config.floatX))
        self.diffbias = theano.shared(
                value=numpy.zeros(self.idim, dtype=theano.config.floatX))

    # Transfer function.
    def forward(self, x):
        return T.dot(x, self.w) + self.bias
    
    def save(self, fname):
        numpy.savez(fname, type="linear", w=self.w, bias=self.bias)       

# Softmax class
def SoftmaxLayer(LinearLayer):
    def forward(self, x):
        return T.nnet.softmax(super(SoftmaxLayer, self).forward(x))
    def save(self, fname):
        numpy.savez(fname, type="softmax", w=self.w, bias=self.bias)       


# Sigmoid class
def SigmoidLayer(LinearLayer):
    def forward(self, x):
        return T.nnet.sigmoid(super(SigmoidLayer, self).forward(x))
    def save(self, fname):
        numpy.savez(fname, type="sigmoid", w=self.w, bias=self.bias)       

        

# Rectified Linear class.
def ReLULayer(LinearLayer):
    def forward(self, x):
        return T.max(super(ReLULayer, self).forward(x))
    def save(self, fname):
        numpy.savez(fname, type="softmax", w=self.w, bias=self.bias)       
