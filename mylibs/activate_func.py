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
        floatX = theano.config.floatX

        self.idim = weight.shape[0]
        self.odim = weight.shape[1]

        # Allocate the memory of parameter.
        self.w    = theano.shared(value=weight)
        self.bias = theano.shared(value=bias)

        # Allocate the memory of parameter's difference.
        self.diffw    = theano.shared(
                value=numpy.zeros((self.idim,self.odim), dtype=floatX))
        self.diffbias = theano.shared(
                value=numpy.zeros(self.idim, dtype=floatX))

    # Transfer function.
    def forward(self, x):
        return T.dot(x, self.w) + self.bias
    
    def save(self, fname, acttype="linear"):
        numpy.savez(fname, type=acttype, w=self.w.get_value(), bias=self.bias.get_value())       

# Softmax class
class SoftmaxLayer(LinearLayer):
    def __init__(self, weight, bias):
        super(SoftmaxLayer, self).__init__(weight, bias)
    def forward(self, x):
        return T.nnet.softmax(super(SoftmaxLayer, self).forward(x))
    def save(self, fname):
        super(SoftmaxLayer, self).save(fname, "softmax")       


# Sigmoid class
class SigmoidLayer(LinearLayer):
    def __init__(self, weight, bias):
        super(SigmoidLayer, self).__init__(weight, bias)
    def forward(self, x):
        return T.nnet.sigmoid(super(SigmoidLayer, self).forward(x))
    def save(self, fname):
        super(SigmoidLayer, self).save(fname, "sigmoid")       
               

        

# Rectified Linear class.
class ReLULayer(LinearLayer):
    def __init__(self, weight, bias):
        super(ReLULayer, self).__init__(weight, bias)
    def forward(self, x):
        return T.max(super(ReLULayer, self).forward(x))
    def save(self, fname):
        super(ReLULayer, self).save(fname, "relu")       
