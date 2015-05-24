#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import inspect
import numpy
import theano
import theano.tensor as T


""" This is the libraries in activatefer functions. """


def load(fname):
    f = numpy.load(fname)
    return generetor(f['type'], f['w'], f['bias'])

def generetor(func, weight, bias):
    if func == "LinearLayer":
        return LinearLayer(weight, bias)
    elif func == "SigmoidLayer":
        return SigmoidLayer(weight, bias)
    elif func == "SoftmaxLayer":
        return SoftmaxLayer(weight, bias)
    elif func == "ReLULayer":
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

myself = sys.modules[__name__]
percptron_type = [n for n,m in inspect.getmembers(myself, inspect.isclass)]

