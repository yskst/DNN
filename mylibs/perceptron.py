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
    return generetor(f['type'], f['w'], f['bias'], f['vbias'])

def generetor(func, weight, hbias, vbias=None):
		return eval(func+'(weight,hbias,vbias)')



class LinearLayer(object):

    # Constructer
    def __init__(self, weight, hbias, vbias):
        floatX = theano.config.floatX

        self.idim = weight.shape[0]
        self.odim = weight.shape[1]

        # Allocate the memory of parameter.
        self.w    = theano.shared(value=weight)
        self.bias = theano.shared(value=hbias)

			  if vbias:
					  self.vbias = theano.shared(value=vbias)

    # Transfer function.
    def forward(self, x):
        return T.dot(x, self.w) + self.bias
    
    def save(self, fname, acttype="LinearLayer"):
        numpy.savez(fname, type=acttype, w=self.w.get_value(), bias=self.bias.get_value(), vbias=self.bias.get_value())       

# Softmax class
class SoftmaxLayer(LinearLayer):
    def __init__(self, weight, bias):
        super(SoftmaxLayer, self).__init__(weight, bias)
    def forward(self, x):
        return T.nnet.softmax(super(SoftmaxLayer, self).forward(x))
    def save(self, fname):
        super(SoftmaxLayer, self).save(fname, self.__class__.__name__)


# Sigmoid class
class SigmoidLayer(LinearLayer):
    def __init__(self, weight, bias):
        super(SigmoidLayer, self).__init__(weight, bias)
    def forward(self, x):
        return T.nnet.sigmoid(super(SigmoidLayer, self).forward(x))
    def save(self, fname):
        super(SigmoidLayer, self).save(fname, self.__class__.__name__)       
               

        

# Rectified Linear class.
class ReLULayer(LinearLayer):
    def __init__(self, weight, bias):
        super(ReLULayer, self).__init__(weight, bias)
    def forward(self, x):
        return T.max(super(ReLULayer, self).forward(x))
    def save(self, fname):
        super(ReLULayer, self).save(fname, self.__class__.__name__)       

myself = sys.modules[__name__]
percptron_type = [n for n,m in inspect.getmembers(myself, inspect.isclass)]

