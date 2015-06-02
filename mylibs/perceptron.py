#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import inspect
import numpy
import theano
import theano.tensor as T


""" This is the libraries in activatefer functions. """


def load(fname):
    dat = numpy.load(fname)
    if 'arr_3' in dat:
        arr['arr_3'] = None
    return generetor(dat['arr_0'], dat['arr_1'], dat['arr_2'], dat['arr_3'])

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
        if vbias is None:
            self.vbias = theano.shared(
                    value=numpy.zeros(self.odim, 
                                      dtype=theano.config.floatX))
        else:
            self.vbias = theano.shared(value=vbias)

    # Transfer function.
    def forward(self, x):
        return T.dot(x, self.w) + self.bias

    def inverse(self):
        generetor(self.__class__.__name__, self.w.T, self.vbias, self.bias)
    
    def save(self, fname, acttype="LinearLayer"):
        numpy.savez(fname, [acttype, self.w.get_value(), self.bias.get_value(), self.bias.get_value()])       

# Softmax class
class SoftmaxLayer(LinearLayer):
    def __init__(self, weight, bias, vbias):
        super(SoftmaxLayer, self).__init__(weight, bias, vbias)
    def forward(self, x):
        return T.nnet.softmax(super(SoftmaxLayer, self).forward(x))
    def save(self, fname):
        super(SoftmaxLayer, self).save(fname, self.__class__.__name__)


# Sigmoid class
class SigmoidLayer(LinearLayer):
    def __init__(self, weight, bias, vbias):
        super(SigmoidLayer, self).__init__(weight, bias, vbias)
    def forward(self, x):
        return T.nnet.sigmoid(super(SigmoidLayer, self).forward(x))
    def save(self, fname):
        super(SigmoidLayer, self).save(fname, self.__class__.__name__)       
               

        

# Rectified Linear class.
class ReLULayer(LinearLayer):
    def __init__(self, weight, bias,vbias):
        super(ReLULayer, self).__init__(weight, bias, vbias)
    def forward(self, x):
        return T.max(super(ReLULayer, self).forward(x))
    def save(self, fname):
        super(ReLULayer, self).save(fname, self.__class__.__name__)       

myself = sys.modules[__name__]
percptron_type = [n for n,m in inspect.getmembers(myself, inspect.isclass)]

