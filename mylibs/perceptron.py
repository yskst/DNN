#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
import inspect
import numpy
import theano
import theano.tensor as T


""" This is the libraries in activatefer functions. """

percptron_type = [] # This array stored perceptron class name.

def load(fname):
    """
    Load perceptron class from npz file.
    
    Parameter:
        fname: The file name or file like object.

    Return:
        The class object of loaded perceptron.
    """
    dat = numpy.load(fname)
    if 'vbias_0' in dat:
        vbias = dat['vbias_0']
    else:
        vbias = None
    return generetor(dat['type_0'], dat['w_0'], dat['hbias_0'], dat['vbias_0'])


def generetor(func, weight, hbias, vbias=None):
    """ 
    Generate perceptron class from parameter.

    Parameter:
        func:   The name of activate function which must be included in perceptron_type.
        weight: The matrix of weight parameter which is numpy.ndarray.
        hbias:  The vector of hidden layer's bias which is numpy.ndarray.
        vbias:  The vector of hidden layer's bias which is numpy.ndarray.
                (optional)

    Return:
        Created class object.

    """
    return eval(str(func)+'(weight,hbias,vbias)')

class LinearLayer(object):
""" The base class of perceptron 
    Parameter:
        idim: the dimension of visible layer.
        odim: the dimension of hidden layer.
        w:    the shared variable of weight parameter.
        bias: the shared variable of the hidden side layer.
        vbias:the shared variable of the visible layer.
"""
    def __init__(self, weight, hbias, vbias=None):
        """ Constructer """
        floatX = theano.config.floatX

        self.idim = weight.shape[0]
        self.odim = weight.shape[1]

        # Allocate the memory of parameter.
        self.w    = theano.shared(value=weight)
        self.bias = theano.shared(value=hbias)

        if vbias is None: self.vbias = None
        else:             self.vbias = theano.shared(value=vbias)

    # Transfer function.
    def forward(self, x):
        """ Get symbol of forwarding from x. 
            Parameter:
                x: symbolic matrix or vector         
        """
        return T.dot(x, self.w) + self.bias

    def inverse(self):
        """ Create perceptron which transform to inverse of this. """
        return generetor(self.__class__.__name__, self.w.get_value().T, self.vbias.get_value(), self.bias.get_value())
    
    def save(self, fname, acttype="LinearLayer"):
        """ Save to parameter of this class. """
        numpy.savez(fname, type_0  = acttype, 
                           w_0     = self.w.get_value(), 
                           hbias_0 = self.bias.get_value(), 
                           vbias_0 = self.bias.get_value()) 

# Softmax class
class SoftmaxLayer(LinearLayer):
    """ The softmax layer  which is override of LinearLayer. """
    def __init__(self, weight, bias, vbias):
        super(SoftmaxLayer, self).__init__(weight, bias, vbias)
    def forward(self, x):
        return T.nnet.softmax(super(SoftmaxLayer, self).forward(x))
    def save(self, fname):
        super(SoftmaxLayer, self).save(fname, self.__class__.__name__)


# Sigmoid class
class SigmoidLayer(LinearLayer):
    """ The sigmoid layer  which is override of LinearLayer. """
    def __init__(self, weight, bias, vbias):
        super(SigmoidLayer, self).__init__(weight, bias, vbias)
    def forward(self, x):
        return T.nnet.sigmoid(super(SigmoidLayer, self).forward(x))
    def save(self, fname):
        super(SigmoidLayer, self).save(fname, self.__class__.__name__)       
               

        

# Rectified Linear class.
    """ The rectified linear layer  which is override of LinearLayer. """
class ReLULayer(LinearLayer):
    def __init__(self, weight, bias,vbias):
        super(ReLULayer, self).__init__(weight, bias, vbias)
    def forward(self, x):
        return T.max(super(ReLULayer, self).forward(x))
    def save(self, fname):
        super(ReLULayer, self).save(fname, self.__class__.__name__)       

myself = sys.modules[__name__]
percptron_type = [n for n,m in inspect.getmembers(myself, inspect.isclass)]

