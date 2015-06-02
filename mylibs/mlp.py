#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Multi Layer Perceptron library """

import numpy as np
import theano.tensor as T
import perceptron

def __get_npitem__(array, index):
    return array['arr_' + str(index)]

def load(fname):
     d = np.load(fname)
     print d.keys()
     rbms = []
     for i in range(0, len(d), 3):
         rbms.append(perceptron.generetor(__get_npitem__(d,i), 
                                          __get_npitem__(d,i+1), 
                                          __get_npitem__(d,i+2)))
     return mlp(rbms)

class mlp:
    def __init__(self, rbms):
        x        = T.fvector("x")
        self.__percepts__ = rbms
        self.f            = rbms[0].forward(x)
        for rbm in rbms[1:]:
            self.f = rbm.forward(self.f)

    def __len__(self):
        return len(self.__percepts__)

    def __getitem__(self, key):
        return self.__percepts__[key]
    
    def __iter__(self):
        for l in self.__percepts__:
            yield l

    def forward(self, x):
        return self.f(x)

    def save(self, fname):
        dlist = []
        for p in self.__percepts__:
            dlist.extend([p.__class__.__name__, p.w.get_value(), p.bias.get_value()])

        np.savez(fname, *dlist)
