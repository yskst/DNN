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
     d.keys()
     rbms = []
     i = 0
     while 'w_'+str(i) in d:
         s = str(i)
         rbms.append(perceptron.generetor(d['type_' +s],
                                          d['w_'    +s]
                                          d['hbias_'+s]
                                          d['vbias_'+s])
         i+=1
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
        dlist = {}
        for i,p in enumerate(self.__percepts__):
            s = str(i)
            dlist['type_' +s] = p.__class__.name__
            dlist['w_'    +s] = p.w.get_value()
            dlist['hbias_'+s] = p.bias.get_value()

        np.savez(fname, **dlist)
