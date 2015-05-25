#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Multi Layer Perceptron library """
import numpy as np
import theano.tensor as T
import perceptron

def load(fname):
     d = np.load(fname)
     rbms = []
     for i in range(0, len(d), 3):
         rbms.append(perceptron.generetor(d[i], d[i+1], d[i+2]))
     return mlp(rbms)

class mlp:
    def __init__(rbms):
        __x__        = T.fvector("x")
        __percepts__ = rbms
        f            = rbms[0].forward(__x__)
        for rbm in rbms[1:]:
            f = rbm.forward(f)

    def forward(self, x):
        return self.f(x)

    def __len__(self):
        return len(self.__percepts__)

    def __getitem__(self, key):
        return self.__percepts__[key]

    def save(self, fname):
        dlist = []
        for p in __percepts__:
            dlist.extend([type(p), p.w.get_values(), p.bias.get_values()])

        np.savez(fname, dlist)
