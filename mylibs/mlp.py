#!/usr/bin/python
# -*- coding: utf-8 -*-

""" Multi Layer Perceptron library """

import theano.tensor as T

class mlp:
    def __init__(rbms):
        __x__        = T.fvector("x")
        __percepts__ = rbms
        __f__        = rbms[0].forward(__x__)
        for rbm in rbms[1:]:
            __f__ = rbm.forward(__f__)

    def forward(x):
        return __f__(x)

    def __getitem__(self, key):
        __percepts__[key]
