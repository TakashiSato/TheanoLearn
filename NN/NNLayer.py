# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

class NNLayer(object):
    def __init__(self, rng, input, n_in, n_out, activation, W=None, b=None):
        """ ニューラルネットワークの層

        :type rng: numpy.random.RandomState
        :param rng: 重みの初期化に使う乱数発生器

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden layer
        """


        self.input = input

        if not W:
            W_values = np.asarray(rng.uniform(
                low=-np.sqrt(6. / (n_in + n_out)),
                high=np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if not b:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value = b_values, name = 'b', borrow = True)

        self.W = W
        self.b = b

        linOutput = T.dot(input, self.W) + self.b
        self.output = (linOutput if activation is None else activation(linOutput))

        self.params = [self.W, self.b]
