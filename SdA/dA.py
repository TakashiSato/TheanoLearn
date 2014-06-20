# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T

class dA(object):
    def __init__(self, numpy_rng, theano_rng, input, n_visible, n_hidden,
                 hidden_layer_activation, output_layer_activation,
                 corruption_level, corruption_distribution=None,
                 W=None, bhid=None, bvis=None):
        """ denoising Auto-encoder

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: 重みの初期化に使う乱数発生器

        :type theano_rng: numpy.random.RandomState
        :param theano_rng: 重みの初期化に使う乱数発生器

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_visible: int
        :param n_visible: dimensionality of visible units

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type hidden_layer_activation: theano.Op or function
        :param hidden_activation: Non linearity to be applied in the hidden layer

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
        """

        self.input = input
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.corruption_level = corruption_level

        if not corruption_distribution:
            self.corruption_distribution = theano_rng.binomial(
                                             size=self.input.shape, n=1,
                                             p=1 - self.corruption_level,
                                             dtype=theano.config.floatX)

        if not W:
            W_values = np.asarray(numpy_rng.uniform(
                low=-np.sqrt(6. / (n_hidden + n_visible)),
                high=np.sqrt(6. / (n_hidden + n_visible)),
                size=(n_visible, n_hidden)), dtype=theano.config.floatX)
            if hidden_layer_activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value = W_values, name = 'W', borrow = True)

        if not bvis:
            bvis = theano.shared(value = np.zeros(n_visible, dtype=theano.config.floatX),
                                 borrow = True)
        if not bhid:
            bhid = theano.shared(value = np.zeros(n_hidden, dtype=theano.config.floatX),
                                 name = 'b', borrow = True)

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T

        self.params = [self.W, self.b, self.b_prime]

        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial

                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``

                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.

        """
        self.corrupted_input = self.corruption_distribution * self.input

        """ Computes the values of the hidden layer """
        self.h = self.hidden_layer_activation(T.dot(self.corrupted_input, self.W) \
                                 + self.b)


        """Computes the reconstructed input given the values of the
        hidden layer"""
        self.reconstructed_input = self.output_layer_activation( \
                                        T.dot(self.h, self.W_prime) + self.b_prime)

        self.y_pred = self.output_layer_activation( \
                            T.dot(self.output_layer_activation(T.dot(self.input, self.W) + self.b), \
                            self.W_prime) + self.b_prime)

        # モーメント項のためのパラメータの最終更新値
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        # 損失関数
        # 活性関数がシグモイド関数の時のみcross-entropy
        # 他の活性化関数だとlogに0以下の値が渡されてしまい，nanになる
        if self.output_layer_activation == theano.tensor.nnet.sigmoid:
            self.loss = self.cross_entropy()
            self.each_loss = self.each_cross_entropy()
        else:
            self.loss = self.mse()
            self.each_loss = self.each_mse()

    def cross_entropy(self):
        z = self.reconstructed_input
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        return T.mean(L)

    def each_cross_entropy(self):
        z = self.reconstructed_input
        L = - T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=0)
        return T.mean(L)

    def mse(self):
        z = self.reconstructed_input
        return T.mean((z - self.input) ** 2)

    def each_mse(self):
        z = self.reconstructed_input
        return T.mean((z - self.input) ** 2, axis=0)

    def get_updates(self, learning_rate, momentum):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
        gparams = []
        for param in self.params:
            gparam = T.grad(self.loss, param)
            gparams.append(gparam)

        updates = {}
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]    # 前回の更新量
            upd = momentum * weight_update - learning_rate * gparam
            updates[weight_update] = upd           # モーメント項のために今回の更新量を保存
            updates[param] = param + upd           # パラメータの更新

        return updates
