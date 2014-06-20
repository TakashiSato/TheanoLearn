# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
import logging
import time
import datetime
import matplotlib.pyplot as plt

import sys
sys.path.append('../Utils')
import Threads
from NNLayer import NNLayer
from Base import BaseEstimator

logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class NN(object):
    def __init__(self, input, n_in, hidden_layers_sizes, n_out, activation,
                 numpy_rng):
        """ MultilayerNeuralNetwork
        """

        self.input = input
        self.layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes) + 1		# 中間層数 + 出力層数(1)

        if not numpy_rng:
            numpy_rng = np.random.RandomState(0)

        # ネットワークを構築
        for i in xrange(self.n_layers):

            # 各層への入力数
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layers_sizes[i-1]

            # 各層への入力
            if i == 0:
                layer_input = self.input
            else:
                layer_input = self.layers[-1].output


            # 各層からの出力
            if i == self.n_layers - 1:
                output_size = n_out
            else:
                output_size = hidden_layers_sizes[i]
                
            # 活性化関数(出力層は線形関数)
            if i == self.n_layers - 1:
                activation = lambda x: x

            # 層を作成
            layer = NNLayer(rng=numpy_rng,
                            input=layer_input,
                            n_in=input_size,
                            n_out=output_size,
                            activation=activation)

            # 作成した層をネットワークに追加
            self.layers.append(layer)

            # 作成した層のパラメータをネットワークのパラメータに追加
            self.params.extend(layer.params)

        # shortcut to norms (for monitoring)
        self.l2_norms = {}
        for param in self.params:
            self.l2_norms[param] = T.sqrt(T.sum(param ** 2))

        # モーメント項のためのパラメータの最終更新値
        self.updates = {}
        for param in self.params:
            init = np.zeros(param.get_value(borrow=True).shape,
                            dtype=theano.config.floatX)
            self.updates[param] = theano.shared(init)

        # 出力層の出力 
        self.y_pred = self.layers[-1].output

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        for i in xrange(self.n_layers):
            self.L1 += abs(self.layers[i].W.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        for i in xrange(self.n_layers):
            self.L2_sqr += (self.layers[i].W ** 2).sum()

        self.loss = lambda y:self.mse(y)
        self.each_loss = lambda y:self.each_mse(y)

    def mse(self, y):
        return T.mean((self.y_pred - y) ** 2)

    def each_mse(self, y):
        # each error between output and target
        return T.mean((self.y_pred - y) ** 2, axis=0)

class MetaNN(BaseEstimator):
    def __init__(self, n_in=5, hidden_layers_sizes=[10,10], n_out=5,
                 n_epochs=100, batch_size=100, t_error=1e-6,
                 L1_reg=0.00, L2_reg=0.00, activation='tanh',
                 learning_rate=0.01, learning_rate_decay=1,
                 final_momentum=0.9, initial_momentum=0.5, momentum_switchover=100,
                 numpy_rng_seed=89677,
                 snapshot_every=None, snapshot_path='./tmp'):

        self.n_in = int(n_in)
        self.hidden_layers_sizes = map(int, hidden_layers_sizes)
        self.n_out = int(n_out)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.t_error = float(t_error)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.numpy_rng_seed = int(numpy_rng_seed)
        if snapshot_every is not None:
            self.snapshot_every = int(snapshot_every)
        else:
            self.snapshot_every = None
        self.snapshot_path = snapshot_path

        self.ready()
        
    def ready(self):
        # input (where first dimension is time)
        self.x = T.matrix()
        # target (where first dimension is time)
        self.y = T.matrix(name='y', dtype=theano.config.floatX)
        # learning rate
        self.lr = T.scalar()
        
        # setting activation
        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        # generate numpy rng
        numpy_rng = np.random.RandomState(self.numpy_rng_seed)

        # construct estimator
        self.estimator = NN(input=self.x, n_in=self.n_in,
                            hidden_layers_sizes=self.hidden_layers_sizes, n_out=self.n_out,
                            activation=activation,
                            numpy_rng=numpy_rng)

        # make predict function
        self.predict = theano.function(inputs=[self.x, ],
                                       outputs=self.estimator.y_pred,
                                       mode=mode)

        # get time stamp
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y%m%d-%H%M%S')
        self.timestamp = date_str
        
        # initialize errorlog
        self.errorlog = []
    
    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validate_every=100, show_norms=False, show_output=False, error_logging=True):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (T x n_in)
        Y_train : ndarray (T x n_out)

        validation_frequency : int
            in terms of number of epochs

        """
        if X_test is not None:
            assert(Y_test is not None)
            self.interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            self.interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))

        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        n_train = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches = int(np.ceil(1.0 * n_train / self.batch_size))
        if self.interactive:
            n_test = test_set_x.get_value(borrow=True).shape[0]
            n_test_batches = int(np.ceil(1.0 * n_test / self.batch_size))

        #validate_every is specified in terms of epochs
        validation_frequency = validate_every * n_train_batches

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        index = T.lscalar('index')    # index to a [mini]batch
        n_ex = T.lscalar('n_ex')      # total number of examples
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

        cost = self.estimator.loss(self.y) \
            + self.L1_reg * self.estimator.L1 \
            + self.L2_reg * self.estimator.L2_sqr

        # Proper implementation of variable-batch size evaluation
        # Note that classifier.errors() returns the mean error
        # But the last batch may be a smaller size
        # So we keep around the effective_batch_size (whose last element may
        # be smaller than the rest)
        # And weight the reported error by the batch_size when we average
        # Also, by keeping batch_start and batch_stop as symbolic variables,
        # we make the theano function easier to read
        batch_start = index * self.batch_size
        batch_stop = T.minimum(n_ex, (index + 1) * self.batch_size)
        effective_batch_size = batch_stop - batch_start

        get_batch_size = theano.function(inputs=[index, n_ex],
                                          outputs=effective_batch_size)

        compute_train_error = theano.function(inputs=[index, n_ex],
            outputs=self.estimator.loss(self.y),
            givens={self.x: train_set_x[batch_start:batch_stop],
                    self.y: train_set_y[batch_start:batch_stop]},
            mode=mode)

        compute_train_each_error = theano.function(inputs=[index, n_ex],
            outputs=self.estimator.each_loss(self.y),
            givens={self.x: train_set_x[batch_start:batch_stop],
                    self.y: train_set_y[batch_start:batch_stop]},
            mode=mode)

        if self.interactive:
            compute_test_error = theano.function(inputs=[index, n_ex],
                outputs=self.estimator.loss(self.y),
                givens={self.x: test_set_x[batch_start:batch_stop],
                        self.y: test_set_y[batch_start:batch_stop]},
                mode=mode)

        self.get_norms = {}
        for param in self.estimator.params:
            self.get_norms[param] = theano.function(inputs=[],
                    outputs=self.estimator.l2_norms[param], mode=mode)

        # compute the gradient of cost with respect to theta
        gparams = []
        for param in self.estimator.params:
            gparam = T.grad(cost, param)
            gparams.append(gparam)

        updates = {}
        for param, gparam in zip(self.estimator.params, gparams):
            weight_update = self.estimator.updates[param]         # 前回の更新量
            upd = mom * weight_update - l_r * gparam
            updates[weight_update] = upd                    # モーメント項のために今回の更新量を保存
            updates[param] = param + upd                    # パラメータの更新

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs=[index, n_ex, l_r, mom],
            outputs=cost,
            updates=updates,
            givens={self.x: train_set_x[batch_start:batch_stop],
                    self.y: train_set_y[batch_start:batch_stop]},
            mode=mode)

        ###############
        # TRAIN MODEL #
        ###############
        logger.info('... training')
        epoch = 0
        this_train_loss = np.inf
        stopFlg = False
        keyMonitoringThread = Threads.KeyMonitoringThread()
        keyMonitoringThread.start()
        t0 = time.time()

        while (epoch < self.n_epochs) and (this_train_loss > self.t_error) and (stopFlg is False):
            epoch = epoch + 1
            effective_momentum = self.final_momentum \
                                 if epoch > self.momentum_switchover \
                                 else self.initial_momentum

            for minibatch_idx in xrange(n_train_batches):
                minibatch_avg_cost = train_model(minibatch_idx, n_train,
                                                 self.learning_rate,
                                                 effective_momentum)

                # iteration number (how many weight updates have we made?)
                # epoch is 1-based, index is 0 based
                iter = (epoch - 1) * n_train_batches + minibatch_idx + 1

                if iter % validation_frequency == 0:
                    # compute loss on training set
                    train_losses = [compute_train_error(i, n_train)
                                    for i in xrange(n_train_batches)]
                    train_batch_sizes = [get_batch_size(i, n_train)
                                         for i in xrange(n_train_batches)]

                    this_train_loss = np.average(train_losses,
                                                 weights=train_batch_sizes)

                    # compute each output unit loss on training set
                    if error_logging is True:
                        train_each_losses = np.array([compute_train_each_error(i, n_train)
                                        for i in xrange(n_train_batches)])
                        train_batch_sizes_for_each = []
                        for i in xrange(self.n_out):
                            train_batch_sizes_for_each.append(train_batch_sizes)


                        this_train_each_loss = np.average(train_each_losses.T,
                                                     weights=train_batch_sizes_for_each, axis=1)
                        el = np.r_[np.array([epoch]), this_train_each_loss]
                        self.errorlog = np.vstack((self.errorlog, el)) \
                                                   if self.errorlog is not None \
                                                   else np.array([el])
                        # エラーの推移をpngに保存
                        self.save_errorlog_png()

                    if self.interactive:
                        test_losses = [compute_test_error(i, n_test)
                                        for i in xrange(n_test_batches)]

                        test_batch_sizes = [get_batch_size(i, n_test)
                                        for i in xrange(n_test_batches)]

                        this_test_loss = np.average(test_losses,
                                                weights=test_batch_sizes)

                        logger.info('epoch %i, mb %i/%i, tr loss %f '
                                'te loss %f lr: %f mom: %f'% \
                        (epoch, minibatch_idx + 1, n_train_batches,
                         this_train_loss, this_test_loss,
                         self.learning_rate, effective_momentum))

                    else:
                        logger.info('epoch %i, mb %i/%i, train loss %f'
                                ' lr: %f mom: %f' % (epoch,
                                                 minibatch_idx + 1,
                                                 n_train_batches,
                                                 this_train_loss,
                                                 self.learning_rate,
                                                 effective_momentum))

                    self.optional_output(train_set_x, show_norms, show_output)

            self.learning_rate *= self.learning_rate_decay

            # 学習途中のパラメータをスナップショットとして保存
            if self.snapshot_every is not None:
                if (epoch + 1) % self.snapshot_every == 0:
                    date_obj = datetime.datetime.now()
                    date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
                    class_name = self.__class__.__name__
                    fname = '%s.%s-snapshot-%d' % (class_name,
                                                       date_str, epoch + 1)
                    self.save(fpath=self.snapshot_path, fname=fname)
            
            # 学習中のコマンド入力を別スレッドで受け取る
            var = keyMonitoringThread.GetInput()
            # 'q' を受け取った場合、学習を途中で切り上げる
            if var == 'q':
                stopFlg = True
                keyMonitoringThread.Stop()

        h, m = divmod(time.time() - t0, 3600)
        m, s = divmod(m, 60)
        print "Elapsed time: %d hour %d min %f sec" % (int(h), int(m), s)

        # コマンド入力スレッドの停止
        # ('q'入力による学習の途中終了ではなく、終了条件を満たして
        # 学習が正常に終了した場合、スレッドを明示的に終了する必要がある)
        keyMonitoringThread.Stop()
        print 'Press any key...'

def test_real():
    """ Test NN with real-valued outputs. """
    hidden_layers_sizes = [10,10]
    n_in = 5
    n_out = 3
    n_seq = 10

    seq = np.random.randn(n_seq, n_in)
    targets = np.zeros((n_seq, n_out))
    targets += 0.1 * np.random.standard_normal(targets.shape)

    model = MetaNN(n_in=n_in, hidden_layers_sizes=hidden_layers_sizes, n_out=n_out,
                   n_epochs=100000, batch_size=10, t_error=1e-6,
                   L1_reg=0.00, L2_reg=0.00, activation='tanh',
                   learning_rate=0.001, learning_rate_decay=0.999999,
                   final_momentum=0.9, initial_momentum=0.5, momentum_switchover=100,
                   numpy_rng_seed=89677,
                   snapshot_every=None, snapshot_path='./models/tmp')

    model.fit(seq, targets, validate_every=1000, show_norms=True, error_logging=True)
   
    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq)
    ax1.set_title('input')

    ax2 = plt.subplot(212)
    true_targets = plt.plot(targets)

    guess = model.predict(seq)
    guessed_targets = plt.plot(guess, linestyle='--')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')
    plt.show()

# main
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_real()

