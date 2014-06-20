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
from Base import BaseEstimator

logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class RNN(object):
    """    Recurrent neural network class

    Supported output types:
    real : linear output units, use mean-squared error
    binary : binary output units, use cross-entropy error
    softmax : single softmax out, use cross-entropy error

    """
    def __init__(self, input, n_in, n_hidden, n_out, truncated_num, activation,
                 numpy_rng):

        self.input = input
        self.activation = activation

        self.batch_size = T.iscalar()

        # theta is a vector of all trainable parameters
        # it represents the value of W, W_in, W_out, h0, bh, by
        theta_shape = n_hidden ** 2 + n_in * n_hidden + n_hidden * n_out + \
                      n_hidden + n_hidden + n_out
        self.theta = theano.shared(value=np.zeros(theta_shape,
                                                  dtype=theano.config.floatX))

        # Parameters are reshaped views of theta
        param_idx = 0  # pointer to somewhere along parameter vector

        # recurrent weights as a shared variable
        self.W = self.theta[param_idx:(param_idx + n_hidden ** 2)].reshape(
            (n_hidden, n_hidden))
        self.W.name = 'W'
        W_init = np.asarray(numpy_rng.uniform(
                              low=-np.sqrt(6. / (n_hidden + n_hidden)),
                              high=np.sqrt(6. / (n_hidden + n_hidden)),
                              size=(n_hidden, n_hidden)), dtype=theano.config.floatX)
        param_idx += n_hidden ** 2

        # input to hidden layer weights
        self.W_in = self.theta[param_idx:(param_idx + n_in * \
                                          n_hidden)].reshape((n_in, n_hidden))
        self.W_in.name = 'W_in'
        W_in_init = np.asarray(numpy_rng.uniform(
                               low=-np.sqrt(6. / (n_in + n_hidden)),
                               high=np.sqrt(6. / (n_in + n_hidden)),
                               size=(n_in, n_hidden)), dtype=theano.config.floatX)
        param_idx += n_in * n_hidden

        # hidden to output layer weights
        self.W_out = self.theta[param_idx:(param_idx + n_hidden * \
                                           n_out)].reshape((n_hidden, n_out))
        self.W_out.name = 'W_out'

        W_out_init = np.asarray(numpy_rng.uniform(
                               low=-np.sqrt(6. / (n_hidden + n_out)),
                               high=np.sqrt(6. / (n_hidden + n_out)),
                               size=(n_hidden, n_out)), dtype=theano.config.floatX)
        param_idx += n_hidden * n_out

        self.h0 = self.theta[param_idx:(param_idx + n_hidden)]
        self.h0.name = 'h0'
        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        param_idx += n_hidden

        self.bh = self.theta[param_idx:(param_idx + n_hidden)]
        self.bh.name = 'bh'
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        param_idx += n_hidden

        self.by = self.theta[param_idx:(param_idx + n_out)]
        self.by.name = 'by'
        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        param_idx += n_out

        assert(param_idx == theta_shape)

        # for convenience
        self.params = [self.W, self.W_in, self.W_out, self.h0, self.bh,
                        self.by]

        # shortcut to norms (for monitoring)
        self.l2_norms = {}
        for param in self.params:
            self.l2_norms[param] = T.sqrt(T.sum(param ** 2))

        # initialize parameters
        # DEBUG_MODE gives division by zero error when we leave parameters
        # as zeros
        self.theta.set_value(np.concatenate([x.ravel() for x in
            (W_init, W_in_init, W_out_init, h0_init, bh_init, by_init)]))

        self.theta_update = theano.shared(
            value=np.zeros(theta_shape, dtype=theano.config.floatX))

        # recurrent function (using tanh activation function) and arbitrary output
        # activation function
        def step(x_t, h_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) + \
                                  T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        # Note the implementation of weight-sharing h0 across variable-size
        # batches using T.ones multiplying h0
        # Alternatively, T.alloc approach is more robust
        [self.h, self.y_pred], _ = theano.scan(step,
                    sequences=self.input,
                    outputs_info=[T.alloc(self.h0, self.input.shape[1],
                                          n_hidden), None],
                    truncate_gradient=truncated_num)
                    # outputs_info=[T.ones(shape=(self.input.shape[1],
                    # self.h0.shape[0])) * self.h0, None])

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        self.L1 += abs(self.W_out.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

        self.loss = lambda y: self.mse(y)
        self.each_loss = lambda y: self.each_mse(y)

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def each_mse(self, y):
        # each error between output and target
        return T.mean(T.mean((self.y_pred - y) ** 2, axis=0), axis=0)

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                ('y', y.type, 'y_out', self.y_out.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_out, y))
        else:
            raise NotImplementedError()


class MetaRNN(BaseEstimator):
    def __init__(self, n_in=5, n_hidden=50, n_out=5,
                 truncated_num=-1, n_epochs=100, batch_size=100, t_error=1e-6,
                 L1_reg=0.00, L2_reg=0.00, activation='tanh',
                 learning_rate=0.01, learning_rate_decay=1,
                 final_momentum=0.9, initial_momentum=0.5, momentum_switchover=100,
                 numpy_rng_seed=89677,
                 snapshot_every=None, snapshot_path='./models/tmp'):

        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.truncated_num = int(truncated_num)
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
        self.x = T.tensor3(name='x')
        # target (where first dimension is time)
        self.y = T.tensor3(name='y', dtype=theano.config.floatX)

        # learning rate
        self.lr = T.scalar()

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

        self.estimator = RNN(input=self.x, n_in=self.n_in,
                             n_hidden=self.n_hidden, n_out=self.n_out,
                             truncated_num=self.truncated_num,
                             activation=activation, numpy_rng=numpy_rng)

        self.predict = theano.function(inputs=[self.x, ],
                                       outputs=self.estimator.y_pred,
                                       mode=mode)
        
        # get time stamp
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y%m%d-%H%M%S')
        self.timestamp = date_str
        
        # initialize errorlog
        self.errorlog = []

    # @override
    def __getstate__(self):
        """ Return state sequence."""
        params = self.get_params()  # parameters set in constructor
        theta = self.estimator.theta.get_value()
        state = (params, theta)
        return state

    # @override
    def _set_weights(self, theta):
        """ Set fittable parameters from weights sequence.
        """
        self.estimator.theta.set_value(theta)

    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validate_every=100, optimizer='sgd', compute_zero_one=False,
            show_norms=True, show_output=True, error_logging=True):
        """ Fit model

        Pass in X_test, Y_test to compute test error and report during
        training.

        X_train : ndarray (T x n_in)
        Y_train : ndarray (T x n_out)

        validation_frequency : int
            in terms of number of epochs

        optimizer : string
            Optimizer type.
            Possible values:
                'sgd'  : batch stochastic gradient descent
                'cg'   : nonlinear conjugate gradient algorithm
                         (scipy.optimize.fmin_cg)
                'bfgs' : quasi-Newton method of Broyden, Fletcher, Goldfarb,
                         and Shanno (scipy.optimize.fmin_bfgs)
                'l_bfgs_b' : Limited-memory BFGS (scipy.optimize.fmin_l_bfgs_b)

        compute_zero_one : bool
            in the case of binary output, compute zero-one error in addition to
            cross-entropy error
        show_norms : bool
            Show L2 norms of individual parameter groups while training.
        show_output : bool
            Show the model output on first training case while training.
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
        n_train = train_set_x.get_value(borrow=True).shape[1]
        n_train_batches = int(np.ceil(1.0 * n_train / self.batch_size))
        if self.interactive:
            n_test = test_set_x.get_value(borrow=True).shape[1]
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
            givens={self.x: train_set_x[:, batch_start:batch_stop],
                    self.y: train_set_y[:, batch_start:batch_stop]},
            mode=mode)

        compute_train_each_error = theano.function(inputs=[index, n_ex],
            outputs=self.estimator.each_loss(self.y),
            givens={self.x: train_set_x[:, batch_start:batch_stop],
                    self.y: train_set_y[:, batch_start:batch_stop]},
            mode=mode)

        if self.interactive:
            compute_test_error = theano.function(inputs=[index, n_ex],
                outputs=self.estimator.loss(self.y),
                givens={self.x: test_set_x[:, batch_start:batch_stop],
                        self.y: test_set_y[:, batch_start:batch_stop]},
                mode=mode)

        self.get_norms = {}
        for param in self.estimator.params:
            self.get_norms[param] = theano.function(inputs=[],
                    outputs=self.estimator.l2_norms[param], mode=mode)

        # compute the gradient of cost with respect to theta using BPTT
        gtheta = T.grad(cost, self.estimator.theta)

        if optimizer == 'sgd':

            updates = {}
            theta = self.estimator.theta
            theta_update = self.estimator.theta_update
            # careful here, update to the shared variable
            # cannot depend on an updated other shared variable
            # since updates happen in parallel
            # so we need to be explicit
            upd = mom * theta_update - l_r * gtheta
            updates[theta_update] = upd
            updates[theta] = theta + upd

            # compiling a Theano function `train_model` that returns the
            # cost, but in the same time updates the parameter of the
            # model based on the rules defined in `updates`
            train_model = theano.function(inputs=[index, n_ex, l_r, mom],
                outputs=cost,
                updates=updates,
                givens={self.x: train_set_x[:, batch_start:batch_stop],
                        self.y: train_set_y[:, batch_start:batch_stop]},
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
                                                       if self.errorlog != [] \
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

                        self.optional_output(train_set_x, show_norms,
                                             show_output)

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

        elif optimizer == 'cg' or optimizer == 'bfgs' or optimizer == 'l_bfgs_b':
            # compile a theano function that returns the cost of a minibatch
            batch_cost = theano.function(inputs=[index, n_ex],
                outputs=cost,
                givens={self.x: train_set_x[:, batch_start:batch_stop],
                        self.y: train_set_y[:, batch_start:batch_stop]},
                mode=mode, name="batch_cost")

            # compile a theano function that returns the gradient of the
            # minibatch with respect to theta
            batch_grad = theano.function(inputs=[index, n_ex],
                outputs=T.grad(cost, self.estimator.theta),
                givens={self.x: train_set_x[:, batch_start:batch_stop],
                        self.y: train_set_y[:, batch_start:batch_stop]},
                mode=mode, name="batch_grad")

            # creates a function that computes the average cost on the training
            # set
            def train_fn(theta_value):
                theta_value=np.array(theta_value, dtype=theano.config.floatX)
                self.estimator.theta.set_value(theta_value, borrow=True)
                train_losses = [batch_cost(i, n_train)
                                for i in xrange(n_train_batches)]
                train_batch_sizes = [get_batch_size(i, n_train)
                                     for i in xrange(n_train_batches)]
                return np.average(train_losses, weights=train_batch_sizes)

            # creates a function that computes the average gradient of cost
            # with respect to theta
            def train_fn_grad(theta_value):
                theta_value=np.array(theta_value, dtype=theano.config.floatX)
                self.estimator.theta.set_value(theta_value, borrow=True)

                train_grads = [batch_grad(i, n_train)
                                for i in xrange(n_train_batches)]
                train_batch_sizes = [get_batch_size(i, n_train)
                                     for i in xrange(n_train_batches)]

                return np.average(train_grads, weights=train_batch_sizes,
                                  axis=0)

            # validation function, prints useful output after each iteration
            def callback(theta_value):
                self.epoch += 1
                if (self.epoch) % validate_every == 0:
                    theta_value=np.array(theta_value, dtype=theano.config.floatX)
                    self.estimator.theta.set_value(theta_value, borrow=True)
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
                        el = np.r_[np.array([self.epoch]), this_train_each_loss]
                        self.errorlog = np.vstack((self.errorlog, el)) \
                                                   if self.errorlog is not None \
                                                   else np.array([el])
                        # エラーの推移をpngに保存
                        self.save_errorlog_png(fname=optimizer)

                    if self.interactive:
                        test_losses = [compute_test_error(i, n_test)
                                        for i in xrange(n_test_batches)]

                        test_batch_sizes = [get_batch_size(i, n_test)
                                        for i in xrange(n_test_batches)]

                        this_test_loss = np.average(test_losses,
                                                    weights=test_batch_sizes)

                        logger.info('epoch %i, tr loss %f, te loss %f' % \
                                    (self.epoch, this_train_loss,
                                     this_test_loss, self.learning_rate))

                    else:
                        logger.info('epoch %i, train loss %f ' % \
                                    (self.epoch, this_train_loss))

                    self.optional_output(train_set_x, show_norms, show_output)

            ###############
            # TRAIN MODEL #
            ###############
            logger.info('... training')
            # using scipy conjugate gradient optimizer
            import scipy.optimize
            if optimizer == 'cg':
                of = scipy.optimize.fmin_cg
            elif optimizer == 'bfgs':
                of = scipy.optimize.fmin_bfgs
            elif optimizer == 'l_bfgs_b':
                of = scipy.optimize.fmin_l_bfgs_b
            logger.info("Optimizing using %s..." % of.__name__)
            start_time = time.clock()

            # keep track of epochs externally
            # these get updated through callback
            self.epoch = 0

            # interface to l_bfgs_b is different than that of cg, bfgs
            # however, this will be changed in scipy 0.11
            # unified under scipy.optimize.minimize
            if optimizer == 'cg' or optimizer == 'bfgs':
                best_theta = of(
                    f=train_fn,
                    x0=self.estimator.theta.get_value(),
                    #x0=np.zeros(self.estimator.theta.get_value().shape,
                    #             dtype=theano.config.floatX),
                    fprime=train_fn_grad,
                    callback=callback,
                    disp=1,
                    retall=1,
                    maxiter=self.n_epochs)
            elif optimizer == 'l_bfgs_b':
                best_theta, f_best_theta, info = of(
                    func=train_fn,
                    x0=self.estimator.theta.get_value(),
                    fprime=train_fn_grad,
                    iprint=validate_every,
                    maxfun=self.n_epochs)  # max number of feval

            end_time = time.clock()

            h, m = divmod(end_time - start_time, 3600)
            m, s = divmod(m, 60)
            print "Optimization time: %d hour %d min %f sec" % (int(h), int(m), s)

        else:
            raise NotImplementedError
        

def test_real():
    """ Test RNN with real-valued outputs. """
    n_hidden = 10
    n_in = 5
    n_out = 3
    n_steps = 10
    n_seq = 10  # per batch
    n_batches = 10

    np.random.seed(0)
    # simple lag test
    seq = np.random.randn(n_steps, n_seq * n_batches, n_in)
    targets = np.zeros((n_steps, n_seq * n_batches, n_out))

    targets[1:, :, 0] = seq[:-1, :, 3]  # delayed 1
    targets[1:, :, 1] = seq[:-1, :, 2]  # delayed 1
    targets[2:, :, 2] = seq[:-2, :, 0]  # delayed 2

    targets += 0.01 * np.random.standard_normal(targets.shape)

    seq = np.float32(seq)
    targets = np.float32(targets)
    model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                    truncated_num=5,
                    learning_rate=0.01, learning_rate_decay=0.999999,
                    n_epochs=1000, batch_size=n_seq, activation='tanh',
                    L2_reg=1e-3, snapshot_every=10000, numpy_rng_seed=89677)
#    model.fit(seq, targets, validate_every=100, optimizer='bfgs',
#            show_norms = False, show_output = False)
    model.fit(seq, targets, validate_every=100, optimizer='sgd',
            show_norms = False, show_output = False)

    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(211)
    plt.plot(seq[:, 0, :])
    ax1.set_title('input')
    ax2 = plt.subplot(212)
    true_targets = plt.plot(targets[:, 0, :])

    guess = model.predict(seq[:, 0, :][:, np.newaxis, :])

    guessed_targets = plt.plot(guess.squeeze(), linestyle='--')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')
    plt.show()
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_real()
