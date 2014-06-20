# -*- coding: utf-8 -*-
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import logging
import time
import datetime
import os
import matplotlib.pyplot as plt
import math

import sys
sys.path.append('../Utils')
sys.path.append('../NN')
import Threads
from NNLayer import NNLayer
from dA import dA
from Base import BaseEstimator

logger = logging.getLogger(__name__)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'
plt.rc('figure.subplot',left=0.06,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)

class SdA(object):
    """Stacked denoising auto-encoder class (SdA)

    A stacked denoising autoencoder model is obtained by stacking several
    dAs. The hidden layer of the dA at layer `i` becomes the input of
    the dA at layer `i+1`. The first layer dA gets as input the input of
    the SdA, and the hidden layer of the last dA represents the output.
    Note that after pretraining, the SdA is dealt with as a normal MLP,
    the dAs are only used to initialize the weights.
    """
    def __init__(self, encoder_input, decoder_input, n_in, hidden_layers_sizes,
                 hidden_layer_activation, output_layer_activation,
                 corruption_levels, numpy_rng, theano_rng):
        """ This class is made to support a variable number of layers.

        :type n_in: int
        :param n_in: dimension of the input to the sdA

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        """

        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.encode_layers = []
        self.dA_layers = []
        self.decode_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)	# 中間層数

        # エンコーダを構築
        for i in xrange(self.n_layers):

            # 各層への入力数
            if i == 0:
                input_size = n_in
            else:
                input_size = hidden_layers_sizes[i-1]

            # 各層への入力
            if i == 0:
                layer_input = self.encoder_input
            else:
                layer_input = self.encode_layers[-1].output


            # 各層からの出力
            output_size = hidden_layers_sizes[i]
            
            # 活性化関数(一層目の出力層はデコード層出力に対応する)
            if i == 0:
                dA_output_layer_activation = output_layer_activation
            else:
                dA_output_layer_activation = hidden_layer_activation

            # denoising autoencoder を作成
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=output_size,
                          hidden_layer_activation=hidden_layer_activation,
                          output_layer_activation=dA_output_layer_activation,
                          corruption_level=corruption_levels[i])
            self.dA_layers.append(dA_layer)

            # 作成した層のパラメータをネットワークのパラメータに追加
            self.params.extend(dA_layer.params)

            # dAとウェイトを共有するエンコード用のNNを構築
            encode_layer = NNLayer(rng=numpy_rng,
                               input=layer_input,
                               n_in=input_size,
                               n_out=output_size,
                               activation=hidden_layer_activation,
                               W=dA_layer.W,
                               b=dA_layer.b)

            # 作成した層をネットワークに追加
            self.encode_layers.append(encode_layer)


        # dAとウェイトを共有するデコード用のNNを構築
        for i in xrange(self.n_layers):

            # 各層への入力数
            input_size = hidden_layers_sizes[-1 * i - 1]

            # 各層への入力
            if i == 0:
                layer_input = self.decoder_input
            else:
                layer_input = self.decode_layers[-1].output


            # 各層からの出力
            if i == self.n_layers - 1:
                output_size = n_in
            else:
                output_size = hidden_layers_sizes[-1 * i - 2]
                
            # 活性化関数
            if i == self.n_layers - 1:
                nn_activation = output_layer_activation
            else:
                nn_activation = hidden_layer_activation

            # 層を作成
            decode_layer = NNLayer(rng=numpy_rng,
                               input=layer_input,
                               n_in=input_size,
                               n_out=output_size,
                               activation=nn_activation,
                               W=self.dA_layers[-1 * i -1].W_prime,
                               b=self.dA_layers[-1 * i -1].b_prime)

            # 作成した層をネットワークに追加
            self.decode_layers.append(decode_layer)

        # エンコーダ出力
        self.encode_output = self.encode_layers[-1].output
        # デコーダ出力
        self.decode_output = self.decode_layers[-1].output

    def mse(self):
        return T.mean((self.decode_output - self.encode_output) ** 2)
    

class MetaSdA(BaseEstimator):
    def __init__(self, n_in=10, hidden_layers_sizes=[5,5],
                 n_epochs=1000, batch_size=100, t_error=1e-6,
                 corruption_levels=[0.1,0.1],
                 hidden_layer_activation='tanh', output_layer_activation='linear',
                 learning_rate=0.01, learning_rate_decay=1,
                 final_momentum=0.9, initial_momentum=0.5, momentum_switchover=100,
                 numpy_rng_seed=89677, theano_rng_seed=2 ** 30, 
                 snapshot_every=None, snapshot_path='./tmp'):

        self.n_in = int(n_in)
        self.hidden_layers_sizes = map(int, hidden_layers_sizes)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.t_error = float(t_error)
        self.corruption_levels = map(float, corruption_levels)
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer_activation = output_layer_activation
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.numpy_rng_seed = numpy_rng_seed
        self.theano_rng_seed = theano_rng_seed
        if snapshot_every is not None:
            self.snapshot_every = int(snapshot_every)
        else:
            self.snapshot_every = None
        self.snapshot_path = snapshot_path

        self.ready()

    def ready(self):
        # encoder input (where first dimension is time)
        self.x = T.matrix()
        # decoder input (where first dimension is time)
        self.z = T.matrix()
        # learning rate
        self.lr = T.scalar()

        # setting activation
        if self.hidden_layer_activation == 'tanh':
            hidden_layer_activation = T.tanh
        elif self.hidden_layer_activation == 'sigmoid':
            hidden_layer_activation = T.nnet.sigmoid
        elif self.hidden_layer_activation == 'relu':
            hidden_layer_activation = lambda x: x * (x > 0)
        elif self.hidden_layer_activation == 'cappedrelu':
            hidden_layer_activation = lambda x: T.minimum(x * (x > 0), 6)
        elif self.hidden_layer_activation == 'linear':
            hidden_layer_activation = lambda x: x
        else:
            raise NotImplementedError
        if self.output_layer_activation == 'tanh':
            output_layer_activation = T.tanh
        elif self.output_layer_activation == 'sigmoid':
            output_layer_activation = T.nnet.sigmoid
        elif self.output_layer_activation == 'relu':
            output_layer_activation = lambda x: x * (x > 0)
        elif self.output_layer_activation == 'cappedrelu':
            output_layer_activation = lambda x: T.minimum(x * (x > 0), 6)
        elif self.output_layer_activation == 'linear':
            output_layer_activation = lambda x: x
        else:
            raise NotImplementedError

        # generate numpy rng
        numpy_rng = np.random.RandomState(self.numpy_rng_seed)
        theano_rng = RandomStreams(numpy_rng.randint(self.theano_rng_seed))

        # construct estimator
        self.estimator = SdA(encoder_input=self.x, decoder_input=self.z, n_in=self.n_in,
                             hidden_layers_sizes=self.hidden_layers_sizes,
                             hidden_layer_activation=hidden_layer_activation,
                             output_layer_activation=output_layer_activation,
                             corruption_levels=self.corruption_levels,
                             numpy_rng=numpy_rng, theano_rng=theano_rng)

        # make predict function
        self.encode = theano.function(inputs=[self.x, ],
                                       outputs=self.estimator.encode_output,
                                       mode=mode)
        self.decode = theano.function(inputs=[self.z, ],
                                       outputs=self.estimator.decode_output,
                                       mode=mode)

        # get time stamp
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y%m%d-%H%M%S')
        self.timestamp = date_str
        
        # initialize errorlog
        self.errorlog = []


    # @override
    def shared_dataset(self, data):
        """ Load the dataset into shared variables """
        shared_data = theano.shared(np.asarray(data,
                                     dtype=theano.config.floatX))
        return shared_data

    # override
    def save_errorlog_png(self, fpath='./errorlog', fname=None):
        class_name = self.__class__.__name__
        # Make directory if not exist save dir
        if os.path.isdir(fpath) is False:
            os.makedirs(fpath)

        if fname is None:
            fabspath = os.path.join(fpath, 'Errorlog_' + class_name + '_' + self.timestamp + '.png')
        else:
            fabspath = os.path.join(fpath, 'Errorlog_' + class_name + '_' + fname + '_' + self.timestamp + '.png')


        logger.info("Saving to %s ..." % fabspath)
        plt.close('all')
        
#        plt.plot(self.errorlog[-1][:,0],self.errorlog[-1][:,1:])
#        plt.yscale('log')
        
        col = 2.0 if len(self.errorlog) > 1 else 1.0
        row = int(math.ceil(len(self.errorlog) / col))
        for i in xrange(len(self.errorlog)):
            ax = plt.subplot(row,col,i+1)
            plt.plot(self.errorlog[i][:,0],self.errorlog[i][:,1:])
            plt.yscale('log')
        plt.savefig(fabspath)

    def fit(self, train, test=None,
            validate_every=100, show_norms=False, show_output=False, error_logging=True):
        """ Fit model

        Pass in test to compute test error and report during
        training.

        train : ndarray (T x n_in)

        validation_frequency : int
            in terms of number of epochs

        """
        if test is not None:
            self.interactive = True
            test_set = self.shared_dataset(test)
        else:
            self.interactive = False

        train_set = self.shared_dataset(train)

        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        n_train = train_set.get_value(borrow=True).shape[0]
        n_train_batches = int(np.ceil(1.0 * n_train / self.batch_size))
        if self.interactive:
            n_test = test_set.get_value(borrow=True).shape[0]
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

        compute_train_error = []
        compute_each_train_error = []
        compute_test_error = []
        train_model = []
        for da in self.estimator.dA_layers:
            loss = da.loss
            updates = da.get_updates(l_r, mom)

            f_ctrain = theano.function(inputs=[index, n_ex],
                outputs=loss,
                givens={self.x: train_set[batch_start:batch_stop]},
                mode=mode)
            compute_train_error.append(f_ctrain)
            
            f_cetrain = theano.function(inputs=[index, n_ex],
                outputs=da.each_loss,
                givens={self.x: train_set[batch_start:batch_stop]},
                mode=mode)
            compute_train_error.append(f_ctrain)
            compute_each_train_error.append(f_cetrain)
            
            if self.interactive:
                f_ctest = theano.function(inputs=[index, n_ex],
                        outputs=loss,
                        givens={self.x: test_set[batch_start:batch_stop]},
                        mode=mode)
                compute_test_error.append(f_ctest)
            
            
            # compiling a Theano function `train_model` that returns the
            # cost, but in the same time updates the parameter of the
            # model based on the rules defined in `updates`
            f_train = theano.function(inputs=[index, n_ex, l_r, mom],
                outputs=loss,
                updates=updates,
                givens={self.x: train_set[batch_start:batch_stop]},
                mode=mode)

            train_model.append(f_train)

        ###############
        # TRAIN MODEL #
        ###############
        keyMonitoringThread = Threads.KeyMonitoringThread()
        keyMonitoringThread.start()
        initial_learning_rate = self.learning_rate
        t0 = time.time()
        for n in xrange(self.estimator.n_layers):
            logger.info('... training dA layer[%d]' % n)
            epoch = 0
            this_train_loss = np.inf
            stopFlg = False
            t0_l = time.time()

            self.learning_rate = initial_learning_rate
            self.errorlog.append([])

            while (epoch < self.n_epochs) and (this_train_loss > self.t_error) and (stopFlg is False):
                epoch = epoch + 1
                effective_momentum = self.final_momentum \
                                     if epoch > self.momentum_switchover \
                                     else self.initial_momentum

                for minibatch_idx in xrange(n_train_batches):
                    minibatch_avg_cost = train_model[n](minibatch_idx, n_train,
                                                     self.learning_rate,
                                                     effective_momentum)

                    # iteration number (how many weight updates have we made?)
                    # epoch is 1-based, index is 0 based
                    iter = (epoch - 1) * n_train_batches + minibatch_idx + 1

                    if iter % validation_frequency == 0:
                        # compute loss on training set
                        train_losses = [compute_train_error[n](i, n_train)
                                        for i in xrange(n_train_batches)]
                        train_batch_sizes = [get_batch_size(i, n_train)
                                             for i in xrange(n_train_batches)]

                        this_train_loss = np.average(train_losses,
                                                     weights=train_batch_sizes)

                        # compute each output unit loss on training set
                        if error_logging is True:
                            train_each_losses = np.array([compute_each_train_error[n](i, n_train)
                                            for i in xrange(n_train_batches)])
                            train_batch_sizes_for_each = []
                            for i in xrange(train_each_losses.shape[1]):
                                train_batch_sizes_for_each.append(train_batch_sizes)


                            this_train_each_loss = np.average(train_each_losses.T,
                                                         weights=train_batch_sizes_for_each, axis=1)
                            el = np.r_[np.array([epoch]), this_train_each_loss]
                            self.errorlog[n] = np.vstack((self.errorlog[n], el)) \
                                                       if self.errorlog[n] != [] \
                                                       else np.array([el])
                            # エラーの推移をpngに保存
                            self.save_errorlog_png()
#                            self.save_errorlog_png(fname='dALayer%d' % n)

                        if self.interactive:
                            test_losses = [compute_test_error[n](i, n_test)
                                            for i in xrange(n_test_batches)]

                            test_batch_sizes = [get_batch_size(i, n_test)
                                            for i in xrange(n_test_batches)]

                            this_test_loss = np.average(test_losses,
                                                    weights=test_batch_sizes)

                            logger.info('*** dA layer[%d] *** epoch %i, mb %i/%i, tr loss %f '
                                    'te loss %f lr: %f mom: %f'% \
                            (n, epoch, minibatch_idx + 1, n_train_batches,
                             this_train_loss, this_test_loss,
                             self.learning_rate, effective_momentum))

                        else:
                            logger.info('*** dA layer[%d] *** epoch %i, mb %i/%i, train loss %f'
                                    ' lr: %f mom: %f' % (n, epoch,
                                                     minibatch_idx + 1,
                                                     n_train_batches,
                                                     this_train_loss,
                                                     self.learning_rate,
                                                     effective_momentum))

                        self.optional_output(train_set, show_norms, show_output)

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

            h, m = divmod(time.time() - t0_l, 3600)
            m, s = divmod(m, 60)
            print "*** dA layer[%d] *** Elapsed time: %d hour %d min %f sec" % (n, int(h), int(m), s)

        h, m = divmod(time.time() - t0, 3600)
        m, s = divmod(m, 60)
        print "Elapsed time: %d hour %d min %f sec" % (int(h), int(m), s)

        # コマンド入力スレッドの停止
        # ('q'入力による学習の途中終了ではなく、終了条件を満たして
        # 学習が正常に終了した場合、スレッドを明示的に終了する必要がある)
        keyMonitoringThread.Stop()
        print 'Press any key...'

def test_real():
    """ Test SdA with real-valued outputs. """
    hidden_layers_sizes = [8, 5]
    n_in =10 
    n_seq = 100
    corruption_levels = [.1,.2]

    z = np.random.rand(n_seq, n_in)
    z = np.sort(z,axis=1)
    seq = np.sin(z)

    model = MetaSdA(n_in=n_in, hidden_layers_sizes=hidden_layers_sizes,
                    corruption_levels=corruption_levels,
                    learning_rate=0.001, learning_rate_decay=0.999999,
                    n_epochs=100000, batch_size=100, t_error=1e-6,
                    hidden_layer_activation='tanh', output_layer_activation='linear',
                    numpy_rng_seed=89677, theano_rng_seed=2**30,
                    final_momentum=0.9, initial_momentum=0.5, momentum_switchover=100,
                    snapshot_every=None, snapshot_path='./models/tmp')

    model.fit(seq, validate_every=5000, show_norms=False, show_output=False, error_logging=True)
    
    plt.close('all')
    fig = plt.figure()
    true_sequence = plt.plot(seq[:,0])

    dec = model.encode(seq)
    guess = model.decode(dec)
    guessed_targets = plt.plot(guess[:,0], linestyle='--')
    for i, x in enumerate(guessed_targets):
        x.set_color(true_sequence[i].get_color())
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_real()
