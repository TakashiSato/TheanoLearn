# -*- coding: utf-8 -*-

import time
import datetime
import math

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import logging
import cPickle as pickle

logger = logging.getLogger(__name__)
plt.rc('figure.subplot',left=0.1,right=0.94,hspace=0,wspace=0,bottom=0.1,top=0.94)

def PlotErrorlog(errorlog, max_epoch=None, save=False):
    fig = plt.figure()

    epoch = errorlog[:,0]
    errors = errorlog[:,1:]
    
    def get_index(mat, value):
        for i,e in enumerate(mat):
            if e == value:
                return i+1
        return None

    if not max_epoch:
        index = len(epoch) - 1
    else:
        index = get_index(epoch, max_epoch)
                

    plt.close('all')
    errorplt = plt.plot(epoch[0:index], errors[0:index])
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
#    plt.xlim(0,p)
    plt.ylim(1e-5,1)

    if save is True:
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
        plt.savefig(date_str+'.png', transparent=True)
    else:
        plt.show()
        

def load(path):
    """ Load model parameters from path. """
    logger.info("Loading from %s ..." % path)
    file = open(path, 'rb')
    errorlog = pickle.load(file)
    file.close()
    return errorlog

#===============================================================================
# Methods
#===============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

#    errorlog = load('./models/NN_DALL_MSPT_Short/NN_DALL_MSPT_Short_Errorlog.pkl')
    errorlog = load('../RecurrentNeuralNetwork_Theano/models/RNN_minibatch_DALL_MSPT_BPTT-1_Short/RNN_minibatch_DALL_MSPT_BPTT-1_Short_Errorlog.pkl')
    PlotErrorlog(errorlog, 10000, save=True)
