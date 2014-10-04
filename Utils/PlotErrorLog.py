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
import seaborn as sns

logger = logging.getLogger(__name__)
plt.rc('figure.subplot',left=0.1,right=0.94,hspace=0,wspace=0,bottom=0.1,top=0.94)

def PlotErrorlog(errorlog, max_epoch=None, min_error_axis=None,save=False, fileName=""):
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
                
    if not min_error_axis:
        min_error_axis = 1e-5

    plt.close('all')
    errorplt = plt.plot(epoch[0:index], errors[0:index])
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
#    plt.xlim(0,p)
    plt.ylim(min_error_axis,1)

    if save is True:
        if fileName == "":
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            savename = "ErrorLog_" + date_str + '.png'
        else:
            savename = "ErrorLog_" + fileName + '.png'
        plt.savefig(savename)#, transparent=True)
        print "Save:", savename
        
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

    ESTIMATOR = "RNN"
    NAME='RNN_minibatch_D204060_MSPT_Short_f'
    
    loadDir = '../' + ESTIMATOR + '/models/' + NAME + '/' + NAME + '_Errorlog.pkl'

    errorlog = load(loadDir)
    PlotErrorlog(errorlog, max_epoch=300000, min_error_axis=1e-4, save=True, fileName=NAME)
