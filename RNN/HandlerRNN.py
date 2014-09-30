# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import *

import sys
sys.path.append('../Utils')
from DataIO import LoadHandlingData, PrepLearningData, ReshapeForRNN_minibatch, ShorteningTimeStep
from RNN import MetaRNN

logger = logging.getLogger(__name__)
plt.rc('figure.subplot',left=0.03,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)

#===============================================================================
# Methods
#===============================================================================
def PlotOutput(model, x_t, dataRange, save=False):
    dataRange = np.array(dataRange) - 1
    sequences = x_t.shape[1]
    step = x_t.shape[0]
    col = 5.0
    row = int(math.ceil(sequences / col))

    fig = plt.figure()

    for seq in xrange(sequences):
        ax = plt.subplot(row,col,seq+1)
        xs_t = x_t[:, seq, :]
        xs_tp1 = x_t[:, seq, :]
        
        true_targets = plt.plot(xs_tp1[:,dataRange])
        plt.xlim(0,step)
        plt.ylim(-1,1)

        if (seq % col) != 0:
            ax.yaxis.set_major_formatter(NullFormatter())
        if int(math.ceil((seq+1) / col)) != row:
            ax.xaxis.set_major_formatter(NullFormatter())

        guess = model.predict(xs_t[:, np.newaxis, :])
        guessed_targets = plt.plot(guess[:,0,dataRange], linestyle='--')
        for i, x in enumerate(guessed_targets):
            x.set_color(true_targets[i].get_color())
    if save is True:
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
        plt.savefig(date_str+'.png', transparent=False)
    else:
        plt.show()

def Learning(x_t, x_tp1, NAME):
    batch_size = x_t.shape[1]
    
    # ネットワークと訓練モデルの構築
    model = MetaRNN(n_in=x_t.shape[2], n_hidden=100, n_out=x_tp1.shape[2],
                    truncated_num=-1,
                    learning_rate=0.001, learning_rate_decay=0.99999,
                    n_epochs=100000, t_error = 1e-6, batch_size=batch_size,
                    activation='tanh', L2_reg=1e-4,
                    snapshot_every=10000, snapshot_path='./models/tmp/'+NAME)

    # 訓練モデルを用いてネットワークの学習を実行('q'で学習の切り上げ)
    model.fit(x_t, x_tp1, validate_every=100, optimizer='sgd',
            show_norms = False, show_output = False, error_logging=True)

    # 学習したネットワークパラメータを保存
    model.save(fpath='./models/'+NAME,fname=NAME,save_errorlog=True)

def Testing(x_t, NAME):
    # ネットワークの構築
    model = MetaRNN()

    loadDir = './models/' + NAME + '/' + NAME + '.pkl'

    # 学習済みのネットワークパラメータを読み込む
    model.load(loadDir)

    # Plot
    PlotOutput(model, x_t, HandlingData.RANGE['MOTOR'])
#    PlotOutput(model, x_t, HandlingData.RANGE['SIXAXIS'])
#    PlotOutput(model, x_t, HandlingData.RANGE['SIXAXIS'][0:6])
#    PlotOutput(model, x_t, HandlingData.RANGE['SIXAXIS'][6:12])
#    PlotOutput(model, x_t, HandlingData.RANGE['PSV'])
#    PlotOutput(model, x_t, HandlingData.RANGE['TACTILE'])

#===============================================================================
# Main
#===============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 操り試技データの読み込み
    HandlingData= LoadHandlingData()
    
    # 読み込んだ操り試技データを学習用に整
    HandlingData = ShorteningTimeStep(HandlingData)
#     x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV','TACTILE','SIZE'])
    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV','TACTILE'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR'])
    x_t, x_tp1 = ReshapeForRNN_minibatch(x_t, x_tp1)

    NAME='RNN_minibatch_D204060_MSPT_Short'

    # 学習
    Learning(x_t, x_tp1, NAME)
    
    # テスト
#    Testing(x_t, NAME)
