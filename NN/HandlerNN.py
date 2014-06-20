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
from DataIO import LoadHandlingData, PrepLearningData, ReshapeForNN, ShorteningTimeStep
from NN import MetaNN

logger = logging.getLogger(__name__)
plt.rc('figure.subplot',left=0.03,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)

#===============================================================================
# Methods
#===============================================================================
def PlotOutput(model, x_t, dataRange, save=False):
    dataRange = np.array(dataRange) - 1
    sequences = x_t.shape[0]
    step = x_t.shape[1]
    col = 5.0
    row = int(math.ceil(sequences / col))

    fig = plt.figure()

    for seq in xrange(sequences):
        ax = plt.subplot(row,col,seq+1)
        xs_t = x_t[seq]

        true_sequence = plt.plot(xs_t[:,dataRange])
        plt.xlim(0,step)
        plt.ylim(-1,1)

        if (seq % col) != 0:
            ax.yaxis.set_major_formatter(NullFormatter())
        if int(math.ceil((seq+1) / col)) != row:
            ax.xaxis.set_major_formatter(NullFormatter())
        guess = model.predict(xs_t)
        guessed_targets = plt.plot(guess[:,dataRange], linestyle='--')
        for i, x in enumerate(guessed_targets):
            x.set_color(true_sequence[i].get_color())
    if save is True:
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
        plt.savefig(date_str+'.png', transparent=False)
    else:
        plt.show()
        
def Learning(x_t, x_tp1):
    batch_size = x_t.shape[1]
    x_t, x_tp1 = ReshapeForNN(x_t, x_tp1)
    
    NAME='NN_DALL_MSPT_Short'
    
    # ネットワークと訓練モデルの構築
    model = MetaNN(n_in=x_t.shape[1], hidden_layers_sizes=[100], n_out=x_tp1.shape[1],
                   n_epochs=1000000, batch_size=batch_size, t_error = 1e-6, 
                   L1_reg=0.00, L2_reg=1e-4, activation='tanh',
                   learning_rate=0.001, learning_rate_decay=0.999999,
                   final_momentum=0.9, initial_momentum=0.5, momentum_switchover=100,
                   numpy_rng_seed=int(time.time()),
                   snapshot_every=10000, snapshot_path='./models/tmp/'+NAME)

    # 訓練モデルを用いてネットワークの学習を実行('q'で学習の切り上げ)
    model.fit(x_t, x_tp1, validate_every=100, show_norms = False, show_output = False, error_logging=True)

    # 学習したネットワークパラメータを保存
    model.save(fpath='./models/'+NAME,fname=NAME,save_errorlog=True)

def Testing(x_t):
    # ネットワークの構築
    model = MetaNN()

    # 学習済みのネットワークパラメータを読み込む
    model.load('./models/NN_DALL_MSPT_Short/NN_DALL_MSPT_Short.pkl')

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
    
    # 読み込んだ操り試技データを学習用に整形
    HandlingData = ShorteningTimeStep(HandlingData)
    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV','TACTILE'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR'])

    # 学習
#    Learning(x_t, x_tp1)
    
    # テスト
#    Testing(x_t)
