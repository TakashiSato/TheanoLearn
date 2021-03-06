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
from SdA import MetaSdA

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
        dec = model.encode(xs_t)
        guess = model.decode(dec)
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
    batch_size = 1000
    x_t, x_tp1 = ReshapeForNN(x_t, x_tp1)

#    hidden_layers_sizes = [50,30,10]
#    hidden_layers_sizes = [8,4]
    hidden_layers_sizes = [70,50,30,10]
#    hidden_layers_sizes = [80,60,40,20]
#    corruption_levels = [.0,.0,.0,.0]
    corruption_levels = [.1,.2,.3,.3]
#    corruption_levels = [.1,.2]
    
#    NAME='SdA_DALL_H1510_corr.1.2_NoTac_Long'
#    NAME='SdA_D40_H80604020_corr.1.2.3.3_Long'
#    NAME='SdA_DALL_H70503010_corr.0.0.0.0_Long'
#    NAME='SdA_DALL_H70503010_corr.1.2.3.3_Long_outlin_lrd1'
#    NAME='SdA_DALL_H503010_corr.1.2.3_Long_outlin_lrd1_onlyTac'
#    NAME='SdA_DALL_H0804_corr.1.2_Long_outlin_lrd1_only6axis'
    NAME='SdA_DALL_H70503010_corr.1.2.3.3_Long_outlin_lrd1_Tac6axis'
    
    # ネットワークと訓練モデルの構築
    model = MetaSdA(n_in=x_t.shape[1], hidden_layers_sizes=hidden_layers_sizes,
                    corruption_levels=corruption_levels,
                    learning_rate=0.001, learning_rate_decay=1,#0.999999,
                    n_epochs=10000, batch_size=batch_size, t_error=1e-6,
                    hidden_layer_activation='tanh', output_layer_activation='linear',
                    numpy_rng_seed=int(time.time()), theano_rng_seed=int(time.time()),
                    final_momentum=0.9, initial_momentum=0.5, momentum_switchover=300,
                    snapshot_every=1000, snapshot_path='./models/tmp/'+NAME)

    # 訓練モデルを用いてネットワークの学習を実行('q'で学習の切り上げ)
    model.fit(x_t, validate_every=100, show_norms = False, show_output = False, error_logging=True)

    # 学習したネットワークパラメータを保存
    model.save(fpath='./models/'+NAME, fname=NAME,save_errorlog=True)

def Testing(x_t, x_tp1):
    x_t, x_tp1 = ReshapeForNN(x_t, x_tp1)

    # ネットワークの構築
    model = MetaSdA()

    # 学習済みのネットワークパラメータを読み込む
    model.load('./models/SdA_DALL_H70503010_corr.1.2.3.3_Long_outlin_lrd1_Tac6axis/SdA_DALL_H70503010_corr.1.2.3.3_Long_outlin_lrd1_Tac6axis.pkl')

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
#    HandlingData = ShorteningTimeStep(HandlingData)
    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV','TACTILE'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['TACTILE'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,['SIXAXIS'])
    
    # 学習
    Learning(x_t, x_tp1)
    
    # テスト
#    Testing(x_t, x_tp1)
