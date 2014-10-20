# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
import datetime
import math
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import seaborn

import sys
sys.path.append('../Utils')
from DataIO import LoadHandlingData, PrepLearningData, ReshapeForNN, ShorteningTimeStep
from NN import MetaNN

logger = logging.getLogger(__name__)
# plt.rc('figure.subplot',left=0.08,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)
plt.rc('figure.subplot',left=0.05,right=0.982,hspace=0,wspace=0,bottom=0.05,top=0.970)

#===============================================================================
# Methods
#===============================================================================
def PlotOutput(model, x_t, dataRange, plotNum=None, save=False):
    dataRange = np.array(dataRange) - 1
    step = x_t.shape[1]

    if plotNum is None:
        sequences = x_t.shape[0]
    else:
        sequences = plotNum

    MAX_PLOT = 25
    col = 5
    
    if sequences < col:
        col = sequences
    
    if sequences > MAX_PLOT:
        row = MAX_PLOT / col
    else:
        row = int(math.ceil(float(sequences) / col))
    for seq in xrange(sequences):
        if (seq % MAX_PLOT) == 0:
            plt.figure()
            count = 1

        ax = plt.subplot(row,col,count)
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
        count = count + 1

    if save is True:
        date_obj = datetime.datetime.now()
        date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
        plt.savefig(date_str+'.png', transparent=False)
    else:
        plt.legend(['T CM1','T CM2', 'T MP', 'T IP', 'I MP1', 'I MP2', 'I PIP'], loc='best')
        plt.show()
        
def Learning(x_t, x_tp1, NAME='', randomize=False):
    batch_size = 100#x_t.shape[1]
    x_t, x_tp1 = ReshapeForNN(x_t, x_tp1)
    
    # Randomize sequences(Not divide test data, use all loaded data)
    if randomize == True:
        x_t_train, x_t_test, x_tp1_train, x_tp1_test = train_test_split(x_t, x_tp1, test_size=0.0, random_state=int(time.time()))
        x_t = x_t_train
        x_tp1 = x_tp1_train

    # ネットワークと訓練モデルの構築
    model = MetaNN(n_in=x_t.shape[1], hidden_layers_sizes=[100], n_out=x_tp1.shape[1],
                   n_epochs=100000, batch_size=batch_size, t_error = 1e-6, 
                   L1_reg=0.00, L2_reg=1e-4, activation='tanh',
                   learning_rate=0.001, learning_rate_decay=0.99999,
                   final_momentum=0.9, initial_momentum=0.5, momentum_switchover=2000,
                   numpy_rng_seed=int(time.time()),
                   snapshot_every=10000, snapshot_path='./models/tmp/'+NAME)

    # 訓練モデルを用いてネットワークの学習を実行('q'で学習の切り上げ)
    model.fit(x_t, x_tp1, validate_every=100, show_norms = False, show_output = False, error_logging=True)

    # 学習したネットワークパラメータを保存
    model.save(fpath='./models/'+NAME,fname=NAME,save_errorlog=True)

def Testing(x_t, NAME='', plotNum=None):
    # ネットワークの構築
    model = MetaNN()
    
    loadDir = './models/' + NAME + '/' + NAME + '.pkl'

    # 学習済みのネットワークパラメータを読み込む
    model.load(loadDir)

    # Plot
#    PlotOutput(model, x_t, HandlingData.RANGE['MOTOR'], plotNum=plotNum)
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
#     HandlingData= LoadHandlingData(['../../AnalysisData/D60/Success'])
    
    # 読み込んだ操り試技データを学習用に整形
#     HandlingData = ShorteningTimeStep(HandlingData)
    x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV','TACTILE','SIZE'])
#     x_t, x_tp1 = PrepLearningData(HandlingData,['MOTOR','SIXAXIS','PSV','TACTILE'])
#    x_t, x_tp1 = PrepLearningData(HandlingData,
#                                  trainType  =['MOTOR','SIXAXIS','PSV','TACTILE'],
#                                  teacherType=['MOTOR','SIXAXIS','PSV','TACTILE'])
#     NAME='NN_20140919_MSPTS'
#     NAME='NN_DALL_MSPTS_Short'
#    NAME='NN_DALL_MSPT_Short_batch20_f'
#    NAME='NN_D204060_MSPT_Short_batch20_f'
    NAME='NN_D204060_MSPTS_Short_batch100_f'
#     NAME='NN_DALL_MSPT_Short_batch100_f'
#     NAME='NN_D204060_MSPTS_Short_batch1f'
#     NAME='NN_D204060_MSPTS_Short_batch100_m2000_lr99999_f'
#    NAME='NN_D204060_MSPTS_Short'

    # 学習
#     Learning(x_t, x_tp1, NAME, randomize=True)
    
    # テスト
    Testing(x_t, NAME)
#     Testing(x_t, NAME, plotNum=4)
