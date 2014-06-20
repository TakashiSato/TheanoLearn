# -*- coding: utf-8 -*-

import os
import csv
import numpy
import theano

import math
import matplotlib.pyplot as plt
from matplotlib.ticker import *
plt.rc('figure.subplot',left=0.03,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)

#===============================================================================
# Learning File Directories
#===============================================================================
LEARNING_FILE_DIR = ['../../AnalysisData/debug']
#LEARNING_FILE_DIR = ['../../AnalysisData/D40/Success']
#LEARNING_FILE_DIR = ['../../AnalysisData/D20/Success',\
#                     '../../AnalysisData/D40/Success',\
#                     '../../AnalysisData/D60/Success']
#LEARNING_FILE_DIR = ['../../AnalysisData/D20/Success',\
#                     '../../AnalysisData/D30/Success',\
#                     '../../AnalysisData/D40/Success',\
#                     '../../AnalysisData/D50/Success',\
#                     '../../AnalysisData/D60/Success']

#===============================================================================
# Hand Parameter
#===============================================================================
# Range of Sensor Data
RANGE_TIME      = range(0,1)
RANGE_MOTOR     = range(1,9)
RANGE_SIXAXIS   = range(9,21)
RANGE_PSV       = range(21,23)
RANGE_TACTILE   = range(23,95)

# Limit of Sensor Data
# モータ角LIMIT
#LIMIT_MOTOR = numpy.array([[-100, 11000], [0, 9100], [-1000, 8000], [-1000, 8000], [-500, 9700], [-1500, 1000], [0, 9500], [0, 9500]])
LIMIT_MOTOR = numpy.array([[-100, 11500], [0, 9500], [-1000, 8500], [-1000, 8500], [-500, 10000], [-1500, 5000], [0, 9500], [0, 9500]])
# 6軸力覚センサLIMIT
# 拇指，示指共通
LIMIT_SIXAXIS = numpy.tile([-15000, 15000], (len(RANGE_SIXAXIS),1))     # -15000~15000
# ポテンショLIMIT
# MP; DIP
LIMIT_PSV = numpy.array([[-500, 4000], [-500, 6000]])    # -500~4000, -500~11000
# タクタイルLIMIT
LIMIT_TACTILE = numpy.tile([0, 200], (len(RANGE_TACTILE),1))    # -50~32670

#LIMIT_MOTOR     = [11500, 9500, 8500, 8500, 10000, 5000, 9500, 1]
#LIMIT_FORCE     = [7200]
#LIMIT_PSV       = [5000, 9500]
#LIMIT_TACTILE   = [200]

DROP_PSV = 300      # PSVがこの値を下回ったら対象物落下とみなす閾値
DROP_FORCE = 400    # 6軸合力がこの値を下回ったら対象物落下とみなす閾値
#------------------------------------------------------------------------------ 

#===============================================================================
# Class
#===============================================================================
class CHandlingData(object):
    def __init__(self, data):
        self.data = data
        self.LIMIT = {"MOTOR":LIMIT_MOTOR,\
                      "SIXAXIS":LIMIT_SIXAXIS,\
                      "PSV":LIMIT_PSV,\
                      "TACTILE":LIMIT_TACTILE}
        self.RANGE = {"TIME":RANGE_TIME,\
                      "MOTOR":RANGE_MOTOR,\
                      "SIXAXIS":RANGE_SIXAXIS,\
                      "PSV":RANGE_PSV,\
                      "TACTILE":RANGE_TACTILE}

#===============================================================================
# Methods 
#===============================================================================
def LoadCSV(loadFile):
    ''' Load External CSV File
    1行目：ラベル，2行目以降：数値 となっているcsvファイルを読み込む
        
    :type loadFile: string
    :param loadFile: 読み込むcsvファイルの場所を示す文字列 
        
    '''
    
    [dataDir, dataFile] = os.path.split(loadFile)
    
    # Check
    if not os.path.isfile(loadFile):
        print('File not found.')
        return

    print('Load [%s]...' % dataFile)
    csvfile = open(loadFile)

    data = []
    for i, row in enumerate(csv.reader(csvfile)):
        # 空白文字の削除
        while row.count('') > 0:
            row.remove('')

        # ラベル部は文字列として格納
        if i == 0:
            label = row

        # 数値部は数値に変換してリストに格納
        else:
            data.append(map((lambda x: float(x)), row))
    
    csvfile.close()

    return label, data

def SaveCSV(saveFileName, label, data):
    csvfile = csv.writer(file(saveFileName, 'w'))
    csvfile.writerow(label)
    for row in data:
        csvfile.writerow(row)


def LoadFile(loadFilePath):
    files = os.listdir(loadFilePath)
    handlingData = []

    print('Loading [%s]' % loadFilePath + '/')
    for file in files:
#        print file
        [label, data] = LoadCSV(loadFilePath + '/' + file)
        handlingData.append(data)
        
    return numpy.array(handlingData)


def CheckLimit(handlingData):
    for i in xrange(4):
        # MOTOR
        if   i == 0:
            RANGE = RANGE_MOTOR
            LIMIT = LIMIT_MOTOR
        # SIXAXIS
        elif i == 1:
            RANGE = RANGE_SIXAXIS
            LIMIT = LIMIT_SIXAXIS
        # PSV
        elif i == 2:
            RANGE = RANGE_PSV
            LIMIT = LIMIT_PSV
        # TACTILE
        elif i == 3:
            RANGE = RANGE_TACTILE
            LIMIT = LIMIT_TACTILE

        # Round Data
        for j in xrange(len(RANGE)):
#            print('[%d] LIMIT: %d ~ %d' % (j, LIMIT[j, 0], LIMIT[j, 1]))
            idx = RANGE[0] + j
            handlingData[handlingData[:,:,idx] < LIMIT[j, 0], idx] = LIMIT[j, 0]
            handlingData[handlingData[:,:,idx] > LIMIT[j, 1], idx] = LIMIT[j, 1]
#            print handlingData[:, :, idx]


def ScalingHandlingData(handlingData):
    for i in xrange(4):
        # MOTOR
        if   i == 0:
            RANGE = RANGE_MOTOR
            LIMIT = LIMIT_MOTOR
        # SIXAXIS
        elif i == 1:
            RANGE = RANGE_SIXAXIS
            LIMIT = LIMIT_SIXAXIS
        # PSV
        elif i == 2:
            RANGE = RANGE_PSV
            LIMIT = LIMIT_PSV
        # TACTILE
        elif i == 3:
            RANGE = RANGE_TACTILE
            LIMIT = LIMIT_TACTILE

        # Scaling Data
        handlingData[:,:,RANGE] = (handlingData[:,:,RANGE] - (LIMIT[:,1] + LIMIT[:,0]) / 2.) / ((LIMIT[:,1] - LIMIT[:,0]) / 2.)

    return handlingData

def JudgeFallingTimeForFailureData(handlingData, isPlot=False):
    sequences = handlingData.shape[0]
    step = handlingData.shape[1]

    # DIPのPSV伸展角(DIPはノイズが多いのでMPのみを見る)
    mppsv = handlingData[:,:,RANGE_PSV[0]]
    # 拇指の6軸力成分(示指はノイズが多いので拇指のみを見る)
    thumbsixaxis = handlingData[:,:,RANGE_SIXAXIS[0:3]]
    # 拇指の6軸合力
    forcemagnitude = numpy.sqrt(numpy.sum(thumbsixaxis ** 2, axis=2))
    
    # PSVオフセット除去
    mppsv = (mppsv.T - numpy.min(mppsv, axis=1)).T
    # 合力オフセット除去
    forcemagnitude = (forcemagnitude.T - numpy.min(forcemagnitude, axis=1)).T
    
    # PSV伸展角基準
    judgePSV = mppsv < DROP_PSV
    # 6軸合力基準
    judgeForce = forcemagnitude < DROP_FORCE
    
    # 双方の基準を満たす場合を落下とみなす
    judge = judgePSV & judgeForce
    
    # 落下時刻を表すインデックスを取得
    FallingTime = numpy.zeros(sequences)
    for seq in xrange(sequences):
        idx = numpy.argwhere(judge[seq] == False)
        FallingTime[seq] = idx[-1]
        judge[seq, 1:idx[-1]] = 0

    # プロット
    if isPlot is True:
        col = 5.0
        row = int(math.ceil(sequences / col))
        fig = plt.figure()
        for seq in xrange(sequences):
            ax = plt.subplot(row,col,seq+1)
        
            plt.plot(forcemagnitude[seq])
            plt.plot(mppsv[seq])
            plt.plot(judge[seq]*6500)
            plt.xlim(0,step)
            plt.ylim(0,7000)
        
            if (seq % col) != 0:
                ax.yaxis.set_major_formatter(NullFormatter())
            if int(math.ceil((seq+1) / col)) != row:
                ax.xaxis.set_major_formatter(NullFormatter())
        plt.show()

    return FallingTime

def LoadHandlingData(loadDirs=LEARNING_FILE_DIR):
    print('------------------------------')
    print('| Load Handling Data...      |')
    print('------------------------------')

    HD = []
    for loadDir in loadDirs:
        handlingData = LoadFile(loadDir)
        
        print('------------------------------')
        print('| Check Limit...             |')
        print('------------------------------')
        CheckLimit(handlingData)
        
        print('------------------------------')
        print('| Scaling Data...            |')
        print('------------------------------')
        handlingData = ScalingHandlingData(handlingData)
#        HD.append(HandlingData(handlingData))
        HD.append(handlingData)
    
    print('------------------------------')
    print('| Complete...                |')
    print('------------------------------')
    return CHandlingData(HD)

def PrepLearningData(HandlingData, useSensor=['MOTOR', 'SIXAXIS', 'PSV']):
    # Formating Data for Learning
    train = []
    teacher = []
    for s in xrange(len(HandlingData.data)):
        for i in xrange(HandlingData.data[s].shape[0]):
            trainRange   = range(0, HandlingData.data[s].shape[1] - 1)    # Train data (t)
            teacherRange = range(1, HandlingData.data[s].shape[1])        # Teacher data (t+1)
            train.append(HandlingData.data[s][i, trainRange])
            teacher.append(HandlingData.data[s][i, teacherRange])
    train = numpy.array(train, dtype=theano.config.floatX)
    teacher = numpy.array(teacher, dtype=theano.config.floatX)
    
    # Delete Unnecessary Label Data
    lb = []
    for sensor in useSensor:
        lb += HandlingData.RANGE[sensor]
    
    train = train[:, :, lb]       # delete 'Time', 'Tactile' row
    teacher = teacher[:, :, lb]     # delete 'Time', 'Tactile' row
    
    return train, teacher

def ReshapeForRNN_minibatch(Learning_train, Learning_teacher):
    reshape_train = []

    for i in xrange(Learning_train.shape[1]):
        reshape_train.append(Learning_train[:,i,:])
    reshape_train = numpy.array(reshape_train)

    reshape_teacher = []
    for i in xrange(Learning_teacher.shape[1]):
        reshape_teacher.append(Learning_teacher[:,i,:])
    reshape_teacher = numpy.array(reshape_teacher)

    return reshape_train, reshape_teacher

def ReshapeForNN(Learning_train, Learning_teacher):

    for i in xrange(Learning_train.shape[0]):
        if i == 0:
            reshape_train = Learning_train[i]
        else:
            reshape_train = numpy.vstack([reshape_train, Learning_train[i]])

    for i in xrange(Learning_teacher.shape[0]):
        if i == 0:
            reshape_teacher = Learning_teacher[i]
        else: reshape_teacher = numpy.vstack([reshape_teacher, Learning_teacher[i]])

    return reshape_train, reshape_teacher

def ShorteningTimeStep(HandlingData, timeStep=100):
    shortHD = []
    for s in xrange(len(HandlingData.data)):
        l = HandlingData.data[s].shape[1]
        shortHD.append(HandlingData.data[s][:,range(0,l,l/timeStep),:])

    return CHandlingData(shortHD)


def SaveScalledHandlingData(loadDirs, failureTrial=False):
    print('------------------------------')
    print('| Start Saving Data...       |')
    print('------------------------------')
    for loadDir in loadDirs:
        handlingData = LoadFile(loadDir)
        
        if failureTrial is True:
            fallingTime = JudgeFallingTimeForFailureData(handlingData, isPlot=False)

        CheckLimit(handlingData)
        handlingData = ScalingHandlingData(handlingData)

        if failureTrial is True:
            for i in xrange(handlingData.shape[0]):
                handlingData[i,fallingTime[i]:,1:] = 0

        files = os.listdir(loadDir)
        for i, file in enumerate(files):
            [label, data] = LoadCSV(loadDir + '/' + file)
            SaveCSV(file, label, handlingData[i,:,:])

    print('------------------------------')
    print('| Done...                    |')
    print('------------------------------')
        
        
if __name__ == '__main__':
    # 操り試技データの読み込み
    handlingData = LoadHandlingData(LEARNING_FILE_DIR)
#    sparse = ShorteningTimeStep(handlingData)
#    plt.subplot(2,1,1)
#    plt.plot(handlingData.data[0][0,:,1:])
#    plt.subplot(2,1,2)
#    plt.plot(sparse.data[0][0,:,1:])
#    plt.show()

#    train, teacher = PrepLearningData(handlingData, ['MOTOR','SIXAXIS','PSV','TACTILE'])
#    plt.subplot(2,1,1)
#    plt.plot(handlingData.data[0][0,:,1:])
#    plt.subplot(2,1,2)
#    plt.plot(train[0,:,1:])
#    plt.show()

#    train, teacher = ReshapeForRNN_minibatch(train, teacher)
#    plt.plot(train[:,1,1:])
#    plt.show()

#    SaveScalledHandlingData(['../../AnalysisData/D60/Failure'], failureTrial=False)

