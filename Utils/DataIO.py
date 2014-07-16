# -*- coding: utf-8 -*-

import os
import csv
import numpy
import theano

import math
import matplotlib.pyplot as plt
#from Kinematics import CalculateForwardKinematics
from matplotlib.ticker import *
plt.rc('figure.subplot',left=0.03,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)

#===============================================================================
# Learning File Directories
#===============================================================================
LEARNING_FILE_DIR = ['../../AnalysisData/debug']
#LEARNING_FILE_DIR = ['../../AnalysisData/D30/Success']
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
RANGE_TIME       = range(0,1)
RANGE_MOTOR      = range(1,8)    # 9:示指DIPのデータは示指PIPと同じ関節角であり、ログ値は常に0なので使わない
RANGE_SIXAXIS    = range(9,21)
RANGE_PSV        = range(21,23)
RANGE_TACTILE    = range(23,95)
RANGE_SIZE       = range(95,96)

# Limit of Sensor Data
# モータ角LIMIT
#LIMIT_MOTOR = numpy.array([[-100, 11000], [0, 9100], [-1000, 8000], [-1000, 8000], [-500, 9700], [-1500, 1000], [0, 9500], [0, 9500]])
LIMIT_MOTOR = numpy.array([[-100, 11500], [0, 9500], [-1000, 8500], [-1000, 8500], [-500, 10000], [-1500, 5000], [0, 9500]])#, [0, 9500]])
# 6軸力覚センサLIMIT
# 拇指，示指共通
LIMIT_SIXAXIS = numpy.tile([-15000, 15000], (len(RANGE_SIXAXIS),1))     # -15000~15000
# ポテンショLIMIT
# MP; DIP
LIMIT_PSV = numpy.array([[-500, 4000], [-500, 6000]])    # -500~4000, -500~11000
# タクタイルLIMIT
LIMIT_TACTILE = numpy.tile([0, 200], (len(RANGE_TACTILE),1))    # -50~32670
# 物体サイズLIMIT
LIMIT_SIZE = numpy.array([[0,100]])

# Threshold
DROP_PSV = 300      # PSVがこの値を下回ったら対象物落下とみなす閾値
DROP_FORCE = 400    # 6軸合力がこの値を下回ったら対象物落下とみなす閾値

# Hand Link Parameter for Calculating Forward Kinematics
L_HT = numpy.array([21.4, 0.05, 14.0, 39.6, 39.5, 29.7])     # Thumb link length
L_HI = numpy.array([35.0, 70.7, 50.0, 32.0, 27.0])           # Index finger link length
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
                      "TACTILE":LIMIT_TACTILE,\
                      "SIZE":LIMIT_SIZE}
        self.RANGE = {"TIME":RANGE_TIME,\
                      "MOTOR":RANGE_MOTOR,\
                      "SIXAXIS":RANGE_SIXAXIS,\
                      "PSV":RANGE_PSV,\
                      "TACTILE":RANGE_TACTILE,\
                      "SIZE":RANGE_SIZE}

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
    for i in xrange(5):
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
        # OBJECT SIZE
        elif i == 4:
            RANGE = RANGE_SIZE
            LIMIT = LIMIT_SIZE

        # Round Data
        for j in xrange(len(RANGE)):
            idx = RANGE[0] + j
            handlingData[handlingData[:,:,idx] < LIMIT[j, 0], idx] = LIMIT[j, 0]
            handlingData[handlingData[:,:,idx] > LIMIT[j, 1], idx] = LIMIT[j, 1]

def ScalingHandlingData(handlingData):
    for i in xrange(5):
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
        # OBJECT SIZE
        elif i == 4:
            RANGE = RANGE_SIZE
            LIMIT = LIMIT_SIZE

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

def CalculateObjectSizeBySolvingForwardKinematics(handlingData):
    motor = handlingData[:,:,RANGE_MOTOR] / 100 * (numpy.pi / 180)  # モータ角[rad]
    psv = handlingData[:,:,RANGE_PSV] / 100 * (numpy.pi / 180)      # バネ伸展角[rad]

    linkParamT = []
    linkParamI = []
    N = handlingData.shape[0]
    T = handlingData.shape[1]

    # リンクパラメータ行列
    # 時刻tにおける4次元の行ベクトルを転置して4次元列ベクトルにし，それを関節毎に行方向にならべ，
    # その列ベクトルを時間毎に列方向に並べたものがLinkParam行列(4*6(関節数)*2(拇指，示指) = 48(*T)次元)
    # ai-1, αi-1, di, Θi
    for i in xrange(N):
        lpt = numpy.array([
                 [numpy.tile(L_HT[0],T), numpy.tile( .5*numpy.pi,T), numpy.tile(     .0,T),  numpy.tile(.5*numpy.pi,T)],
                 [numpy.tile(L_HT[1],T), numpy.tile(          .0,T), numpy.tile(     .0,T), motor[i,:,0] +    numpy.pi],
                 [numpy.tile(L_HT[2],T), numpy.tile(-.5*numpy.pi,T), numpy.tile(L_HT[3],T), motor[i,:,1] +    numpy.pi],
                 [numpy.tile(     .0,T), numpy.tile(-.5*numpy.pi,T), numpy.tile(     .0,T), motor[i,:,2] - .5*numpy.pi],
                 [numpy.tile(L_HT[4],T), numpy.tile(          .0,T), numpy.tile(     .0,T), motor[i,:,3]              ],
                 [numpy.tile(L_HT[5],T), numpy.tile(          .0,T), numpy.tile(     .0,T),  numpy.tile(         .0,T)],
             ])
        lpi = numpy.array([
                 [numpy.tile(L_HI[0],T), numpy.tile( .5*numpy.pi,T), numpy.tile(L_HI[1],T),              numpy.tile(-.5*numpy.pi,T)],
                 [numpy.tile(     .0,T), numpy.tile( .5*numpy.pi,T), numpy.tile(     .0,T), motor[i,:,4] - psv[i,:,0] + .5*numpy.pi],
                 [numpy.tile(     .0,T), numpy.tile(-.5*numpy.pi,T), numpy.tile(     .0,T), motor[i,:,5]                           ],
                 [numpy.tile(L_HI[2],T), numpy.tile( .5*numpy.pi,T), numpy.tile(     .0,T), motor[i,:,6]                           ],
                 [numpy.tile(L_HI[3],T), numpy.tile(          .0,T), numpy.tile(     .0,T), motor[i,:,6] - psv[i,:,1]              ],
                 [numpy.tile(L_HI[4],T), numpy.tile(          .0,T), numpy.tile(     .0,T),              numpy.tile(          .0,T)],
             ])
        linkParamT.append(lpt)
        linkParamI.append(lpi)
    LinkParam = {"Thumb": numpy.array(linkParamT), "Index": numpy.array(linkParamI)}
    
    # 同次変換行列(転置表現)
    # Thumb[i] or Index[i]の添字i:各リンク(L0~L5)に対応
    # 行方向：Thumb_L0~L5,IndexL_0~L_5に対する同次変換行列を4*4から16次元の列ベクトルに整形したもの
    # 列方向：時間方向
    # -------------------------------------以下の行列を転置して，16次元の列ベクトルに整形
    # cosθi           ,    -sinθi          ,    0         ,    ai
    # cosαi*sinθi    ,    cosαi*cosθi    ,    -sinαi   ,    -sinαi*di
    # sinαi*sinθi    ,    sinαi*cosθi    ,    cosαi    ,    cosαi*di
    # 0                ,    0                ,    0         ,    1 
    class CHTMatrix():
        def __init__(self):
            self.Thumb = numpy.zeros([N,6,16,T])
            self.Index = numpy.zeros([N,6,16,T])
    HTMatrix = CHTMatrix()
        
    for i in xrange(N):
        for j in xrange(6):
            HTMatrix.Thumb[i,j] = numpy.array([
                 numpy.cos(LinkParam["Thumb"][i,j,3]),
                 numpy.cos(LinkParam["Thumb"][i,j,1]) * numpy.sin(LinkParam["Thumb"][i,j,3]),
                 numpy.sin(LinkParam["Thumb"][i,j,1]) * numpy.sin(LinkParam["Thumb"][i,j,3]),
                 numpy.tile(0,T),
                -numpy.sin(LinkParam["Thumb"][i,j,3]),
                 numpy.cos(LinkParam["Thumb"][i,j,1]) * numpy.cos(LinkParam["Thumb"][i,j,3]),
                 numpy.sin(LinkParam["Thumb"][i,j,1]) * numpy.cos(LinkParam["Thumb"][i,j,3]),
                 numpy.tile(0,T),
                 numpy.tile(0,T),
                -numpy.sin(LinkParam["Thumb"][i,j,1]),
                 numpy.cos(LinkParam["Thumb"][i,j,1]),
                 numpy.tile(0,T),
                 LinkParam["Thumb"][i,j,0],
                -numpy.sin(LinkParam["Thumb"][i,j,1]) * LinkParam["Thumb"][i,j,2],
                 numpy.cos(LinkParam["Thumb"][i,j,1]) * LinkParam["Thumb"][i,j,2],
                 numpy.tile(1,T),
            ])
            HTMatrix.Index[i,j] = numpy.array([
                 numpy.cos(LinkParam["Index"][i,j,3]),
                 numpy.cos(LinkParam["Index"][i,j,1]) * numpy.sin(LinkParam["Index"][i,j,3]),
                 numpy.sin(LinkParam["Index"][i,j,1]) * numpy.sin(LinkParam["Index"][i,j,3]),
                 numpy.tile(0,T),
                -numpy.sin(LinkParam["Index"][i,j,3]),
                 numpy.cos(LinkParam["Index"][i,j,1]) * numpy.cos(LinkParam["Index"][i,j,3]),
                 numpy.sin(LinkParam["Index"][i,j,1]) * numpy.cos(LinkParam["Index"][i,j,3]),
                 numpy.tile(0,T),
                 numpy.tile(0,T),
                -numpy.sin(LinkParam["Index"][i,j,1]),
                 numpy.cos(LinkParam["Index"][i,j,1]),
                 numpy.tile(0,T),
                 LinkParam["Index"][i,j,0],
                -numpy.sin(LinkParam["Index"][i,j,1]) * LinkParam["Index"][i,j,2],
                 numpy.cos(LinkParam["Index"][i,j,1]) * LinkParam["Index"][i,j,2],
                 numpy.tile(1,T),
            ])

    # 同次変換行列を掛けあわせたドット積（世界座標系における各リンクの位置・姿勢を計算）
    # Thumb[i] or Index[i]の添字i:各リンク(L0~L5)に対応
    # 行方向:最終的な4*4行列を16次元列ベクトルに整形して行方向に並べたもの
    # 列方向:時間方向
    CalcHTM = CHTMatrix()
    
    # Thumb_L0; Index_L0 の同次変換行列を以後のドット積計算のために転置を戻した形式（通常の形式）に変更しておく
    for i in xrange(N):
        CalcHTM.Thumb[i,0] = numpy.vstack([HTMatrix.Thumb[i,0,0:13:4], HTMatrix.Thumb[i,0,1:14:4], HTMatrix.Thumb[i,0,2:15:4], HTMatrix.Thumb[i,0,3:16:4]])
        CalcHTM.Index[i,0] = numpy.vstack([HTMatrix.Index[i,0,0:13:4], HTMatrix.Index[i,0,1:14:4], HTMatrix.Index[i,0,2:15:4], HTMatrix.Index[i,0,3:16:4]])
    
#    print CalcHTM.Thumb[0,0].shape
#    print HTMatrix.Thumb[0,0].shape
    for i in xrange(N):
        for j in xrange(6-1):
            CalcHTM.Thumb[i,j+1] = numpy.array([
               numpy.sum(CalcHTM.Thumb[i,j,0:4  ] * HTMatrix.Thumb[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,0:4  ] * HTMatrix.Thumb[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,0:4  ] * HTMatrix.Thumb[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,0:4  ] * HTMatrix.Thumb[i,j+1,12:16], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,4:8  ] * HTMatrix.Thumb[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,4:8  ] * HTMatrix.Thumb[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,4:8  ] * HTMatrix.Thumb[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,4:8  ] * HTMatrix.Thumb[i,j+1,12:16], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,8:12 ] * HTMatrix.Thumb[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,8:12 ] * HTMatrix.Thumb[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,8:12 ] * HTMatrix.Thumb[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,8:12 ] * HTMatrix.Thumb[i,j+1,12:16], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,12:16] * HTMatrix.Thumb[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,12:16] * HTMatrix.Thumb[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,12:16] * HTMatrix.Thumb[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Thumb[i,j,12:16] * HTMatrix.Thumb[i,j+1,12:16], axis=0),
            ])
            CalcHTM.Index[i,j+1] = numpy.array([
               numpy.sum(CalcHTM.Index[i,j,0:4  ] * HTMatrix.Index[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,0:4  ] * HTMatrix.Index[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,0:4  ] * HTMatrix.Index[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,0:4  ] * HTMatrix.Index[i,j+1,12:16], axis=0),
               numpy.sum(CalcHTM.Index[i,j,4:8  ] * HTMatrix.Index[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,4:8  ] * HTMatrix.Index[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,4:8  ] * HTMatrix.Index[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,4:8  ] * HTMatrix.Index[i,j+1,12:16], axis=0),
               numpy.sum(CalcHTM.Index[i,j,8:12 ] * HTMatrix.Index[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,8:12 ] * HTMatrix.Index[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,8:12 ] * HTMatrix.Index[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,8:12 ] * HTMatrix.Index[i,j+1,12:16], axis=0),
               numpy.sum(CalcHTM.Index[i,j,12:16] * HTMatrix.Index[i,j+1,0:4  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,12:16] * HTMatrix.Index[i,j+1,4:8  ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,12:16] * HTMatrix.Index[i,j+1,8:12 ], axis=0),
               numpy.sum(CalcHTM.Index[i,j,12:16] * HTMatrix.Index[i,j+1,12:16], axis=0),
            ])
            
    fingertipCenterPos = {
                            "Thumb": CalcHTM.Thumb[:,5,3:13:4],
                            "Index": CalcHTM.Index[:,5,3:13:4],
                         }
    
    fingertipDistance = numpy.sqrt((fingertipCenterPos["Thumb"][:,0] - fingertipCenterPos["Index"][:,0]) ** 2 +
                                   (fingertipCenterPos["Thumb"][:,1] - fingertipCenterPos["Index"][:,1]) ** 2 +
                                   (fingertipCenterPos["Thumb"][:,2] - fingertipCenterPos["Index"][:,2]) ** 2)

    # 時刻0の指先中心間距離を把持物体のサイズとして認識する
    objectSize = fingertipDistance[:,0]

    # 学習用に整形
    objectSize = numpy.tile(objectSize[:,numpy.newaxis], T)
    objectSize = objectSize[:,:,numpy.newaxis]
    
    return objectSize


def LoadHandlingData(loadDirs=LEARNING_FILE_DIR):
    print('------------------------------')
    print('| Load Handling Data...      |')
    print('------------------------------')

    HD = []
    for loadDir in loadDirs:
        handlingData = LoadFile(loadDir)
        
        print('------------------------------')
        print('| Calculate Object Size...   |')
        print('------------------------------')
        objectSize = CalculateObjectSizeBySolvingForwardKinematics(handlingData)
        # 物体サイズを行列に付加
        handlingData = numpy.c_[handlingData, objectSize]
        
        print('------------------------------')
        print('| Check Limit...             |')
        print('------------------------------')
        CheckLimit(handlingData)

        print('------------------------------')
        print('| Scaling Data...            |')
        print('------------------------------')
        handlingData = ScalingHandlingData(handlingData)

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

