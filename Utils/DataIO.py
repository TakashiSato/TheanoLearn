# -*- coding: utf-8 -*-

import os
import csv
import numpy
import theano

import math
import matplotlib.pyplot as plt
#from Kinematics import CalculateForwardKinematics
from matplotlib.ticker import *
from theano.tensor.tests.test_extra_ops import numpy_16
from scipy.stats.vonmises_cython import numpy
plt.rc('figure.subplot',left=0.03,right=0.982,hspace=0,wspace=0,bottom=0.03,top=0.985)

#===============================================================================
# Learning File Directories
#===============================================================================
# LEARNING_FILE_DIR = ['../../AnalysisData/debug']
LEARNING_FILE_DIR = ['../../AnalysisData/20140919/D60']
# LEARNING_FILE_DIR = ['../../AnalysisData/D60/Success']
# LEARNING_FILE_DIR = ['../../AnalysisData/D20/Success',\
#                      '../../AnalysisData/D40/Success',\
#                      '../../AnalysisData/D60/Success']
# LEARNING_FILE_DIR = ['../../AnalysisData/D20/Success',\
#                      '../../AnalysisData/D30/Success',\
#                      '../../AnalysisData/D40/Success',\
#                      '../../AnalysisData/D50/Success',\
#                      '../../AnalysisData/D60/Success']

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
# RANGE_CURRENT    = range(95,103)
# RANGE_SIZE       = range(103,104)

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
# 電流LIMIT
# LIMIT_CURRENT = numpy.tile([-1000,10000], (len(RANGE_CURRENT),1))
# 物体サイズLIMIT
LIMIT_SIZE = numpy.array([[10,80]])

# Threshold
DROP_PSV = 300      # PSVがこの値を下回ったら対象物落下とみなす閾値
DROP_FORCE = 400    # 6軸合力がこの値を下回ったら対象物落下とみなす閾値

# Hand Link Parameter for Calculating Forward Kinematics
L_HT = numpy.array([21.4, 0.05, 14.0, 39.6, 39.5, 29.7])     # Thumb link length
L_HI = numpy.array([35.0, 70.7, 50.0, 32.0, 27.0])           # Index finger link length

# タクタイル接触判定圧力値
CONTACT_THRESHOLD_THUMB = 20
CONTACT_THRESHOLD_INDEX = 18 
    
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
        data = numpy.array(data)
        handlingData.append(data[:, 0:RANGE_TACTILE[-1]+1])     # 電流が追記されたCSVに対応するための措置
        
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

# タクタイル圧力値から接触セルの位置(13分割)を推定する
def EstimateContactCellPosition(handlingData):
    tactile = handlingData[:,0,RANGE_TACTILE]   # 時刻0のものだけ抜きだし
    N = tactile.shape[0]                        # データセット数
    Tactile = {"Thumb": tactile[:,0:36], "Index": tactile[:,37:72]}
    
    # セル配置テーブル:拇指
    #  X:指短手方向で、1-31番セルの列を座標0、12-19番セルの列を座標6と定義
    #  Y:指長手方向で、1-5番セルの列を座標0、35-36番セルの列を座標6と定義
    X = {"Thumb": numpy.zeros(36), "Index": numpy.zeros(36)}
    Y = {"Thumb": numpy.zeros(36), "Index": numpy.zeros(36)}
    for cell in range(1,37):
        x = cell - 1 - (cell > 5)*5 - (cell > 12)*7 - (cell > 19)*7 - (cell > 25)*6 - (cell > 30)*5 - (cell > 34)*3;
        y = (cell > 5)*1 + (cell > 12)*1 + (cell > 19)*1 + (cell > 25)*1 + (cell > 30)*1 + (cell > 34)*1;
        X["Thumb"][cell-1] = x
        Y["Thumb"][cell-1] = y
    # セル配置テーブル:示指
    #  X:指短手方向で、71-101番セルの列を座標0、82-89番セルの列を座標6と定義
    #  Y:指長手方向で、71-75番セルの列を座標0、105-106番セルの列を座標6と定義
    for cell in range(71,107):
        x = (cell - 71 - (cell > 75)*5 - (cell > 82)*7 - (cell > 89)*7 - (cell > 95)*6 - (cell > 100)*5 - (cell > 104)*3);
        y = (cell > 75)*1 + (cell > 82)*1 + (cell > 89)*1 + (cell > 95)*1 + (cell > 100)*1 + (cell > 104)*1;
        X["Index"][cell - 71] = x
        Y["Index"][cell - 71] = y
        
    # 圧力値を閾値と比較して接触セルを判定する
    contactCell = {"Thumb": Tactile["Thumb"] > CONTACT_THRESHOLD_THUMB,
                   "Index": Tactile["Index"] > CONTACT_THRESHOLD_INDEX}
    # 一つも接触セルがないと判定されてしまった場合、
    # 最も圧力値が大きいセルを接触セルとする
    for i in xrange(N):
        # 拇指
        if any(contactCell["Thumb"][i]) == False:
            temp_threshold = CONTACT_THRESHOLD_THUMB
            # 閾値を1ずつ下げていき、接触セルが見つかるか閾値が0になるまでループ
            while(any(contactCell["Thumb"][i]) == False and temp_threshold > 0):
                contactCell["Thumb"][i] = Tactile["Thumb"][i] > temp_threshold
                temp_threshold = temp_threshold - 1
            if temp_threshold == 0:
                print "!!!Error: Thumb[" + i + "]: Not found contact cell!!!"
                raw_input()
        # 示指
        if any(contactCell["Index"][i]) == False:
            temp_threshold = CONTACT_THRESHOLD_THUMB
            # 閾値を1ずつ下げていき、接触セルが見つかるか閾値が0になるまでループ
            while(any(contactCell["Index"][i]) == False and temp_threshold > 0):
                contactCell["Index"][i] = Tactile["Index"][i] > temp_threshold
                temp_threshold = temp_threshold - 1
            if temp_threshold == 0:
                print "!!!Error: Index[" + i + "]: Not found contact cell!!!"
                raw_input()

    
    # 接触中心の座標を計算: 接触とみなされたセルをX,Y独立に見たときの位置の平均値を接触中心として推定する
    contactCenterPos = {"Thumb": {"X": numpy.zeros(N), "Y": numpy.zeros(N)},
                        "Index": {"X": numpy.zeros(N), "Y": numpy.zeros(N)}}    # 計算結果格納用
    for i in xrange(N):
        contactCellPosUnique = {"Thumb": 
                                    {"X": numpy.unique(X["Thumb"][contactCell["Thumb"][i]]),
                                     "Y": numpy.unique(Y["Thumb"][contactCell["Thumb"][i]])},
                                "Index": 
                                    {"X": numpy.unique(X["Index"][contactCell["Index"][i]]),
                                     "Y": numpy.unique(Y["Index"][contactCell["Index"][i]])}}
        contactCenterPos["Thumb"]["X"][i] = numpy.mean(contactCellPosUnique["Thumb"]["X"])
        contactCenterPos["Thumb"]["Y"][i] = numpy.mean(contactCellPosUnique["Thumb"]["Y"])
        contactCenterPos["Index"]["X"][i] = numpy.mean(contactCellPosUnique["Index"]["X"])
        contactCenterPos["Index"]["Y"][i] = numpy.mean(contactCellPosUnique["Index"]["Y"])
    # 小数点を消すために、すべての結果を2倍する(それでも少数が出る場合があるので四捨五入する)
    # この計算により、X方向、Y方向ともに0,1,...,12で座標値が表される(つまり分解能は13)
    contactCenterPos["Thumb"]["X"] = numpy.round(contactCenterPos["Thumb"]["X"] * 2)
    contactCenterPos["Thumb"]["Y"] = numpy.round(contactCenterPos["Thumb"]["Y"] * 2)
    contactCenterPos["Index"]["X"] = numpy.round(contactCenterPos["Index"]["X"] * 2)
    contactCenterPos["Index"]["Y"] = numpy.round(contactCenterPos["Index"]["Y"] * 2)
    
    return contactCenterPos
    
def GetFingertipShapeTable():
    ## 指先形状テーブルの作成(※かなりアバウトな値)
    # 実測代表点: ProE 指先柔軟肉CADにおけるy座標.指短手方向の中心を通る軸がy軸.
    # y_sample[0],[1],...,[12]がそれぞれおEstimateContactCellPosition()で推定した各指Y座標値の0,1,...,12に対応する
    y_sample = numpy.array([19.9, 19.7, 19.4, 18.8, 18.0, 16.8, 15.3, 13.8, 12.0, 10.1, 8.0, 5.7, 3.0])
    z_sample = numpy.array(-(-32 * (1 - (1./20 * y_sample) ** 2) ** (1./3) + 3))
    
    # CAD座標からハンド座標(正確には、z原点:爪固定ネジ位置、y原点:指中心軸上)へ変換(実測に基づく...のでかなりアバウト)
    y_hand = y_sample - 7
    z_hand = z_sample - 19
    
    # 同次変換行列計算用パラメータの算出(ハンド座標原点から見たリンク長と回転角度)
    LinkLength = numpy.sqrt(z_hand ** 2 + y_hand ** 2)
    Theta = numpy.arctan2(y_hand, z_hand)
    
    return {"Y": y_hand, "Z": z_hand, "LinkLength": LinkLength, "Theta": Theta}

# 関節角度、タクタイルから推定した指先接触中心位置、指先形状テーブルから、時刻0時点でのハンド座標における指先接触位置を計算する
def CalculateObjectSize(handlingData, contactCenterPos, fingertipShapeTable):
    motor = handlingData[:,:,RANGE_MOTOR] / 100 * (numpy.pi / 180)  # モータ角[rad]
    psv = handlingData[:,:,RANGE_PSV] / 100 * (numpy.pi / 180)      # バネ伸展角[rad]

    linkParamT = []
    linkParamI = []
    N = handlingData.shape[0]
    T = 1 #handlingData.shape[1]    # どうせ時刻0の情報しか使わないので時間の長さは1としている
    
    # リンクパラメータ行列
    # 時刻tにおける4次元の行ベクトルを転置して4次元列ベクトルにし，それを関節毎に行方向にならべ，
    # その列ベクトルを時間毎に列方向に並べたものがLinkParam行列(4*6(関節数)*2(拇指，示指) = 48(*T)次元)
    # ai-1, αi-1, di, Θi
    for i in xrange(N):
        tipLLt = fingertipShapeTable["LinkLength"][contactCenterPos["Thumb"]["Y"][i]]
        tipTht = fingertipShapeTable["Theta"][contactCenterPos["Thumb"]["Y"][i]]
        lpt = numpy.array([
                 [numpy.tile(L_HT[0],T), numpy.tile( .5*numpy.pi,T), numpy.tile(     .0,T),    numpy.tile(.5*numpy.pi,T)],
                 [numpy.tile(L_HT[1],T), numpy.tile(          .0,T), numpy.tile(     .0,T), motor[i,0:T,0] +    numpy.pi],
                 [numpy.tile(L_HT[2],T), numpy.tile(-.5*numpy.pi,T), numpy.tile(L_HT[3],T), motor[i,0:T,1] +    numpy.pi],
                 [numpy.tile(     .0,T), numpy.tile(-.5*numpy.pi,T), numpy.tile(     .0,T), motor[i,0:T,2] - .5*numpy.pi],
                 [numpy.tile(L_HT[4],T), numpy.tile(          .0,T), numpy.tile(     .0,T), motor[i,0:T,3]              ],
                 [numpy.tile(L_HT[5],T), numpy.tile(          .0,T), numpy.tile(     .0,T),    numpy.tile(         .0,T)],
                 [numpy.tile( tipLLt,T), numpy.tile(          .0,T), numpy.tile(     .0,T),    numpy.tile(     tipTht,T)],
             ])

        tipLLi = fingertipShapeTable["LinkLength"][contactCenterPos["Index"]["Y"][i]]
        tipThi = fingertipShapeTable["Theta"][contactCenterPos["Index"]["Y"][i]]
        lpi = numpy.array([
                 [numpy.tile(L_HI[0],T), numpy.tile( .5*numpy.pi,T), numpy.tile(L_HI[1],T),                  numpy.tile(-.5*numpy.pi,T)],
                 [numpy.tile(     .0,T), numpy.tile( .5*numpy.pi,T), numpy.tile(     .0,T), motor[i,0:T,4] - psv[i,0:T,0] + .5*numpy.pi],
                 [numpy.tile(     .0,T), numpy.tile(-.5*numpy.pi,T), numpy.tile(     .0,T), motor[i,0:T,5]                             ],
                 [numpy.tile(L_HI[2],T), numpy.tile( .5*numpy.pi,T), numpy.tile(     .0,T), motor[i,0:T,6]                             ],
                 [numpy.tile(L_HI[3],T), numpy.tile(          .0,T), numpy.tile(     .0,T), motor[i,0:T,6] - psv[i,0:T,1]              ],
                 [numpy.tile(L_HI[4],T), numpy.tile(          .0,T), numpy.tile(     .0,T),                  numpy.tile(          .0,T)],
                 [numpy.tile( tipLLi,T), numpy.tile(          .0,T), numpy.tile(     .0,T),                   numpy.tile(     tipThi,T)],
             ])
        linkParamT.append(lpt)
        linkParamI.append(lpi)
    LinkParam = {"Thumb": numpy.array(linkParamT), "Index": numpy.array(linkParamI)}
    
    LN = LinkParam["Thumb"].shape[1]
    
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
            self.Thumb = numpy.zeros([N,LN,16,T])
            self.Index = numpy.zeros([N,LN,16,T])
    HTMatrix = CHTMatrix()
        
    for i in xrange(N):
        for j in xrange(LN):
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
        for j in xrange(LN-1):
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
            
    # 指先中心(爪固定ネジ原点)を基準とした指先間距離計算(従来)
#     fingertipCenterPos = {
#                             "Thumb": CalcHTM.Thumb[:,5,3:13:4],
#                             "Index": CalcHTM.Index[:,5,3:13:4],
#                          }
#     
#     fingertipDistance = numpy.sqrt((fingertipCenterPos["Thumb"][:,0] - fingertipCenterPos["Index"][:,0]) ** 2 +
#                                    (fingertipCenterPos["Thumb"][:,1] - fingertipCenterPos["Index"][:,1]) ** 2 +
#                                    (fingertipCenterPos["Thumb"][:,2] - fingertipCenterPos["Index"][:,2]) ** 2)
#     objectSize_old = fingertipDistance[:,0]
#     print "    従来:",objectSize_old
#     print "平均:", numpy.mean(objectSize_old), "分散:", numpy.var(objectSize_old)

    # 指先形状と接触位置を考慮した指先接触位置間距離計算
    contactCenterPos = {
                            "Thumb": CalcHTM.Thumb[:,6,3:13:4],
                            "Index": CalcHTM.Index[:,6,3:13:4],
                         }
     
    contactPosDistance = numpy.sqrt((contactCenterPos["Thumb"][:,0] - contactCenterPos["Index"][:,0]) ** 2 +
                                    (contactCenterPos["Thumb"][:,1] - contactCenterPos["Index"][:,1]) ** 2 +
                                    (contactCenterPos["Thumb"][:,2] - contactCenterPos["Index"][:,2]) ** 2)

    # 時刻0の指先接触位置間距離を把持物体のサイズとして認識する
    objectSize = contactPosDistance[:,0]
#     print "近似計算:",objectSize
#     print "平均:", numpy.mean(objectSize), "分散:", numpy.var(objectSize)
    print "Object Size: ", objectSize, "[mm]"


    # 学習用に整形
    objectSize = numpy.tile(objectSize[:,numpy.newaxis], handlingData.shape[1])
    objectSize = objectSize[:,:,numpy.newaxis]
    
    return objectSize


def LoadHandlingData(loadDirs=LEARNING_FILE_DIR):
    print('------------------------------')
    print('| Load Handling Data...      |')
    print('------------------------------')

    HD = []
    for loadDir in loadDirs:
        handlingData = LoadFile(loadDir)

        # タクタイル圧力情報から接触中心位置を推定
        contactCenterPos = EstimateContactCellPosition(handlingData);
#         print contactCenterPos
        
        # 指先形状テーブルの取得
        fingertipShapeTable = GetFingertipShapeTable()

        # 接触中心位置情報を基に指先形状テーブルを参照
        
        
        print('------------------------------')
        print('| Calculate Object Size...   |')
        print('------------------------------')
        objectSize = CalculateObjectSize(handlingData, contactCenterPos, fingertipShapeTable)
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

def PrepLearningData(HandlingData, trainType=['MOTOR', 'SIXAXIS', 'PSV'], teacherType=['MOTOR']):
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
    lb_train = []
    lb_teacher = []
    for type in trainType:
        lb_train += HandlingData.RANGE[type]
    for type in teacherType:
        lb_teacher += HandlingData.RANGE[type]
    
    train = train[:, :, lb_train]
    teacher = teacher[:, :, lb_teacher]
    
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
#    for i in xrange(5):
#        print numpy.min(handlingData.data[i][:,:,95])
#        print numpy.max(handlingData.data[i][:,:,95])

#     print handlingData.data
#    sparse = ShorteningTimeStep(handlingData)
#    plt.subplot(2,1,1)
    plt.plot(handlingData.data[0][6,:,1:])
#    plt.subplot(2,1,2)
#    plt.plot(sparse.data[0][0,:,1:])
    plt.show()

#     train, teacher = PrepLearningData(handlingData, ['MOTOR','SIXAXIS','PSV','TACTILE'])
#     plt.subplot(2,1,1)
#     plt.plot(handlingData.data[0][0,:,1:])
#     plt.subplot(2,1,2)
#     plt.plot(train[0,:,1:])
#     plt.show()

#    train, teacher = ReshapeForRNN_minibatch(train, teacher)
#    plt.plot(train[:,1,1:])
#    plt.show()

#    SaveScalledHandlingData(['../../AnalysisData/D60/Failure'], failureTrial=False)

