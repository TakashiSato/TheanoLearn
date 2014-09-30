# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
import theano
import matplotlib.pyplot as plt

import sys
sys.path.append('../Utils')
import DataIO 
import Threads
from NN import MetaNN
from HostCommunication import HostCommunication

logger = logging.getLogger(__name__)

#===============================================================================
# Parameters
#===============================================================================
RANGE = {
         "TIME"     : range(0,1),
         "MOTOR"    : range(1,8),
         "SIXAXIS"  : range(9,21),
         "PSV"      : range(21,23),
         "TACTILE"  : range(23,95),
         "SIZE"     : range(95,96),
         }

# 物体サイズは初めに一度だけ計算させたいため、それを管理するフラグ
IS_GET_OBJECT_SIZE = False

# Global
OBJECT_SIZE = 40
CONTROL_TIME = 0

CONTROL_TIME_MAX = 60000   # msec
CONTROL_INTERVAL = 50      # msec

#===============================================================================
# Methods
#===============================================================================
def PrepNetwork(NAME=''):
    # ネットワークの構築
    model = MetaNN()
    
    # 学習済みのネットワークパラメータを読み込む
    loadParam = './models/' + NAME + '/' + NAME + '.pkl'
    model.load(loadParam)

    return model

def ReshapeRecvDataForNetwork(data, inputType=['MOTOR', 'SIXAXIS', 'PSV', 'TACTILE']):
    # グローバル変数の参照のための設定
    global IS_GET_OBJECT_SIZE
    global OBJECT_SIZE

    Time = 0    # Dummy
    motor   = np.array(data["Motor"][0:8])
    psv     = np.array(data["Psv"][2:4])
    force   = np.array(data["Force"][0:12])
    tactile = np.r_[data["Tactile"][0:36], data["Tactile"][72:108]]
    
    # handlingData形式に変換
    handlingData = np.array([np.r_[Time, motor, psv, force, tactile]])
    handlingData = handlingData[np.newaxis, :]

    if (IS_GET_OBJECT_SIZE is False) and ('SIZE' in inputType):
    # タクタイル圧力情報から接触中心位置を推定
        contactCenterPos, contactCell = DataIO.EstimateContactCellPosition(handlingData);
         
        # 関節角度、接触中心位置情報、指先形状テーブルを利用して物体サイズを計算
        objectSize_withTac, objectSize_ftd = DataIO.CalculateObjectSize(handlingData, contactCenterPos)
         
        # 値を保持
        objectSize = objectSize_ftd[0,0,0]
        print "Recognized Object Size: ", objectSize
        time.sleep(1)
        OBJECT_SIZE = objectSize
        IS_GET_OBJECT_SIZE = True
    else:
        objectSize = OBJECT_SIZE
#     objectSize = 0

    # handlingData形式に変換
    handlingData = np.array([np.r_[Time, motor, psv, force, tactile, objectSize]])
    handlingData = handlingData[np.newaxis, :]
    
    # リミットチェック
    DataIO.CheckLimit(handlingData)

    # -1~+1にスケーリング
    handlingData = DataIO.ScalingHandlingData(handlingData)

    # 整形
    lb = []
    for type in inputType:
        lb += RANGE[type]
    handlingData = handlingData[:, :, lb]

    reshapeData = np.array(handlingData[0], dtype=theano.config.floatX)
    
    return reshapeData

def ReshapeOutputDataForSendToHost(motorData):
    Time    = 0                     # Dummy
    motor   = np.r_[motorData[0], 0]
    psv     = np.zeros(2)           # Dummy
    force   = np.zeros(12)          # Dummy
    tactile = np.zeros(72)          # Dummy
    size    = np.zeros(1)           # Dummy
    
    # handlingData形式に変換
    handlingData = np.array([np.r_[Time, motor, psv, force, tactile, size]])
    handlingData = handlingData[np.newaxis, :]

    # 元のスケールに戻す
    handlingData = DataIO.ReScalingHandlingData(handlingData)

    # int型にキャストし整形
    reshapeData = map(int, handlingData[0][0][1:8])

    return reshapeData

def SetInitialPose(com, div=10):
    initMotor = np.array([10000, 9000, 3500, 400, 6900, 0, 3200])

    print "!!!!! Initialize Motor after 1 sec !!!!!"
    time.sleep(1)
    for i in xrange(div):
        ref = initMotor/div * (i+1)
        com.SetSendParam(ref)
        print "[",i,"] Next Motor:", ref
#         time.sleep(0.5)
        time.sleep(CONTROL_INTERVAL/1000.0)
    print "Complete!!!"
    
def ControlEndCheck():
    global CONTROL_TIME
    
    if ((CONTROL_TIME) % 1000) == 0:
        print "CONTROL_TIME:", CONTROL_TIME

    CONTROL_TIME = CONTROL_TIME + CONTROL_INTERVAL
    if CONTROL_TIME >= CONTROL_TIME_MAX:
        CONTROL_TIME = 0
        return True
    else:
        return False

#===============================================================================
# Main
#===============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    global IS_GET_OBJECT_SIZE
    global CONTROL_TIME

    # 学習済みネットワークを呼び出す
    model = PrepNetwork('NN_D204060_MSPT_Short_batch100_f')
    
    # Hostと通信を行うインスタンスを用意
    com = HostCommunication()
    
    # Hostとの通信を行うスレッドを開始する
    com.start()
     
    keyMonitoringThread = Threads.KeyMonitoringThread()
    keyMonitoringThread.start()
    rtcFlag = False
    print "####################################"
    print "# Command                          #"
    print "####################################"
    print "#  'i': Set Initial Grasping Pose  #"
    print "#  's': Start Real Time Control    #"
    print "#  'p': Pause Real Time Control    #"
    print "#  'q': Quit Program               #"
    print "####################################"
    while True:
        # Hostとの通信が切れるまでループ
        if com.CompleteFlag == True:
            break

        # コマンド入力を別スレッドで受け取る
        var = keyMonitoringThread.GetInput()

        # 制御始めは初期把持姿勢へ移行
        if var == 'i':
            SetInitialPose(com, div=50)
            CONTROL_TIME = 0
        # RealTimeControl開始
        if var == 's':
            print "!!! Waiting 1 sec for Real Time Control!!!"
            time.sleep(1)
            print "!!! Real Time Control Start !!!"
            com.LogEnable()
            IS_GET_OBJECT_SIZE = False
            rtcFlag = True
        # RealTimeControl一時停止
        if var == 'p':
            com.LogDisable()
            print "!!! Real Time Control Pause !!!"
            rtcFlag = False
        # プログラム終了
        if var == 'q':
            com.LogDisable()
            print "!!! Real Time Control Stop !!!"
            rtcFlag = False
            break

        # Real Time Control部分
        if rtcFlag is True:
            # Hostから受信した現在のハンドパラメータをネットワークへ入力できる形に整形
            inputData = ReshapeRecvDataForNetwork(com.CUR_TWHAND, inputType=['MOTOR', 'SIXAXIS', 'PSV', 'TACTILE'])#, 'SIZE'])
             
            # 学習済みのネットワークにハンドパラメータの現在値を入力し(t+1)の出力を得る
            outputData = model.predict(inputData)
             
            # ネットワーク出力をホストに送信できる形に整形
            nextMotor = ReshapeOutputDataForSendToHost(outputData)
#             print "NEXT_MOTOR:", nextMotor
            
            # ネットワークの出力をホストに送信し、ハンド制御を行う
            com.SetSendParam(nextMotor)
#             print "REF_MOTOR:", com.REF_TWHAND["Motor"][0:7]

            # 制御終了処理
            if ControlEndCheck() == True:
                com.LogDisable()
                print "!!! Real Time Control Stop !!!"
                rtcFlag = False
         
        # 受信データの表示
#         com.PrintRecvData()
        
        # Wait
        time.sleep(CONTROL_INTERVAL / 1000.0)
    
    keyMonitoringThread.Stop()
    print "End of Real Time Control"
