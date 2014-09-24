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

RANGE = {
         "TIME"     : range(0,1),
         "MOTOR"    : range(1,8),
         "SIXAXIS"  : range(9,21),
         "PSV"      : range(21,23),
         "TACTILE"  : range(23,95),
         "SIZE"  : range(95,96),
         }

# 物体サイズは初めに一度だけ計算させたいため、それを管理するフラグ
IS_GET_OBJECT_SIZE = False
OBJECT_SIZE = np.inf

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

    time = 0    # Dummy
    motor   = np.array(data["Motor"][0:8])
    psv     = np.array(data["Psv"][2:4])
    force   = np.array(data["Force"][0:12])
    tactile = np.r_[data["Tactile"][0:36], data["Tactile"][72:108]]
    
    # handlingData形式に変換
    handlingData = np.array([np.r_[time, motor, psv, force, tactile]])
    handlingData = handlingData[np.newaxis, :]
    
    if IS_GET_OBJECT_SIZE is False:
    # タクタイル圧力情報から接触中心位置を推定
        contactCenterPos = DataIO.EstimateContactCellPosition(handlingData);
        
        # 指先形状テーブルの取得
        fingertipShapeTable = DataIO.GetFingertipShapeTable()
    
        # 関節角度、接触中心位置情報、指先形状テーブルを利用して物体サイズを計算
        objectSize = DataIO.CalculateObjectSize(handlingData, contactCenterPos, fingertipShapeTable)
        
        # 値を保持
        OBJECT_SIZE = objectSize
        IS_GET_OBJECT_SIZE = True
    else:
        objectSize = OBJECT_SIZE

    # 物体サイズを行列に付加
    handlingData = np.c_[handlingData, objectSize]
    
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
    time    = 0                     # Dummy
    motor   = np.r_[motorData[0], 0]
    psv     = np.zeros(2)           # Dummy
    force   = np.zeros(12)          # Dummy
    tactile = np.zeros(72)          # Dummy
    size    = np.zeros(1)           # Dummy
    
    # handlingData形式に変換
    handlingData = np.array([np.r_[time, motor, psv, force, tactile, size]])
    handlingData = handlingData[np.newaxis, :]

    # 元のスケールに戻す
    handlingData = DataIO.ReScalingHandlingData(handlingData)

    # int型にキャストし整形
    reshapeData = map(int, handlingData[0][0][1:8])

    return reshapeData


#===============================================================================
# Main
#===============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 学習済みネットワークを呼び出す
    model = PrepNetwork('NN_DALL_MSPTS_Short')
    

    # Hostと通信を行うインスタンスを用意
    com = HostCommunication()
    
#     # For Debug
#     handlingData = DataIO.LoadFile('../../AnalysisData/debug')
#     Data = handlingData[2,0,:]
#     
#     dummy = {
#              "Motor":   np.r_[Data[1:9], np.zeros(16)],
#              "Psv":     np.r_[Data[21:23], np.zeros(6)],
#              "Force":   np.r_[Data[9:21], np.zeros(12)],
#              "Tactile": np.r_[Data[23:59], np.zeros(36), Data[59:95], np.zeros(146)],
#             }
#     inputData = ReshapeRecvDataForNetwork(dummy, inputType=['MOTOR', 'SIXAXIS', 'PSV', 'TACTILE', 'SIZE'])
#     outputData = model.predict(inputData)
#     nextMotor = ReshapeOutputDataForSendToHost(outputData)
#     com.SetSendParam(nextMotor)
#     print "REF_MOTOR:", com.REF_TWHAND["Motor"][0:7]

    # Hostとの通信を行うスレッドを開始する
#     com.start()
     
    keyMonitoringThread = Threads.KeyMonitoringThread()
    keyMonitoringThread.start()
    startFlag = False
    print "Please input 'S' to Start Control"
    print "Please input 'Q' to Stop  Control"
    while True:
        # コマンド入力を別スレッドで受け取る
        var = keyMonitoringThread.GetInput()

        if var == 's':
            startFlag = True
        if var == 'p':
            startFlag = False
        if var == 'q':
            startFlag = False
            break

        if startFlag is True:
            # Hostとの通信が切れるまでループ
            if com.CompleteFlag == True:
                break
             
            # Hostから受信した現在のハンドパラメータをネットワークへ入力できる形に整形
            inputData = ReshapeRecvDataForNetwork(com.CUR_TWHAND)
             
            # 学習済みのネットワークにハンドパラメータの現在値を入力し(t+1)の出力を得る
            outputData = model.predict(inputData)
             
            # ネットワーク出力をホストに送信できる形に整形
            nextMotor = ReshapeOutputDataForSendToHost(outputData)
            print "NEXT_MOTOR:", nextMotor
             
            # ネットワークの出力をホストに送信し、ハンド制御を行う
#             com.SetSendParam(nextMotor)
#             print "REF_MOTOR:", com.REF_TWHAND["Motor"][0:7]
         
        # Wait 500mSec
        time.sleep(0.5)
    
    keyMonitoringThread.Stop()
    print "End of Real Time Control"
