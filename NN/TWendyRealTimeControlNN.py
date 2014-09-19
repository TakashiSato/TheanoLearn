# -*- coding: utf-8 -*-

import logging
import time
import numpy as np
import theano

import sys
sys.path.append('../Utils')
import DataIO 
from NN import MetaNN
from HostCommunication import HostCommunication

logger = logging.getLogger(__name__)

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

def ReshapeRecvDataForNetwork(data):
    time = 0    # Dummy
    motor   = np.array(data["Motor"][0:8])
    psv     = np.array(data["Psv"][2:4])
    force   = np.array(data["Force"][0:12])
    tactile = np.r_[data["Tactile"][0:36], data["Tactile"][72:108]]
    
    # handlingData形式に変換
    handlingData = np.array([np.r_[time, motor, psv, force, tactile]])
    handlingData = handlingData[np.newaxis, :]
    
    # タクタイル圧力情報から接触中心位置を推定
    contactCenterPos = DataIO.EstimateContactCellPosGition(handlingData);
    
    # 指先形状テーブルの取得
    fingertipShapeTable = DataIO.GetFingertipShapeTable()

    # 接触中心位置情報を基に指先形状テーブルを参照
    
    
    print('------------------------------')
    print('| Calculate Object Size...   |')
    print('------------------------------')
    objectSize = DataIO.CalculateObjectSize(handlingData, contactCenterPos, fingertipShapeTable)
    # 物体サイズを行列に付加
    handlingData = np.c_[handlingData, objectSize]
    
    print('------------------------------')
    print('| Check Limit...             |')
    print('------------------------------')
    DataIO.CheckLimit(handlingData)

    print('------------------------------')
    print('| Scaling Data...            |')
    print('------------------------------')
    handlingData = DataIO.ScalingHandlingData(handlingData)
    
    print handlingData
    

    size    = 0
    
    # -1~+1にスケーリング
    pass
    
#     reshapeData = np.r_[motor, psv, force, tactile, size]
    reshapeData = reshapeData[np.newaxis,:]
    reshapeData = np.array(reshapeData, dtype=theano.config.floatX)
    
    return reshapeData

def ReshapeOutputDataForSendToHost(data):
    # 元のスケールに戻す
    pass

    data = data*100000

    return map(int,data.reshape(7))


#===============================================================================
# Main
#===============================================================================
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 学習済みネットワークを呼び出す
    model = PrepNetwork('NN_20140919_MSPTS')
    
    # Hostと通信を行うインスタンスを用意
    com = HostCommunication()
    
    # Hostとの通信を行うスレッドを開始する
    com.start()
    
    print "Control Start!"
    while True:
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
#         com.SetSendParam(nextMotor)
#         print "REF_MOTOR:", com.REF_TWHAND["Motor"][0:7]
        
        # Wait 500mSec
        time.sleep(0.5)
    
    print "End of Real Time Control"
