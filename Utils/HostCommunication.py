# -*- coding: utf-8 -*-

import socket
import threading
import time
import struct
import re
import collections
import numpy as np

#############################################################
## Define Parameter                                         #
#############################################################
HOST = "192.168.0.83"       # HOST-subのアドレス
RECVPORT = 7061             # 受信通信ポート番号
SENDPORT = 7060             # 送信通信ポート番号
PACKET_BUFFER_SIZE = 4096   # 受信パケットのバッファサイズ(十分な値を確保しておく)

# E_COMMAND
E_COMMAND = {
             "COMMAND_SUSPEND"      : 0,
             "COMMAND_OPERATE_RENEW": 1,
             "COMMAND_STANDBY"      : 2,
             "COMMAND_OPERATING"    : 3,
             "COMMAND_COMPLETE"     : 4,
            }

# 受信データのフォーマット指定
#    "データの種類名":"データサイズ(配列サイズ) データ型" で指定
RECV_DATA_FORMAT = (
                    ("e_Command"     , "1i"  ),
                    ("Motor"         , "16h" ),
                    ("Psv"           , "8h"  ),
                    ("Force"         , "24h" ),
                    ("Tactile"       , "242h"),
                    ("FileName"      , "20c" ),
                   )
RECV_DATA_FORMAT = collections.OrderedDict(RECV_DATA_FORMAT)

# 送信データのフォーマット指定
SEND_DATA_FORMAT = (
                    ("e_Command"     , "1i"  ),
                    ("Motor"         , "16h" ),
                    ("LogEnable"     , "1i"  ),
                   )
SEND_DATA_FORMAT = collections.OrderedDict(SEND_DATA_FORMAT)
#############################################################
    
class HostCommunication(threading.Thread):
## Hostとの通信を行うクラス(Thread動作)
    def __init__(self):
        # 送受信データの実体
        self.CUR_TWHAND = {}    # 受信データ(Hostから送られてきたデータはスレッド動作中ここに格納される)
        self.REF_TWHAND = {}    # 送信データ(Hostに送るデータはスレッド動作中ここが参照される)
        
        # 送受信データの初期化
        for key in RECV_DATA_FORMAT.keys():
            extract = re.match("\d*", RECV_DATA_FORMAT[key])    # フォーマットから数字を抜き出す(ex. "16h"なら"16"が取り出される)
            dim = int(extract.group())
            self.CUR_TWHAND[key] = np.zeros(dim)
        for key in SEND_DATA_FORMAT.keys():
            extract = re.match("\d*", SEND_DATA_FORMAT[key])    # フォーマットから数字を抜き出す(ex. "16h"なら"16"が取り出される)
            dim = int(extract.group())
            self.REF_TWHAND[key] = np.zeros(dim)

        # プライベート変数
        self.bufSize = PACKET_BUFFER_SIZE
        self.RecvSock = []
        self.SendSock = []
        self.CompleteFlag = False

        ## 受信データのバイナリを解釈して配列化するUnpackerを作成
        fmt = ""
        for key in RECV_DATA_FORMAT.keys():
            fmt = fmt + RECV_DATA_FORMAT[key]
        self.Unpacker = struct.Struct(fmt)

        ## 送信する辞書データをバイナリ化するPackerを作成
        fmt = ""
        for key in SEND_DATA_FORMAT.keys():
            fmt = fmt + SEND_DATA_FORMAT[key]
        self.Packer = struct.Struct(fmt)

        # スレッド動作のための初期化
        super(HostCommunication, self).__init__()
    
    def run(self):
        ## スレッド動作部(Communication.start()で動作が開始される)
        ## ここでは、HostAppが終了するまでデータの送受信を行い続ける
        # ソケット接続
        self.SocketConnect()
        
        while True:
            # データ受信
            data = self.ReadSocketData()
            # HostAppの終了に伴って終了する
            if not data:
                break
            self.CUR_TWHAND = data
        
            # データ送信
            self.SendSocketData(self.REF_TWHAND)
        
        # ソケット切断
        self.SocketClose()
        
        self.CompleteFlag = True

    def Decode(self, recvdata):
        ## 受信したバイナリ化された構造体の解釈
        # 受信データを配列に格納
        udata = self.Unpacker.unpack(recvdata)

        # 受信データを適切に解釈して辞書に格納
        head = 0
        decodedData = {}
        for key in RECV_DATA_FORMAT.keys():
            extract = re.match("\d*", RECV_DATA_FORMAT[key])    # フォーマットから数字を抜き出す(ex. "16h"なら"16"が取り出される)
            tail = int(extract.group())
            decodedData[key] = udata[head:head+tail]
            head = head + tail
        
        return decodedData

    def Encode(self, senddata):
        ## 送信する辞書データをバイナリ化する
        # 送信データを適切に解釈して辞書に格納
        val = []
        for key in SEND_DATA_FORMAT.keys():
            # 引数として与えられた辞書データがフォーマットと違う場合Null配列を返す
            if not senddata.has_key(key):
                print "Send Data Format Error!"
                return []
            val.extend(senddata[key])
        # 送信データをバイナリ化
        encodedData = self.Packer.pack(*val)

        return encodedData

    def SocketConnect(self):
        ## ソケット通信初期化処理
        # ソケットのインスタンスを用意
        self.RecvSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.SendSock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # ソケットを開く
        print "Connecting Socket..."
        self.RecvSock.connect((HOST, RECVPORT))
        self.SendSock.connect((HOST, SENDPORT))
        print "Success!"
        
    def SocketClose(self):
        ## ソケットを閉じる
        print "Closing Socket..."
        self.RecvSock.close()
        self.SendSock.close()
        print "Success!"
        
    def ReadSocketData(self):
        ## データを受信し適切に解釈して格納する
        # バイナリ受信
        data = self.RecvSock.recv(self.bufSize)
        if not data:
            return []
        
        # 受信したバイナリ化された構造体の解釈
        recv = self.Decode(data)

        return recv
    
    def SendSocketData(self, senddata):
        ## データ送信
        send = self.Encode(senddata)
        self.SendSock.send(send)
        
    def SetSendParam(self, motor):
        ## Host送信用辞書に目標角度をセット(拇指・示指のみ)
        self.REF_TWHAND["e_Command"] = [E_COMMAND["COMMAND_OPERATE_RENEW"]]
        
        if len(motor) == 7:
            self.REF_TWHAND["Motor"][0:7] = motor
        else:
            print "[SetSendParam]:Invalid Parameter!"
            
    def LogEnable(self):
        self.REF_TWHAND["LogEnable"] = [1]
            
    def LogDisable(self):
        self.REF_TWHAND["LogEnable"] = [0]
        
    def GetFileName(self):
        return "".join(map(str, self.CUR_TWHAND["FileName"]))
    
    def PrintRecvData(self):
        ## 受信しているデータの現在値を表示
#         print "[e_Command]" , self.CUR_TWHAND["e_Command"]
        print "[Motor]:"     , self.CUR_TWHAND["Motor"][0:8]
        print "[Psv]  :"     , self.CUR_TWHAND["Psv"][2:4]
        print "[Force]:"     , self.CUR_TWHAND["Force"][0:12]
#         print "[Tactile]"   , self.CUR_TWHAND["Tactile"]
        print "[FileName]:"  , self.GetFileName()


        
if __name__ == '__main__':
    com = HostCommunication()
    
    # 通信動作スレッド開始
    com.start()
#     print com.REF_TWHAND
     
    c = 0
    while(True):
        if com.CompleteFlag == True:
            break
        time.sleep(0.5)
        com.LogEnable()
        time.sleep(0.5)
        com.LogDisable()
#         com.SetSendParam([c, c, c, c, 0, 0, c])
#        c=c+10
#         if c > 100:
#             break
 
#         print com.REF_TWHAND
        com.PrintRecvData()
    
    print "END"
