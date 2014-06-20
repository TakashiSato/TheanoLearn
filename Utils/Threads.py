# -*- coding: utf-8 -*-

import threading
import time

class KeyMonitoringThread(threading.Thread):
    def __init__(self):
        super(KeyMonitoringThread, self).__init__()
        self.var = []
        self.loopFlg = True
        
    def run(self):
        while(self.loopFlg is True):
            self.var = raw_input()
            time.sleep(.5)
            
        
    def GetInput(self):
        ret = self.var
        self.var = []
        return ret
    
    def Stop(self):
        self.loopFlg = False
        
if __name__ == '__main__':
    print 'Ha-jima-ruyo---!!'

    th = KeyMonitoringThread()
    th.start()
    
    f=0
    i=0
    while(i < 5 and f != 1):
        i += 1
        time.sleep(1)
        print i, 'kaime!'
        a = th.GetInput()
        if a == 'q':
            print 'OWARUYO!'
            th.Stop()
            f = 1

    print 'Owattayo!'
    th.Stop()
