# -*- coding: utf-8 -*-

import numpy as np
from DataIO import LoadHandlingData

LEARNING_FILE_DIR = ['../../AnalysisData/debug']
#===============================================================================
# Hand Parameter
#===============================================================================
# Hand Link Parameter for Calculating Forward Kinematics
L_HT = np.array([21.4, 0.05, 14.0, 39.6, 39.5, 29.7])     # Thumb link length
L_HI = np.array([35.0, 70,7, 50.0, 32.0, 27.0])           # Index finger link length


#===============================================================================
# Methods
#===============================================================================
def CalculateForwardKinematics(motor):
    linkParam = {
                     "Thumb":[
                              [L_HT[1], np.pi/2, 0, -np.pi/2],
                              [L_HT[2],   np.pi, 0, -np.pi/2],
                              [L_HT[3], np.pi/2, 0, -np.pi/2],
                              [L_HT[1], np.pi/2, 0, -np.pi/2],
                     ]
                }
    
    return linkParam
    
    
if __name__ == '__main__':
    handlingData = LoadHandlingData(LEARNING_FILE_DIR)
    fw = CalculateForwardKinematics(handlingData[:,:,range(1,8)])