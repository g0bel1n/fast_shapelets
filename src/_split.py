import math 
import numpy as np

class Split():
    def __init__(self,split_info,shapelet):
        self.shapelet = shapelet
        self.Lside = np.where(split_info==0)
        self.Rside = np.where(split_info==1)
        self.nL = self.Lside.shape[0]
        self.nR = self.Rside.shape[0]
    
    def info_gain(self,y):
        def get_entropy(y):
        # y binaire 0 or 1
            p = sum(y)/y.shape[0]
            return p*math.log(p) + (1-p)*math.log(1-p)
        N = y.shape[0]
        yL = y[self.Lside]
        yR = y[self.Rside]

        self.gain = get_entropy(y) - (self.nL/N)*get_entropy(yL) -  (self.nR/N)*get_entropy(yR)
        return self.entropy
    
    def gap(self,Dlist):
        if self.nL ==0 or self.nR  ==0:
            self.gap = 0
        else:
            self.gap = np.sum(Dlist[self.Lside])/self.nL - np.sum(Dlist[self.Rside])/self.nR
        return self.gap

        