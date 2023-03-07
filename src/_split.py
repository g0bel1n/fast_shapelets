import math 
import numpy as np

def get_entropy(y):
            # y binaire 0 or 1
            p = sum(y)/y.shape[0]
            if p==1 or p==0:
                return 0
            return p*math.log(p) + (1-p)*math.log(1-p)

class Split():
    def __init__(self,split_info,shapelet):
        self.shapelet = shapelet
        self.Lside = np.where(split_info==0)[0]
        self.Rside = np.where(split_info==1)[0]
        self.nL = self.Lside.shape[0]
        self.nR = self.Rside.shape[0]
    
    def info_gain(self,y):
        N = y.shape[0]
        yL = y[self.Lside]
        yR = y[self.Rside]

        if self.nL == 0 or self.nR  ==0:
            self.gain = -np.inf
            return self.gain
        else:
            self.gain = get_entropy(y) - (self.nL/N)*get_entropy(yL) -  (self.nR/N)*get_entropy(yR)
            return self.gain
    
    def gap(self,Dlist):
        if self.nL ==0 or self.nR  ==0:
            self.gap = 0
        else:
            self.gap = np.sum(Dlist[self.Lside])/self.nL - np.sum(Dlist[self.Rside])/self.nR
        return self.gap

        