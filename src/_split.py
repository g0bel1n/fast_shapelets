import math 
import numpy as np

def get_entropy(y):
            p = np.array([np.sum(y==cls)/y.shape[0] for cls in np.unique(y)])
            p = p[p>0]
            return np.sum(p*np.log2(p))
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
        else:
            self.gain = get_entropy(y) - (self.nL/N)*get_entropy(yL) -  (self.nR/N)*get_entropy(yR)

        return self.gain

    def gap(self,Dlist):
        if self.nL ==0 or self.nR  ==0:
            self.gap = 0
        else:
            self.gap = np.sum(Dlist[self.Lside])/self.nL - np.sum(Dlist[self.Rside])/self.nR
        return self.gap

        