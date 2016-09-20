#! /usr/bin/env python
from __future__ import division
import numpy as np
import pdb
from data import get_data

N_IT = 200
ETA = 0.01

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RBM():

    def __init__(self, data):

        self.F = 100
        self.K = 5
        self.m = 3
        self.h = np.random.rand(self.F)
        self.featureBias = np.random.rand(self.F)
        self.movieBias = np.random.rand(self.m, self.K)
        self.w = np.random.rand(self.F, self.m, self.K)
        self.data = data


    def train(self):

        for it in range(N_IT):
            posAssociations, self.h = self.fwdProp(self.data)
            visibleProb = self.bwdProp(self.h)
            negAssociations, temp = self.fwdProp(visibleProb)
            self.w += ETA * (posAssociations - negAssociations) / len(self.data) #might change len
            error = np.sum((self.data - visibleProb) ** 2)
            print error


    def fwdProp(self, inp):

        hiddenUnit = np.copy(self.featureBias)
        for j in range(self.F):
            hiddenUnit[j] += np.tensordot(inp, self.w[j])
        hiddenProb = sigmoid(hiddenUnit)
        hiddenStates = hiddenProb > np.random.rand(self.F)
        hiddenAssociations = np.zeros_like(self.w)
        for j in range(self.F):
            hiddenAssociations[j] = hiddenProb[j] * inp
        return hiddenAssociations, hiddenStates


    def bwdProp(self, inp):
        
        visibleUnit = np.copy(self.movieBias)
        for j in range(self.F):
            visibleUnit += inp[j] * self.w[j]
        visibleProb = sigmoid(visibleUnit)
        return visibleProb
        


if __name__ == '__main__':
    raw_data = get_data()
    all_users = raw_data.keys()
    all_users.sort()
    user_movies = {}

    data = {}
    for u in all_users:
        user_movies[u] = [movie_id for (movie_id, rating) in raw_data[u]]
        data[u] = [[0]*(rat-1) + [1] + [0]*(5-rat) for (mov_id, rat) in raw_data[u]]
        data[u] = np.asarray(data)
        break
    #data = [[0,0,1,0,0], [0,1,0,0,0], [0,0,0,0,1]]
    #data = np.asarray(data)
    rbm = RBM(data)
    rbm.train()


