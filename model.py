#! /usr/bin/env python
from __future__ import division
import numpy as np
import pdb
from data import get_data
import copy

N_IT = 200
ETA = 0.001

users = {}
user_movies = {}

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RBM():

    def __init__(self, data):

        self.F = 100
        self.K = 5
        self.m = 0
        #self.m = data.shape[0]   # No of movies rated by user
        for i, u in enumerate(users):
            temp = [movie_id for (movie_id, rating) in raw_data[u]]
            self.m = max(self.m, max(i for i in temp) + 1)
        print self.m
        self.h = np.random.rand(self.F) - 0.5
        self.featureBias = np.random.rand(self.F) - 0.5
        self.movieBias = np.random.rand(self.m, self.K) - 0.5
        self.w = np.random.rand(self.F, self.m, self.K) - 0.5
        self.data = data


    def train(self):
        for it in range(N_IT):
            for u in users:
                data = copy.deepcopy(self.data[u])
                w = self.getW(user_movies[u])
                posAssociations, self.h = self.fwdProp(data, user_movies[u])
                visibleProb = self.bwdProp(self.h, user_movies[u])
                negAssociations, temp = self.fwdProp(visibleProb, user_movies[u])
                w += ETA * (posAssociations - negAssociations) / len(user_movies[u]) #might change len
                self.setW(user_movies[u], w)
                error = np.sum((data - visibleProb) ** 2)
                print error


    def getW(self, movies):

        a = np.zeros((self.F, 1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.w[:,m,:], axis=1)), axis=1)
        return a[:,1:,]


    def setW(self, movies, w):
        
        it = 0
        for m in movies:
            self.w[:, m, :] += w[:, it, :]
            it += 1
        
    
    def getMovieBias(self, movies):
        
        a = np.zeros((1, self.K))
        for m in movies:
            a = np.concatenate((a, np.expand_dims(self.movieBias[m,:], axis=0)), axis=0)
        return a[1:,]


    def fwdProp(self, inp, movies):
        hiddenUnit = np.copy(self.featureBias)
        for j in range(self.F):
            hiddenUnit[j] += np.tensordot(inp, self.getW(movies)[j])
        hiddenProb = sigmoid(hiddenUnit)
        hiddenStates = hiddenProb > np.random.rand(self.F)
        hiddenAssociations = np.zeros((self.F, len(movies), self.K))    # Same as self.w for a single user case
        for j in range(self.F):
            hiddenAssociations[j] = hiddenProb[j] * inp
        return hiddenAssociations, hiddenStates


    def bwdProp(self, inp, movies):
        visibleUnit = self.getMovieBias(movies)
        for j in range(self.F):
            visibleUnit += inp[j] * self.getW(movies)[j]
        visibleProb = sigmoid(visibleUnit)
        return visibleProb
        

def demo():
    data = [[0,0,1,0,0], [0,1,0,0,0], [0,0,0,0,1]]
    data = np.asarray(data)
    rbm = RBM(data)
    rbm.train()


if __name__ == '__main__':
    #demo()    
    
    raw_data = get_data()
    users = raw_data.keys()
    users.sort()
    data = {}
    for i, u in enumerate(users):
        user_movies[u] = [movie_id for (movie_id, rating) in raw_data[u]]
        data[u] = [[0]*(rat-1) + [1] + [0]*(5-rat) for (mov_id, rat) in raw_data[u]]
        data[u] = np.asarray(data[u])
    #rbm = RBM(data[data.keys()])
    rbm = RBM(data)
    rbm.train()
