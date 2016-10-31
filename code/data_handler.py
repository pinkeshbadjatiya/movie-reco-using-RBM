#! /usr/bin/env python

import numpy as np
from get_trust import trust
from scipy.io import loadmat
import pdb

class data_handler():
    
    def __init__(self):
        self.n_users = 0
        self.n_prod = 0
        self.n_cat = 1
    
    def get_stats(self):
        return self.n_users, self.n_prod, self.n_cat

    def load_matrices(self):
        # Loading Matrices from data
        users, W = trust()
        # Converting R and W from dictionary to array
        R = []
        for user in users:
            for (m, r) in users[user]:
                R.append([user, m, r, 0])
        R = np.asarray(R)
        pdb.set_trace()
        self.n_users = max(R[:, 0])
        self.n_prod = max(R[:, 1])
        # Selecting entries with the 6 categories given in the paper
        cat_id = [0]
        R_size = R.shape[0]
        # Choosing 70% data for training and rest for testing
        R_train = R[:R_size*0.7]
        R_test = R[R_size*0.7:]
        # Making all eligible Product-Category pairs
        ones = np.ones(R_train.shape[0])
        prod_cat = dict(zip(zip(R_train[:, 1], R_train[:, 3]), ones))
        # Making the mu matrix
        mu = np.zeros(1)
        cat_rating = R_train[:, 2]
        mu[0] = np.mean(cat_rating)
        pdb.set_trace()
        return R_train, R_test, W, prod_cat, mu
            
if __name__ == "__main__":
    data = data_handler("../data/rating_with_timestamp.mat", "../data/trust.mat")
    R_train, R_test, W, PF_pair, mu = data.load_matrices()
    print "done"
