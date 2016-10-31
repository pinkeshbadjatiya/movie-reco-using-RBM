from __future__ import division
from os import listdir
from os.path import isfile, join
from data import get_data
import numpy as np

def trust():
    
    print "getting trust"
    users = get_data()
    cutoff = 0.03
    
    trust_mat = []

    print len(users.keys())
    for user1 in users:
        for user2 in users:

            if user1 == user2:
                continue
            print user1, user2
            cnt = 0
            mo = {}

            for (m1,r1) in users[user1]:
                mo[m1] = 1
            for (m2,r2) in users[user2]:
                if mo.has_key(m2):
                    cnt +=1

            if cnt/len(users[user1]) >= cutoff and cnt/len(users[user2])>=cutoff:
                trust_mat.append([user1, user2])
    print "trust got" 
    return users, np.asarray(trust_mat)


if __name__ == "__main__":
    trust_mat = trust(users)


    import pdb
    pdb.set_trace()
    
