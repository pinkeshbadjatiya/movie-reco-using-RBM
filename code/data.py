from os import listdir
from os.path import isfile, join


mypath = "../data/training_set/"
truncate = True
def get_data():
    files = [ f for f in listdir(mypath) if isfile(join(mypath, f)) ]
    
    users = {}

    ct = 0
    user_map = {}
    user_ct = 0
    for name in files:
        with open(mypath+''+name, 'r') as f:
            index = int(f.readline().split(':')[0])
    
            for line in f.readlines():
                [u, r] = line.split(',')[:2]
                u,r = int(u), int(r)
                if u not in user_map:
                    user_map[u] = user_ct
                    user_ct += 1
                u = user_map[u]
                if u in users:
                    users[u].append((index, r))
                else:
                    users[u] = []
                    users[u].append((index, r))
            ct += 1
            if ct%2:
                print ct
   
    temp = {}
    for i, user in enumerate(users):
        if truncate:
            if i >= 1000:
                break
        temp[user] = users[user]
    
    print "data got"
    return temp


if __name__ == "__main__":
    users = get_data()
    import pdb
    pdb.set_trace()
    
