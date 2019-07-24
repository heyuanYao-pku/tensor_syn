import syntensor
import numpy as np
import json
import os
import sys
np.set_printoptions(linewidth = np.nan)
np.set_printoptions(threshold=np.inf)
image_nums = 95
current_path = 'D:\\paper\\tensor syn\\wolfdata'
'''
fname = 'new_data_10.json'
sys.stdin = open(fname,'r')
data = input()
data = json.loads(data)
Plist = np.array(data['P'],np.ndarray)


mlist = np.array(data['mlist'])
#print(mlist)
#print(Plist[1][0])
'''

Plist = np.zeros([image_nums,image_nums],np.ndarray)
mlist = None
for i in range(1,5):
    print(i)
    f = open(os.path.join(current_path,"%d.json"%i) ,'r')
    j = json.load(f)
    loaded = 0
    if mlist is None:
        mlist = np.array(j['mlist'])

    else :
        loaded = len(mlist)
        mlist = np.hstack([mlist,np.array( j['mlist'] )] )
    l = np.size(mlist)
    for q in range(loaded,l):
        for t in range(loaded,image_nums):
            if type(j['P'][q][t]) == int:
                print(i,q,t)
            Plist[q][t] = j['P'][q][t]
            Plist[t][q] = j['P'][t][q]


n = len(mlist)

PPP = np.zeros([n,n],np.ndarray)
for i in range(n):
    for j in range(n):
        PPP[i][j] = Plist[j][i].copy()
Plist = PPP

tensor = syntensor.SynTensor(n,mlist,Plist)

print('object building finished')

Q = tensor.solution()
np.save(os.path.join(current_path,'156.npy'),Q,)
k = tensor.rounded_solution(0.5,2,1,Q )


print(np.shape(k))

print(k)

tensor.visualize(image_nums ,os.path.join( current_path,'image_list.npy' ), os.path.join(current_path,'tmp_%d'%image_nums) )
#f = open('D:\\paper\\tensor syn\\res.txt','w')
#sys.stdout = f
#print(np.float32(sol))
#print(np.float32(k))