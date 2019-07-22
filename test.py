import syntensor
import numpy as np
import json
import sys


fname = 'new_data_10.json'
sys.stdin = open(fname,'r')
data = input()
data = json.loads(data)
Plist = np.array(data['P'],np.ndarray)

mlist = np.array(data['mlist'])
#print(mlist)
#print(Plist[1][0])

'''
Plist = np.zeros([156,0],np.ndarray)
mlist = None
for i in range(1,6):
    print(i)
    f = open("D:/paper/tensor syn/data/%d.json"%i,'r')
    j = json.load(f)
    loaded = 0
    if mlist is None:
        mlist = np.array(j['mlist'])
        Plist = np.array(j['P'][loaded:len(mlist)], np.ndarray)
    else :
        loaded = len(mlist)
        mlist = np.hstack([mlist,np.array( j['mlist'] )] )
        Plist = np.vstack( [Plist,np.array( j['P'][loaded:len(mlist)] ,np.ndarray)] )

'''
n = len(mlist)

PPP = np.zeros([n,n],np.ndarray)
for i in range(n):
    for j in range(n):
        PPP[i][j] = Plist[j][i].copy()
Plist = PPP

tensor = syntensor.SynTensor(n,mlist,Plist)

print('object building finished')
k = tensor.rounded_solution()
print(np.shape(k))

f = open('D:\\paper\\tensor syn\\res.txt','w')
sys.stdout = f
np.set_printoptions(linewidth = np.nan)
np.set_printoptions(threshold=np.inf)
print(np.float32(k))