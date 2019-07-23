import syntensor
import numpy as np
import json
import sys

image_nums = 156
fname = 'new_data_start_0_end_156_all_156.json'
sys.stdin = open(fname,'r')
data = input()
data = json.loads(data)
Plist = np.array(data['P'],np.ndarray)

np.set_printoptions(linewidth = np.nan)
np.set_printoptions(threshold=np.inf)
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
#sol = tensor.solution()
#sol =
k = tensor.rounded_solution(0.5,2,3 )
print(np.shape(k))

print(k)

tensor.visualize(image_nums ,'D:\\paper\\tensor syn\\data\\image_list.npy', 'D:\\paper\\tensor syn\\tmp_%d'%image_nums)
#f = open('D:\\paper\\tensor syn\\res.txt','w')
#sys.stdout = f
#print(np.float32(sol))
#print(np.float32(k))