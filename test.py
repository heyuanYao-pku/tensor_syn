import syntensor
import numpy as np
import json
import sys

fname = 'data_156.json'
sys.stdin = open(fname,'r')
data = input()
data = json.loads(data)
Plist = np.array(data['P'])

mlist = np.array(data['mlist'])
print(mlist)

n = len(mlist)
for i in range(n):
    for j in range(i+1,n):
        Plist[j][i] = np.transpose(Plist[i][j])

tensor = syntensor.SynTensor(n,mlist,Plist)

print('object building finished')
k = tensor.solution()
print(np.shape(k))
f = open('result156.txt','w')
sys.stdout = f
np.set_printoptions(linewidth = np.nan)
np.set_printoptions(threshold=np.inf)
print(np.float32(k))