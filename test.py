import syntensor
import numpy as np
import json
import sys

fname = 'data_10.json'
sys.stdin = open(fname,'r')
data = input()
data = json.loads(data)
Plist = np.array(data['P'])

mlist = np.array(data['mlist'])
n = len(mlist)
for i in range(n):
    for j in range(i+1,n):
        Plist[j][i] = np.transpose(Plist[i][j])
print(Plist[0][1])
tensor = syntensor.SynTensor(n,mlist,Plist)
print(tensor.solution())