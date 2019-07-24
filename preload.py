import numpy as np
import json

Plist = np.zeros([156,0],np.ndarray)
mlist = None
for i in range(1,6):
    print(i)
    f = open("D:/paper/tensor syn/data/%d.json"%i,'r')
    j = json.load(f)
    loaded = 0
    if mlist is None:
        mlist = np.array(j['mlist'])
        Plist = np.array(j['P'][loaded:len(mlist)])
    else :
        loaded = len(mlist)
        mlist = np.hstack([mlist,np.array( j['mlist'] )] )
        Plist = np.vstack( [Plist,np.array( j['P'][loaded:len(mlist)] ,np.ndarray)] )
    #Plist += np.array(j['P'])



print(Plist[40][0] )
#ans = {'P':x,'mlist':m}
#f = open('D:/paper/tensor syn/data/merged.json')
#json.dump(f)