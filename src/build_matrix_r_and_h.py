import numpy as np
import codecs
import math
import operator

file_train='C:/hoc tap/big data/data/ml-1m/ml-1m/split/train2.txt'

max_user_id=np.load('file npy/max_user_id.npy')[0]
max_item_id=np.load('file npy/max_item_id.npy')[0]


R=np.zeros((max_item_id,max_user_id))
H=np.zeros((max_item_id,max_user_id))

infile=codecs.open(file_train , 'r' , encoding='utf-8' )
for line in infile :
    line=line.split('::')
    user_id=int(line[0])
    item_id=int(line[1])
    rating=int(line[2])
    
    R[item_id-1][user_id-1]=rating
    H[item_id-1][user_id-1]=1
    

np.save('file npy/R.npy',R)
np.save('file npy/H.npy',H)

print(np.shape(R)) 


