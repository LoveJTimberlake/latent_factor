import numpy as np
import codecs
import math
#import operator
import os

file_test='C:/hoc tap/big data/data/ml-1m/ml-1m/split/test1.txt'

muy=np.load('file npy/muy.npy')[0]
Q=np.load('file npy/parameters/3Q.npy')
P=np.load('file npy/parameters/3P.npy')
BX=np.load('file npy/parameters/3BX.npy')
BI=np.load('file npy/parameters/3BI.npy')
max_user_id=np.load('file npy/max_user_id.npy')[0]
max_item_id=np.load('file npy/max_item_id.npy')[0]

R_predict=muy+BI+np.transpose(BX)+np.dot(Q,np.transpose(P))

#load du lieu test
R_test=np.zeros((max_item_id,max_user_id))
H_test=np.zeros((max_item_id,max_user_id))
infile=codecs.open(file_test , 'r' , encoding='utf-8' )
for line in infile :
    line=line.split('::')
    user_id=int(line[0])
    item_id=int(line[1])
    rating=int(line[2])  
    R_test[item_id-1][user_id-1]=rating
    H_test[item_id-1][user_id-1]=1 
num_test_rate=np.sum(H_test)
    
#tinh rmse tren test
test_rmse = math.sqrt( np.sum((H_test*R_test-H_test*R_predict)**2) / num_test_rate )   

print('rmse : '+str(test_rmse))    