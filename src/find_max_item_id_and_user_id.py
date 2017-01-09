import numpy as np
import codecs
import math
import operator


file_train='C:/hoc tap/big data/data/ml-1m/ml-1m/split/train2.txt'
file_test='C:/hoc tap/big data/data/ml-1m/ml-1m/split/test2.txt'

max_item_id=0
max_user_id=0
muy=0
num_muy=0

infile=codecs.open(file_train , 'r' , encoding='utf-8' )
for line in infile :
    line=line.split('::')
    user_id=int(line[0])
    item_id=int(line[1])
    rating=int(line[2])
    
    if user_id>max_user_id:
        max_user_id=user_id
    if item_id>max_item_id:
        max_item_id=item_id  
    
    muy+=rating
    num_muy+=1
             
        
#luu cac gia tri nay
max_user_id=[max_user_id]
max_item_id=[max_item_id]

np.save('file npy/max_user_id.npy',max_user_id)
np.save('file npy/max_item_id.npy',max_item_id)

muy=muy/num_muy
muy=[muy]
np.save('file npy/muy.npy',muy)


print(max_user_id)
print(max_item_id) 
print(muy)         