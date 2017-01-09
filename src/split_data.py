import numpy as np
import codecs
import random

file_data='C:/hoc tap/big data/data/ml-1m/ml-1m/split/train1.txt'

test_data=[]
train_data=[]

infile=codecs.open(file_data , 'r' , encoding='utf-8' )
current_user_id=1
current_list_item=[]
num1=0
num2=0
for line in infile :
    num1+=1
    line=line.split('::')
    user_id=int(line[0])
    item_id=int(line[1])
    rating=int(line[2])
    if user_id != current_user_id:
        #chon ra k item tu list de dua vao tap test
        num2+=len(current_list_item)
        k=int(len(current_list_item)/5)
        list_test_item=[]
        random.shuffle(current_list_item)
        #for i in range(k):
        #    list_test_item.append(current_list_item[i])
        
        #test_data.append([user_id,list_test_item])
        test_data.append([current_user_id,current_list_item[0:k:]])
        train_data.append([current_user_id,current_list_item[k::]])
        
        current_user_id=user_id
        current_list_item=[]
        current_list_item.append((item_id,rating))
        
    else:
        current_list_item.append((item_id,rating))

if True:
    num2+=len(current_list_item)
    k=int(len(current_list_item)/5)
    list_test_item=[]
    random.shuffle(current_list_item)    
    test_data.append([current_user_id,current_list_item[0:k:]])
    train_data.append([current_user_id,current_list_item[k::]])        
print('num1:'+str(num1))
print('num2:'+str(num2))

#in train data va test data ra file
outfile=codecs.open('C:/hoc tap/big data/data/ml-1m/ml-1m/split/train6.txt' , 'w' , encoding='utf-8' )
for i in range(len(train_data)):
    data=train_data[i]
    user_id=data[0]
    list_item=data[1]
    for j in range(len(list_item)):
        outfile.write(str(user_id)+'::'+str(list_item[j][0])+'::'+str(list_item[j][1])+'\r\n')
outfile.close()
            
outfile=codecs.open('C:/hoc tap/big data/data/ml-1m/ml-1m/split/test6.txt' , 'w' , encoding='utf-8' )
for i in range(len(test_data)):
    data=test_data[i]
    user_id=data[0]
    list_item=data[1]
    for j in range(len(list_item)):
        outfile.write(str(user_id)+'::'+str(list_item[j][0])+'::'+str(list_item[j][1])+'\r\n')
outfile.close()            
    