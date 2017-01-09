import numpy as np
import codecs
import math
#import operator
import os
#from numpy import reshape

file_train='C:/hoc tap/big data/data/ml-1m/ml-1m/split/train2.txt'
file_test='C:/hoc tap/big data/data/ml-1m/ml-1m/split/test2.txt'

k=20
gamma=0.0002  # Learning rate

lmbda=0.1

R=np.load('file npy/R.npy')
muy=np.load('file npy/muy.npy')[0]
H=np.load('file npy/H.npy')
max_user_id=np.load('file npy/max_user_id.npy')[0]
max_item_id=np.load('file npy/max_item_id.npy')[0]


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

num_train_rate=np.sum(H)
num_test_rate=np.sum(H_test)

while True:
     
    #khoi tao P,Q,BX,BI
    
    Q = 1-2*np.random.rand(max_item_id,k)
    P = 1-2*np.random.rand(max_user_id,k) 
    BX= 1-2*np.random.rand(max_user_id,1)
    BI= 1-2*np.random.rand(max_item_id,1)
    
    #bat dau train
    if os.path.isfile('file npy/parameters/Q.npy'):
        Q=np.load('file npy/parameters/Q.npy')
    if os.path.isfile('file npy/parameters/P.npy'):
        P=np.load('file npy/parameters/P.npy')   
    if os.path.isfile('file npy/parameters/BX.npy'):
        BX=np.load('file npy/parameters/BX.npy')   
    if os.path.isfile('file npy/parameters/BI.npy'):
        BI=np.load('file npy/parameters/BI.npy')   
    
    n_loop=0 
    if os.path.isfile('file npy/parameters/nloop.npy'): 
        n_loop=np.load('file npy/parameters/nloop.npy')[0]  
    
    test1=1000000
    test2=1000000
    train1=1000000
    train2=1000000    
    while True:
        n_loop+=1
        #print('num loop : '+str(n_loop))
        
        delta_Q= -2*np.dot(H*(R-np.dot(Q,np.transpose(P))-BI-np.transpose(BX)-muy),P) +  2*lmbda*Q
        
        delta_P= -2*np.dot(np.transpose(H*(R-np.dot(Q,np.transpose(P))-BI-np.transpose(BX)-muy)),Q) + 2*lmbda*P
        
        delta_BI= np.sum(-2*H*(R-np.dot(Q,np.transpose(P))-BI-np.transpose(BX)-muy) , axis=1 ) .reshape((max_item_id,1)) + 2*lmbda*BI
        
        delta_BX =np.sum( np.transpose(-2*H*(R-np.dot(Q,np.transpose(P))-BI-np.transpose(BX)-muy)) , axis=1 ).reshape((max_user_id,1)) +2*lmbda*BX
        
        Q=Q-gamma*delta_Q
        P=P-gamma*delta_P
        BI=BI-gamma*delta_BI
        BX=BX-gamma*delta_BX
        
        
        if n_loop % 20==0:
            np.save('file npy/parameters/P.npy',P)
            np.save('file npy/parameters/Q.npy',Q)
            np.save('file npy/parameters/BX.npy',BX)
            np.save('file npy/parameters/BI.npy',BI) 
            nlooplist=[n_loop]
            np.save('file npy/parameters/nloop.npy',nlooplist)
            
            print('num loop : '+str(n_loop))
            
            #tinh rmse tren tap train
            R_new=muy+BI+np.transpose(BX)+np.dot(Q,np.transpose(P))
            #train_rmse = math.sqrt( np.sum((H*R_new-H*R)**2) / num_train_rate )
            train_loss=np.sum((H*R_new-H*R)**2)+lmbda*( np.sum(Q**2)+np.sum(P**2)+np.sum(BI**2)+np.sum(BX**2))
            print('train_loss : '+str(train_loss))
            
            #tinh rmse tren tap test
            test_rmse = math.sqrt( np.sum((H_test*R_test-H_test*R_new)**2) / num_test_rate )
            print('test_rmse : '+str(test_rmse))
            
            if test_rmse > test2 and test2 > test1:
                print('k = '+str(k))
                print('nloop = '+str(n_loop-40))
                print('train_rmse = '+str(train1))
                print('test_rmse = '+str(test1))
                break 
            
            test1=test2
            test2=test_rmse
            train1=train2
            train2=train_loss                       
        
    k+=2    
        