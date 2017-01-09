import numpy as np
import matplotlib.pyplot as plt
import codecs
import math
import sys

file_result='file txt/result.txt'

k_list=[]
rmse_list=[]


infile=codecs.open(file_result , 'r' , encoding='utf-8' )
for line in infile :
    line=line.split('::')
    k_list.append(int(line[0]))
    rmse_list.append(float(line[1]))


plt.ylabel('validation rmse')
plt.xlabel('K')
plt.plot(k_list,rmse_list,marker='o',color='b')
plt.grid()
plt.show()

'''    
fig, ax = plt.subplots()
plt.ylabel('rmse average')
plt.xlabel('K')
ax.plot(k_list,rmse_list,marker='o',color='r')
ax.set_yscale('logit')
plt.show() 
'''
   