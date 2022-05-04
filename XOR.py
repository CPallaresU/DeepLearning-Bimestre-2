# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:00:44 2022

@author: 10
"""

import numpy as np

def w_ (lista,output_len):
  w=[]
  b=[]

  for i in range(0,len(lista)):

    if i != len(lista)-1:

      weights=np.random.rand(lista[i],lista[i+1])
      bias=np.random.rand(lista[i],1)

    else:
      weights=np.random.rand(lista[i],output_len)
      bias=np.random.rand(lista[i],1)
    w.append(weights)
    b.append(bias)
  return w,b

def sigmoid_ (z):
 return 1/(1+np.exp(-z))


x=np.array([[0,0,1,1],[0,1,0,1]])
y=[0,1,1,0]

input_len=2
lista_w_layers = [input_len ,3]
output_len=1

w_0=np.random.rand(lista_w_layers[0],2)
w,b=w_(lista_w_layers,output_len)
lr=0.01
epochs=10000
y_pred=[0,0]
b=np.array([np.random.rand(3,1),np.random.rand(1,1)])
lista=[]


for j in range(epochs):
    
  error=0
  for i in range(np.shape(x)[1]):
     
    z =  np.dot(w[0].T,x[:,i]) + b[0].T
    
    
    y_pred[0]= sigmoid_(z)
    y_= y_pred[0].reshape(len(y_pred[0][0]),1)
    result=np.dot(y_.T, w[1])[0]
    y_pred[1]=  result + b[1][0]
    print(y_pred[1])
    
    
    
    error = error + np.square(y[i]-y_pred[1])


    w_delta= -2*(y[i]-y_pred[1][0])*np.array(y_pred[0])

    w_delta=w_delta.reshape(3, 1)
    w[1] = w[1] - w_delta * lr 
    b[1] = b[1] - -2*(y[i]-y_pred[1])*lr

        
    reshp_=(y_pred[0]-(y_pred[0]*y_pred[0]))#.reshape(np.shape(y_pred[0])[0],1)
    x_=x[:,i].reshape(np.shape(x[:,i])[0],1)
    reshp_= (x_.T*reshp_.T* w[1]).T
    
    w_delta_ = -2*(y[i]-y_pred[1]) * reshp_
    w[0] = w[0] - w_delta_ * lr
    
    
    dot= np.dot(y_pred[0],y_pred[0].T)[0][0]
    b[0] = b[0].T - -2*(y[i]-y_pred[1])*(y_pred[0]-dot)*lr
    b[0]=b[0].T
    
      
  lista.append(error/4)
  
  
  
  
import matplotlib.pyplot as plt
plt.plot(lista)


for i in range(0,4):
    
    z =  np.dot(w[0].T,x[:,i]) + b[0].T
    y_pred[0]= sigmoid_(z)
    
    y_= y_pred[0].reshape(3,1)
    y_pred[1]= np.dot(y_.T , w[1]) + b[1]
    print("Para {} el valor es: {} ".format(x[:,i],y_pred[1][0]))
