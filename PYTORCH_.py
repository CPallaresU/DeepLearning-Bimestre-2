# -*- coding: utf-8 -*-
"""
Created on Thu May 19 21:59:08 2022

@author: 10
"""
!pip install torch
import torch 
import torch.nn as nn
from torch import sigmoid
import matplotlib.pylab as plt
import numpy as np
import random
torch.manual_seed(0)


def model_(w1,w2,w3,x):

  k = torch.tensor(x,dtype=torch.float32)
  t_1=torch.mul(k, k)
  t_2=torch.mul(t_1,w1)
  t_3=torch.mul(k,w2)

  return t_2 + t_3 + w3

def loss_(y,y_hat):

  err = (y_hat-y)**2
  return torch.mean(err)


def train_(epochs,lr,x,w1,w2,w3,y):

  total_error=[]
  t = 0
  for e in range(epochs):

    for i in range(data.size) :

      val = model_(w1,w2,w3,x[i])
      l_ = loss_(y[i],val)
      loss_(y[i],val).backward()
      optimizer.step()
      optimizer.zero_grad()
      t = t + l_.item()

    total_error.append(t) 
    t=0

  return total_error , optimizer.param_groups


data = np.linspace(0.0, 4.0, num=100)
y = -1*np.square(data-2) + 3 + 0.2 * np.sin(16*data)
y_tensor = torch.from_numpy(y)

lr = 0.001
epochs = 1000
w1 = torch.randn(1,requires_grad=True)
w2 = torch.randn(1,requires_grad=True)
w3 = torch.randn(1,requires_grad=True)

optimizer = torch.optim.SGD([w1,w2,w3], lr=lr)
model = model_(w1,w2,w3,data)
t_ , params_ = train_(epochs,lr,data,w1,w2,w3,y_tensor)

#plt.plot(list(t_))

###############################
######## PLOT GRÁFICOS ########
###############################

w_1=params_[0]['params'][0].detach()
w_2=params_[0]['params'][1].detach()
w_3=params_[0]['params'][2].detach()

mod_ = model_(w_1,w_2,w_3,data)
plt.plot(list(mod_.numpy()),label = "PREDICTED DATA")
plt.legend(loc="upper left")



y = -1*np.square(data-2) + 3 + 0.2 * np.sin(16*data)
plt.plot(list(y),label = "REAL DATA")
plt.legend(loc="upper right")

