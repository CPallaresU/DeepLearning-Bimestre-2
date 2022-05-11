# -*- coding: utf-8 -*-
"""
Created on Mon May  9 22:58:46 2022

@author: 10
"""

import random
import numpy as np
import matplotlib.pyplot as plt




#############################################
############### PRIMER PUNTO ################
#############################################

lr = 0.01
x = random.random()
x_ = x
y = -1*(x-2)*(x-2) + 3
dy_dx = (-2 * x) + 4
epochs = 1000


for e in range(epochs):
    
    dy_dx =  (-2 * x) + 4
            
                
    x = x + dy_dx*lr

    if dy_dx > 0 and dy_dx < 0.01:
        
        print("")
        print("PUNTO #1 :")
        print("")
        print("      Valor de mi derivada en el punto x = {} : {}".format(x,dy_dx))
        print("      Se usaron {} epochs con un valor inicial de x : {}".format(e,x_))
        print("")
        
        """
        
        
              El momento en el que se detiene mi proceso es cuando mi derivada está entre 
              0 y aproximadamente 0,  0 < DY/DX < 0.01 (línea 34).
              
              Dependiendo el valor con el que inicialice, se verá afectado el tiempo de 
              convergencia para llegar al mínimo o máximo, entre mas alejado del mínimo 
              me cuentre, más tiempo le tomará al algoritmo converger. El tamaño del 
              learning rate afectará tambíen el tiempo de convergencia, ya que tener uno tan
              grande puede que me haga saltos muy grandes y me sale el máximo o mínimo y nunca
              encontrarlos o encontrarlos pero a un paso muy lento.
              
              Es importante resaltar que nuestra parabola es CONCAVA, por lo que no queremos 
              descender para buscar un mínimo sino queremos ascender para buscar un máximo, 
              por lo que busamos X = X + LR * DY/DX    y no    X = X = X - LR * DY/DX para 
              actualizar el párametro X.
              
              Entre los hiperparametros tenemos:
                  
                  * Valor inicial de x (x = random.random())
                  * Valor del learning rate (lr = 0.01)
                  * Cantidad de epochs (epochs = 1000)
              
              
              
        """
        
        break;
    
        

#############################################
############### SEGUNDO PUNTO ###############
#############################################

random.seed(0)
w1 = random.random()
w2 = random.random()
w3 = random.random()
data = np.linspace(0.0, 4.0, num=10000)
y = -1*np.square(data-2) + 3 + 0.2 * np.sin(16*data)
epochs = 100
lr = 0.001
l_error=[]

print("")
print("PUNTO #2 :")
print("")
print("     Resultados en gráfico")
print("")

plt.figure(1)
plt.subplot(211)

plt.plot((data*data)*w1 + data*w2 + w3, label = "Y_hat before fitting")



for e in range(epochs):
    
    for i in range(0,len(data)):
        
        x = data[i]
        
        y_pred=(x*x)*w1 + x*w2 + w3
        
        dl_w1 = -2*(y[i] - y_pred)*(x*x)
        dl_w2 = -2*(y[i] - y_pred)*(x)
        dl_w3 = -2*(y[i] - y_pred)*1
        
        w1 = w1 - lr*dl_w1
        w2 = w2 - lr*dl_w2
        w3 = w3 - lr*dl_w3
        error = (y[i] - y_pred)*(y[i] - y_pred)/len(y)
        l_error.append(error)
        
        """
        
            La regla de actualización de cada párametro (w1, w2 y w3) está dado por la derivada de Y_hat
            con respecto a cada párametro.
        
        
        """
        
        

plt.plot(y, label = "f(x) = y")
plt.plot((data*data)*w1 + data*w2 + w3, label = "Y_hat fatter fitted W")
plt.legend(loc="upper left")


plt.subplot(212)
plt.plot(l_error, label = "Error through epochs")
plt.legend(loc="upper right")





