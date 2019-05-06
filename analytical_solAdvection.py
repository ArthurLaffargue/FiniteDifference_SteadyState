# coding: utf8
"""
Solution problème modèle équation d'advection-diffusion problème modèle. 
Arthur Laffargue 06/05/2019
"""

import numpy as np 
import matplotlib.pyplot as plt 

L = 1
XPe = [100,10,1,0.1]

x = np.linspace(0,L,500)

plt.figure(1)

for Pe in XPe :
    y = (np.exp(Pe*x/L)-1)/(np.exp(Pe)-1)
    plt.plot(x,y,label = '$P_e$ = '+str(Pe))

plt.xlabel('x')
plt.ylabel('u(x)')
plt.grid()
plt.title('Solution exacte en fonction de Pe')
plt.legend()
plt.show()

#plt.savefig("analytical_solution.png")
