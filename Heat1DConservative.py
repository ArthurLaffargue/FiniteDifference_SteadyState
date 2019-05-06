# coding: utf8
"""
Solve the heat equation in a conservative form with a variable conductivity.
Résolution de l'équation de la chaleur sous forme conservative avec une conductivité variable.
Arthur Laffargue - 06/05/2019 

"""

## Imports 

import numpy as np 
from numpy.linalg import solve
import matplotlib.pyplot as plt 
##Data 

k1,x1 = 0.28,0.1
k2,x2 = 0.84,0.25
k3,x3 = 0.04,0.05
L = x1+x2+x3
k = lambda x : k1*(x<=x1) + k2*(x>x1)*(x<=x2+x1) + k3*(x>x1+x2)*(x<=x2+x1+x3) 




Tleft = 270.15
Tright = 291.15

##Discretization and matrix

N = 75
dx = L/N
x = np.linspace(0,L,N+1)
f = np.zeros_like(x)
K = np.zeros((N+1,N+1))

for i in range(1,N):
    ki = k((x[i-1]+x[i])/2)
    kj = k((x[i+1]+x[i])/2)
    K[i,i-1:i+2:] = [ ki/dx**2,-ki/dx**2-kj/dx**2,kj/dx**2 ]
    
##Boundary conditions
K[0,0] = 1
K[-1,-1] = 1
f[0] = Tleft
f[-1] = Tright 

##Solve 
T = solve(K,f)

##Plot 
fig = plt.figure(1,figsize = (8,4))

plt.subplot(121)
plt.plot(x,k(x),'r-',lw = 1.8)
plt.xlabel('x [m]')
plt.ylabel('k [$W.K^{-1}.m^{-1}$]')
plt.title("Conductivité thermique")
plt.grid(True)

plt.subplot(122)
plt.plot(x,T,'b-',lw = 1.8)
plt.xlabel('x [m]')
plt.ylabel('T [K]')
plt.title("Température solution")
plt.grid(True)

plt.subplots_adjust(hspace=0.3,wspace=0.3)
plt.show()

#fig.savefig("solution_pb2.png")
