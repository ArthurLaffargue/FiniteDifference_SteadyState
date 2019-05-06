# coding: utf8
"""
Comparaison des shémas centré et décentré amont pour l'équation modèle du problème d'advection diffusion. 

Arthur Laffargue 06/05/2019
"""

import numpy as np 
import matplotlib.pyplot as plt 
from numpy.linalg import solve, norm 

##Data 
c = 1.0
d = 0.02
L = 1

Pe = c*L/d
print('Peclet number  = {:.2f}'.format(Pe))

N = [6,10,15,40]
##Analytical solution 

f =lambda x  : (np.exp(Pe*x/L)-1)/(np.exp(Pe)-1)


##Centered scheme 

def centered(n) : 
    
    #discretization
    dx = L/n
    x = np.linspace(0,L,n+1)

    #Matrices
    C = c*( np.eye(n+1,k=1) - np.eye(n+1,k=-1) )/(2*dx)
    D = (d )/dx**2*( np.eye(n+1,k=1) + np.eye(n+1,k=-1) - 2*np.eye(n+1) )
    f = np.zeros_like(x)
    K = C-D
    #Boundary condition
    K[0] = 0 ; K[0,0] = 1 ; f[0] = 0
    K[-1] = 0 ; K[-1,-1] = 1 ; f[-1] = 1

    #step 4 : solve 
    u = solve(K,f)
    return x,u
        


##Upwind scheme 

def upwind(n) : 
    
    #discretization
    dx = L/n
    x = np.linspace(0,L,n+1)
    #Matrices
    C = c*( np.eye(n+1,k=0) - np.eye(n+1,k=-1) )/(dx)
    D = d/dx**2*( np.eye(n+1,k=1) + np.eye(n+1,k=-1) - 2*np.eye(n+1) )
    f = np.zeros_like(x)
    K = C-D
    #Boundary condition
    K[0] = 0 ; K[0,0] = 1 ; f[0] = 0
    K[-1] = 0 ; K[-1,-1] = 1 ; f[-1] = 1

    #step 4 : solve 
    u = solve(K,f)

    return x,u



##Plots
x_ana = np.linspace(0,L)


#Centered
fig1 = plt.figure(1)
plt.suptitle("Comparaison schéma upwind et centré $P_e$ = "+str(Pe))
i = 1
for n in N : 
    
    x_upwind ,u_upwind = upwind(n)
    plt.subplot(2,2,i)
    plt.plot(x_upwind ,u_upwind ,'x-',label = 'upwind '+ str(n) + ' noeuds',alpha = 0.8,lw = 1.0)
    
    x_centered ,u_centered = centered(n)
    plt.plot(x_centered ,u_centered ,'x-',label = 'centré '+ str(n) + ' noeuds',alpha = 0.8,lw = 1.0)
    plt.plot(x_ana,f(x_ana),label = 'Analytical solution')
    plt.legend(loc = 0)
    plt.grid()
    i += 1

plt.show()
#plt.savefig("problème_modèle.png")
