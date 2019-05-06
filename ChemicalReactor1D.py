# coding: utf8
"""
Résolution d'une equation d'adevction-diffusion-réaction en 1D avec la méthode Upwind.
Arthur Laffargue - 06/05/2019 

"""
##Imports 
import numpy as np 
from numpy.linalg import solve
import matplotlib.pyplot as plt 

##Data 

u = 0.5 #inlet velocity 
Cin = 1 #inelt concentration
alpha = 1.85e-5#chemical diffusivity
kr = 1.5 #reaction constant
L = 3.5 #lenght 

##Upwind scheme 

def solveReaction(n) : 
    print("Pe = {:.5f}".format(u*L/alpha))
    #discretization
    dx = L/n
    x = np.linspace(0,L,n+1)
    print("Pe_dx = {:.5f}".format(u*dx/alpha))
    #Matrices
    Umat = u*( np.eye(n+1,k=0) - np.eye(n+1,k=-1) )/(dx) #advection
    Dmat = alpha/dx**2*( np.eye(n+1,k=1) + np.eye(n+1,k=-1) - 2*np.eye(n+1) ) #diffusion
    Rmat = kr*np.eye(n+1) #reaction
    #Write the system A.x = b
    b = np.zeros_like(x)
    A = Umat-Dmat + Rmat
    #Boundary condition
    A[0] = 0 ; A[0,0] = 1 ; b[0] = Cin
    A[-1,-2:] = [-1,1] #right zero gradient
    #Solve 
    C = solve(A,b)
    return x,C


##Solve and plot
x,C = solveReaction(50)

fig = plt.figure(0)
plt.title("Concentration")
plt.plot(x,C,'o-',alpha = 0.7)
plt.xlabel('x [m]')
plt.ylabel('C(x)/C0 ')
plt.grid(True)
plt.show()
#fig.savefig("solution_pb3.png")
