# coding: utf8
"""
Equation de la chaleur 2D résolue avec une méthode vectorielle itérative. 

Arthur Laffargue 06/05/2019
"""


import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors

##DATA

#Discretization 
Lx = 50e-3
Ly = Lx/2

Nx = 100
Ny = Nx//2

print('Number of degree of freedom : {}'.format((Nx+1)*(Ny+1)))
dx = Lx/Nx
dy = Ly/Ny

#Loop condition
tolerance = 1e-4
maxIteration  = 20000

#boundaries 
hotbound =  [j for j in range(3*(Nx+1)//8,5*(Nx+1)//8+1)]
coldbound = [j for j in range(Nx+1)]

Tcold = 293.15
Thot = 350.15



## Solve with a loop 


T_guess = np.ones((Ny+1,Nx+1))*Tcold
T_new = np.zeros_like(T_guess)

delta = np.linalg.norm((T_guess-T_new))
step = 0
while delta > tolerance and step < maxIteration :     
    
    T_new[1:-1,1:-1] = 1/(2/dx**2+2/dy**2)*\
                       ((T_guess[1:-1,:-2] + T_guess[1:-1,2:])/dx**2  +\
                        (T_guess[:-2,1:-1] + T_guess[2:,1:-1])/dy**2 )
    
    #Zeros gradient boundaries 
    T_new[0] = T_new[1]
    T_new[-1] = T_new[-2]
    T_new[:,0] = T_new[:,1]
    T_new[:,-1] = T_new[:,-2]
    
    #Imposed temperature on coldbound 
    T_new[-1,coldbound] = Tcold
    #Imposed temperature on hotbound 
    T_new[0,hotbound] = Thot

    #update delta and k 
    step+=1
    delta = np.max(np.abs(T_guess-T_new))
    T_guess = T_new.copy()


##PLOT
print('\nSOLUTION IS DONE')
print("number of iterative steps : {} \nrelative error : {:.10f}".format(step,delta))

x = np.linspace(0,Lx,Nx+1)
y = np.linspace(0,Ly,Ny+1)
X,Y = np.meshgrid(x,y)

fig = plt.figure(0)

Tmax = T_new.max()
Tmin = T_new.min()

ax = fig.add_subplot(111)
ax.set_title("Solution itérative [Python]")

ax.contour(X,Y,T_new,
                colors = 'white',
                levels=np.linspace(Tmin,Tmax,20),
                norm=colors.Normalize(vmin=Tmin,vmax=Tmax))

ctf = ax.contourf(X,Y,T_new,
                cmap = 'jet',
                levels=np.linspace(Tmin,Tmax,500),
                norm=colors.Normalize(vmin=Tmin,vmax=Tmax))
fig.colorbar(ctf)
ax.axis('equal')
ax.set_xlim(0,Lx)

plt.show()    
plt.grid()
#fig.savefig("solution_2Diterative.png")


