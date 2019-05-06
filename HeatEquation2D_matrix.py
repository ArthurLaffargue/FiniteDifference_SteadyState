# coding: utf8
"""
Equation de la chaleur 2D résolue avec méthode matricielle. 
Arthur Laffargue 06/05/2019
"""


import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import colors
from scipy.sparse import csc_matrix, lil_matrix, kron, eye
from scipy.sparse.linalg import spsolve
##DATA

#Discretization 
Lx = 50e-3
Ly = Lx/2

Nx = 100
Ny = Nx//2
ndof = (Nx+1)*(Ny+1)

print('Number of degree of freedom : {}'.format(ndof))
dx = Lx/Nx
dy = Ly/Ny

hotbound =  [j for j in range(3*(Nx+1)//8,5*(Nx+1)//8+1)]
coldbound = [j+Ny*(Nx+1) for j in range(Nx+1) ]

Tcold = 293.15
Thot = 350.15

##Compute matrices

A = 1/dx**2*(eye(Nx+1,k=1)+eye(Nx+1,k=-1)-2*eye(Nx+1)) - 2*1/dy**2*eye(Nx+1)
B = 1/dy**2*eye(Nx+1)

K = kron(eye(Ny+1),A) + kron(eye(Ny+1,k=1)+eye(Ny+1,k=-1),B)
K = lil_matrix(K) #lil matrix will be more efficient for slicing
F = np.zeros(ndof)

##Boundary conditions 

#Zeros gradient 
#y = 0 
index = np.setdiff1d([j for j in range(Nx+1)],hotbound)
K[index] = 0 
K[index,index] = 1/dy
K[index,index+(Nx+1)] = -1/dy

#x = 0
index = np.array([(Nx+1)*i for i in range(Ny+1)])
K[index] = 0 
K[index,index] = 1/dx
K[index,index+1] = -1/dx

#x = Lx
index = np.array([(Nx+1)*i + Nx for i in range(Ny+1)])

K[index] = 0 
K[index,index] = 1/dx
K[index,index-1] = -1/dx

#imposed temperature
index = hotbound + coldbound
K[index] = 0
K[index,index] = 1
F[index] = [Thot]*len(hotbound) + [Tcold]*len(coldbound)

##Solve
#transform in csc_matrix
K = csc_matrix(K)

memory = K.indices.nbytes + K.indptr.nbytes + K.data.nbytes
print("Dimension of sparse matrix [K] :",K.shape,"; memory :",
      memory,"bytes")

T = spsolve(K,F)
T = T.reshape((Ny+1,Nx+1))

##PLOT
x = np.linspace(0,Lx,Nx+1)
y = np.linspace(0,Ly,Ny+1)
X,Y = np.meshgrid(x,y)

fig = plt.figure(0)

Tmax = T.max()
Tmin = T.min()

ax = fig.add_subplot(111)
ax.set_title("Solution matricielle [Python]")

ax.contour(X,Y,T,
                colors = 'white',
                levels=np.linspace(Tmin,Tmax,20),
                norm=colors.Normalize(vmin=Tmin,vmax=Tmax))

ctf = ax.contourf(X,Y,T,
                cmap = 'jet',
                levels=np.linspace(Tmin,Tmax,500),
                norm=colors.Normalize(vmin=Tmin,vmax=Tmax))
fig.colorbar(ctf)
ax.axis('equal')
ax.set_xlim(0,Lx)

plt.show()    
plt.grid()
#fig.savefig("solution_2Dmatrix.png")
