# coding: utf8
"""
Solveur pour l'équation de la chaleur en 1D.
Arthur Laffargue - 06/05/2019 

"""

##Imports 
from numpy import eye, linspace
from numpy.linalg import solve , norm

##Heat Equation Solver 
def solveHeat1D(k,q,L,n,left,right): 
    
    """ Solve the 1D stationary heat equation : 
    k∇²(T) + q(x) = 0
    with the boundary conditions defined in 'left" and "rigth". 
    
    INPUT : 
    k : thermal conductivity - type : float ; 
    q : volumetric heat source - type : function of x. Must be vectorized ; 
    L : length of the problem - type : float ; 
    n : number of points - type : integer ; 
    left : the left boundary condition - type : tuple ; 
    rigth : the right boundary condition - type : tuple ; 
    
    RETURN ; 
    x : vector of the discretization of the space : numpy.array , size = n+1
    u : vector of nodal temperatures : numpy.array , size = n+1
    
    
    BOUNDARY CONDITIONS : 
    
    for right and left there are 3 possibility ; temperature, flux or convection. 
    
    Imposed temperature ; 
        right/left = ('temperature',T) 
        
        Where T is the imposed temperature (float).
    
    Imposed flux : 
        right/left = ('flux',g)
        
        Where g is the imposed flux (float). 
    
    Convection : 
        right/left = ('convection',h,T)
        
        Where h is the convection factor (float) ; 
        and T is the external temperature (float). 
    
    """
    
    # Step 0 : discretization 
    x = linspace(0,L,n+1)
    dx = L/n
    
    # step 1 : compute the matrices 
    K = k/dx**2*(eye(n+1,k=-1)-2*eye(n+1)+eye(n+1,k=1))
    F = -q(x)
    
    # step 2 : Boundary condition left 
    if left[0].lower() == 'temperature' : 
        T0 = left[1]
        K[0,:2] = [1,0]
        F[0] = T0

    if left[0].lower() == 'flux' : 
        g0 = left[1]
        K[0,:2] = [k/dx,-k/dx]
        F[0] = g0
        
    if left[0].lower() == 'convection':
        h,T0 = left[1:]
        K[0,:2] = [k/dx+h,-k/dx]
        F[0] = h*T0
        
    # step 3 : Boundary condition rigth  
    if right[0].lower() == 'temperature' : 
        Tn = right[1]
        K[-1,-2:] = [0,1]
        F[-1] = Tn
        
    if right[0].lower() == 'flux' : 
        gn = right[1]
        K[-1,-2:] = [-k/dx,k/dx]
        F[-1] = gn

    if right[0].lower() == 'convection' : 
        h,Tn = right[1:]
        K[-1,-2:] = [-k/dx,k/dx+h]
        F[-1] = h*Tn    
        
    #step 4 : solve 
    u = solve(K,F)
    
    print("-"*50)
    print('SOLUTION IS DONE.')
    print("residual : {:.5f}".format(norm(K.dot(u)-F)))
    print("-"*50)
    
    return x,u
        





















##TESTING
if __name__ == '__main__':
    
    import numpy as np
    
    k = 0.5
    n = 150
    L = 0.01
    
    Td,hd = 293.15,5
    Tg,hg = 288.15,12
    
    xc,xd = 0.006 , 0.0005
    source = lambda x : 500/(xd*np.sqrt(np.pi))*np.exp(-(x-xc)**2/xd**2)
    
    left = ('convection',hg,Tg)
    right = ('convection',hd,Td)
    
    
    x,T = solveHeat1D(k,source,L,n,left,right)
    
    import matplotlib.pyplot as plt 
    
    
    fig = plt.figure(1,figsize = (8,3))
    
    plt.subplot(121)
    plt.plot(x,source(x),'r-',lw = 1.8)
    plt.xlabel('x [m]')
    plt.ylabel('Q [$W.m^{-3}$]')
    plt.title("Source volumique de chaleur")
    plt.grid(True)
    
    plt.subplot(122)
    plt.plot(x,T,'b-',lw = 1.8)
    plt.xlabel('x [m]')
    plt.ylabel('T [K]')
    plt.title("Température solution")
    plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    plt.show()
    
    #fig.savefig("solution_pb1.png")
            
        
        
        
