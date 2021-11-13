import numpy as np
import matplotlib.pyplot as plt
from numba import jit #this speeds up the loops in the simulation

Lx = 100
Ly = 100

grid = np.zeros((Lx, Ly))

boundry = np.copy(grid)
boundry[:] = np.float("nan")

#setter grense betingelser
tol = 3
for ix in range(Lx):
    for iy in range(Ly):
        if abs(np.sqrt(ix**2 + iy**2) - (Lx+Ly)/2-1) < tol:
            boundry[ix, iy] = 0



boundry[0, 0] = 1

whole = np.zeros((2*Lx-1, 2*Ly-1))

whole[Lx-1:, Ly-1:] = boundry
whole[Lx-1:, :Ly] = np.transpose(boundry[::-1])
whole[:Lx, Ly-1:] = boundry[::-1]
botleft = np.transpose(boundry[::-1])
whole[:Lx, :Ly] = botleft[::-1]

border = np.where(whole == 0)

"""
plt.contourf(whole)
plt.axis("equal")
plt.show()
"""
@jit
def solvepoisson(b,nrep, border):
    # b = boundary conditions
    # nrep = number of iterations

    z = np.copy(b)     # z = electric potential field
    j = np.where(np.isnan(b)) #checks for where the points have no value, assigns them the value 0
    z[j] = 0.0
    
    znew = np.copy(z)
    Lx = np.size(b,0) #determine the x range of the point grid
    Ly = np.size(b,1) #determine the y range of the point grid
    
    for n in range(nrep): 
        for ix in range(Lx):
            for iy in range(Ly):
                ncount = 0.0 
                pot = 0.0
                if (np.isnan(b[ix,iy])): #check for cases in which the value is unspecified in the original grid
                    #Now, add up the potentials of all the the points around it
                    if (ix>0): 
                        ncount = ncount + 1.0
                        pot = pot + z[ix-1,iy]
                    if (ix<Lx-1):
                        ncount = ncount + 1.0
                        pot = pot + z[ix+1,iy]
                    if (iy>0):
                        ncount = ncount + 1.0
                        pot = pot + z[ix,iy-1]
                    if (iy<Ly-1):
                        ncount = ncount + 1.0
                        pot = pot + z[ix,iy+1]
                    znew[ix,iy] = pot/ncount #Divide by the number of contributing surrounding points to find average potential
                else:
                    znew[ix,iy]=z[ix,iy] #If the value is specified, keep it
        tmp_z = znew # Swapping the field used for the calucaltions with the field from the previous iteration
        znew = z     # (to prepare for the next iteration)
        z = tmp_z
    z[border] = np.float("nan")     
    return z

nrep = 1

s = solvepoisson(whole, nrep, border)

plt.imshow(s)
plt.axis("equal")
plt.show()