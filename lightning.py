import numpy as np
import matplotlib.pyplot as plt
from numba import jit #this speeds up the loops in the simulation

# The jit command ensures fast execution using numba
@jit
def solvepoisson(b,nrep):
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
    return z 


# First, we set up the boundary conditions
Lx = 50
Ly = 50
z = np.zeros((Lx,Ly),float)
b = np.copy(z)
c = np.copy(z)
b[:] = np.float('nan')

# Set the potential at the top of the grid to 1
b[:,0] = 1.0

# Set the potential at the bottom of the grid to 0
b[:,Ly-1]=0.0

# Create a copy of the boundary conditions matrix which will be used to check 
#for possible locations for the lightning's path
zeroneighbor = np.copy(z) 
zeroneighbor[:] = 0.0 #set all values in it equal to 0
#set the values next to the ground equal to 'nan'. This is where the lightning can start
zeroneighbor[:,Ly-2] = np.float('nan') 


nrep = 3000 # Number of jacobi steps
eta = 1.0 #A factor that will be used in probability calculation
ymin = Ly-1 #The y value where we will stop (just above the ground)
ns = 0
while (ymin>0): 
    # First find potential on the entire grid, based on the original boundary conditions
    s = solvepoisson(b,nrep)

    # Probability that lightning will move to a new position may depend on potential to power eta
    sprob = s**eta
    # We also multiply by a random number, uniform between 0 and 1, to introduce some randomness
    # And we multiply with isnan(zeroneighbor) to ensure only 'nan' points can be chosen
    sprob = sprob*np.random.uniform(0,1,(Lx,Ly))*np.isnan(zeroneighbor) 
    
    #now, find the point with max probability 
    [ix,iy] = np.unravel_index(np.argmax(sprob,axis=None),sprob.shape)
    
    # Update the boundary condition array to set the potential where the lightning is to 0
    b[ix,iy] = 0.0
    
    # Update neighbor positions of the lightning path to 'nan' (making them possible choices for the next iteration) 
    if (ix>0):
        zeroneighbor[ix-1,iy]=np.float('nan')
    if (ix<Lx-1):
        zeroneighbor[ix+1,iy]=np.float('nan')
    if (iy>0):
        zeroneighbor[ix,iy-1]=np.float('nan')
    if (iy<Ly-1):
        zeroneighbor[ix,iy+1]=np.float('nan')
        
    ns = ns + 1
    c[ix,iy] = ns #create an array of the lightning's path, scaled by the number of loops
    if (iy<ymin): #iterate to the next set of y-values
        ymin = iy

plt.rcParams['figure.figsize'] = [16, 4]
plt.subplot(1,3,1)
plt.imshow(c.T) #create a plot of the lightning's path
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(s.T) #create a plot of the final potential
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(sprob.T) #create a plot of the relative probabilities of the next step
plt.colorbar()
plt.show()