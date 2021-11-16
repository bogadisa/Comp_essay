import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import animation
plt.style.use('seaborn-pastel')


@jit
def findEfieldinapoint(r, Q, R):
    x, y = r
    #lager V slik at den har samme form som x, y og z
    V = np.zeros(np.shape(x))
    #går igjennom alle elementene i x, y og z matrisene
    for i in range(len(x.flat)):
        #lager lille r (posisjonen til test ladningen)
        r = np.array([x.flat[i], y.flat[i]])
        #regner ut V for alle tre dimensjonene
        V.flat[i] += 1/(4*np.pi)*sum(Q[j]/np.linalg.norm(r - R[j]) for j in range(len(Q)))
    #finner E som den negative gradienten til V
    E=-np.array(np.gradient(V))
    #retunerer E og V
    return E, V

r0 = 10
r1 = 150

N = 100
gen = 10
R = np.zeros((N*gen, 2))
Q = np.zeros(N*gen)
L = np.linspace(-r1, r1, 50)

x, y = np.meshgrid(L, L, indexing="ij")
def animate(i):
    plt.gca().clear()
    for j in range(gen):
        angle = np.random.random()*2*np.pi
        R[gen*i + j, 0] = r0*np.cos(angle)
        R[gen*i + j, 1] = r0*np.sin(angle)
        Q[gen*i + j] = -1
    E, V = findEfieldinapoint([x, y], Q[:gen*(i+1)], R[:gen*(i+1)])
    cont = plt.contourf(x, y, V, 40)
    if i == 0:
        plt.colorbar()
    return cont,

fig = plt.figure()
anim = animation.FuncAnimation(fig, animate, frames=N)

anim.save("figures/potential_storage.gif", writer="imagemagick")
plt.show()