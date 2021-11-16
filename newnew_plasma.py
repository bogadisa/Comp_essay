import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from matplotlib import animation
plt.style.use('seaborn-pastel')

#@jit
def findEfieldinapoint(r, Q, R):
    x, y = r
    #lager V slik at den har samme form som x, y og z
    V = np.zeros(np.shape(x))
    #går igjennom alle elementene i x, y matrisene
    for k in range(len(x.flat)):
        #lager lille r (posisjonen til test ladningen)
        r = np.array([x.flat[k], y.flat[k]])
        #regner ut V for alle tre dimensjonene
        V.flat[k] += 1/(4*np.pi)*sum(Q[j]/np.linalg.norm(r - R[j]) for j in range(len(Q)))
    #finner E som den negative gradienten til V
    E = -np.array(np.gradient(V))
    #retunerer E og V
    return E, V

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

class StartNode:
    def __init__(self, r0, r1):
        angle = np.random.random()*2*np.pi

        self._x = r0*np.cos(angle)
        self._y = r0*np.sin(angle)

        self._angle = angle
        self.r0 = r0

        self._spin = np.random.random() * r0/100 - r0/200

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, angle):
        r0 = self.r0

        self._x = r0*np.cos(angle)
        self._y = r0*np.sin(angle)

        self._angle = angle

    @property
    def source(self):
        x, y = self._x, self._y
        return np.array([x, y])

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, spin):
        self._spin = spin

class Particle:
    speed = 0.5
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1
        angle = np.random.random()*2*np.pi
        r = np.random.randint(r0, r1)

        self.x, self.y = r*np.cos(angle), r*np.sin(angle)
        angle = np.random.random()*2*np.pi
        self.vx, self.vy = self.speed*np.cos(angle), self.speed*np.sin(angle)
        self.r, self.angle = r, angle

    @property
    def position(self):
        return self.x, self.y

    @position.setter
    def position(self, v):
        x, y = self.position
        self.r = np.sqrt(x**2 + y**2)

        vx, vy = v

        self.x += vx
        self.y += vy


    @property
    def velocity(self):
        return np.array([self.vx, self.vy])

    @velocity.setter
    def velocity(self, new_velocity):
        self.vx, self.vy = new_velocity

    @property
    def distanceCenter(self):
        return self.r

    @property
    def distanceEdge(self):
        return self.r1 - self.r

    def distanceFrom(self, x2, y2):
        x1, y1 = self.position
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

class PlasmaBall:
    maxBoltJump = np.sqrt(1200)
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def initStartNodes(self, n):
        r0, r1 = self.r0, self.r1
        startnodes = []
        for i in range(n):
            startnodes.append(StartNode(r0, r1))
        
        return startnodes

    def initParticles(self, n):
        r0, r1 = self.r0, self.r1
        particles = []
        for i in range(n):
            particles.append(Particle(r0, r1))
        
        return particles

    def initBall(self, nNodes, nParticles):
        self.startnodes = self.initStartNodes(nNodes)
        self.particles = self.initParticles(nParticles)

    def drawOuter(self):
        r1 = self.r1
        theta = np.linspace(0, 2*np.pi, 100)
        return plt.plot(r1*np.cos(theta), r1*np.sin(theta), c="black")

    def drawInner(self):
        r0 = self.r0
        theta = np.linspace(0, 2*np.pi, 100)
        return plt.plot(r0*np.cos(theta), r0*np.sin(theta), c="black")

    def drawBolt(self, node, particles):
        r0, r1 = self.r0, self.r1
        nearestParticle = 100000
        bolt = [node.source]
        oldx, oldy = node.source
        lastEdgeDistance = 150
        curEdgeDistance = r1 - r0
        while curEdgeDistance < lastEdgeDistance and curEdgeDistance > 1:
            nearestParticle = 100000
            for i, p in enumerate(particles):
                potentialDistanceEdge = p.distanceEdge
                if potentialDistanceEdge < curEdgeDistance:
                    potentialParticleDistance = p.distanceFrom(oldx, oldy)
                    if  potentialParticleDistance < nearestParticle:
                        nearestParticle = potentialParticleDistance
                        iPotential = i

            pnew = particles[iPotential]
            newx, newy = pnew.position
            newEdgeDistance = pnew.distanceEdge
            bolt.append(pnew.position)
            nearestParticle = p.distanceFrom(oldx, oldy)
            lastEdgeDistance, curEdgeDistance = curEdgeDistance, newEdgeDistance
            oldx, oldy = newx, newy
        bolt = np.array(bolt)
        return bolt

    def drawBall(self):
        outer = self.drawOuter()
        inner = self.drawInner()

        return inner, outer

    def getParticlesPosition(self):
        particles = self.particles
        positions = []
        for p in particles:
            positions.append(p.position)

        return np.array(positions)

    def drawParticles(self):
        positions = self.getParticlesPosition()
        return plt.scatter(positions[:, 0], positions[:, 1], s=0.2, c ="blue")

    def drawNodes(self):
        nodes = self.startnodes
        positions = []
        for node in nodes:
            positions.append(node.source)

        positions = np.array(positions)
        return plt.scatter(positions[:, 0], positions[:, 1])

    def setupGrid(self, L):
        z = np.zeros((L, L))
        b = np.copy(z)
        b[:] = np.float("nan")
        self.b = b

    @property
    def grid(self):
        return self.b

    def ontoGrid(self, b, positions, type):
        r1 = self.r1
        dx1 = 2*r1/len(b)
        #indexes = np.rint((positions - r1)/dx)
        #b[indexes] = 0
        if type == "outer" or type == "bolt":
            V = 0
        if type == "inner":
            V = 1

        ixOld, iyOld = np.rint((positions[0] + r1)/dx1) - 1
        ixOld, iyOld = int(ixOld), int(iyOld)
        for p in positions[1:]:
            ix, iy = np.rint((p + r1)/dx1) - 1
            ix, iy = int(ix), int(iy)
            b[ix, iy] = V
            dx, dy = ixOld - ix, iyOld - iy
            if dx != 0:
                for i in range(ixOld, ix, int(abs(ix - ixOld)/(ix - ixOld))):
                    b[i, iy] = V
            if dy != 0:
                for i in range(iyOld, iy, int(abs(iy - iyOld)/(iy - iyOld))):
                    b[ixOld, i] = V


            ixOld, iyOld = ix, iy
        return b
    
    def setupGridBoundries(self):
        r0, r1 = self.r0, self.r1
        theta = np.linspace(0, 2*np.pi, 100)
        inner = np.array([r0*np.cos(theta), r0*np.sin(theta)])
        outer = np.array([r1*np.cos(theta), r1*np.sin(theta)])
        self.b = self.ontoGrid(self.b, inner.T, "inner")
        self.b = self.ontoGrid(self.b, outer.T, "outer")

    def drawPotential(self, positions, nrep = 3000):
        bcopy = np.copy(self.grid)
        for bolt in positions:
            bcopy = self.ontoGrid(bcopy, bolt, "bolt")
        s = solvepoisson(bcopy, nrep)

        r1 = self.r1
        return plt.imshow(s.T[::-1], extent=(-r1, r1, -r1, r1))
    
    """def drawPotential(self, positions):
        Q = np.zeros(len(positions)) + 1
        x, y = self.grid

        E, V = findEfieldinapoint([x, y], Q, positions)
        return plt.contourf(x, y, V)"""
        
    def drawPlasma(self):
        startnodes = self.startnodes
        particles = self.particles
        plasma = []
        for i, node in enumerate(startnodes):
            plasma.append(self.drawBolt(node, particles))
            plt.plot(plasma[i][:, 0], plasma[i][:, 1], c="red")
        
        potential = self.drawPotential(plasma)
        inner, outer = self.drawBall()
        return inner, outer, plasma, potential

    def show(self):
        self.drawPlasma()
        self.drawParticles()
        self.drawNodes()
        plt.axis("equal")
        plt.axis("off")
        plt.show()
    
    def moveStartNodes(self, dt):
        startnodes = self.startnodes
        r0, r1 = self.r0, self.r1

        new_source = []
        for node in startnodes:
            node.angle += node.spin*dt
            node.spin = np.random.randint(-1,1) * r0/100 - r0/200
            if node.spin > 0.1:
                node.spin = 0.1
            elif node.spin > -0.1:
                node.spin = -0.1
            new_source.append(node.source)

        self.startnodes = startnodes

        return np.array(new_source)

    def moveParticles(self, dt):
        particles = self.particles
        r0, r1 = self.r0, self.r1

        new_posistion = []
        for p in particles:
            v = p.velocity
            p.position = v*dt
            new_posistion.append(p.position)
            if p.distanceCenter < r0 or p.distanceCenter > r1:
                p.velocity = -v
                p.position = v*dt

        self.particles = particles
        return np.array(new_posistion)

    def animate(self, i):
        dt = self.dt
        
        plt.gca().clear()
        plt.axis("off")
        plt.axis("equal")
        

        inner, outer, bolts, potential = self.drawPlasma()

        self.moveParticles(dt)
        nodes = self.moveStartNodes(dt)
        plt.scatter(nodes[:, 0], nodes[:, 1], c="orange")
        return inner, outer, bolts, potential,

    def initAnimation(self, dt, N, L):
        self.dt = dt
        self.setupGrid(L)
        self.setupGridBoundries()
        fig = plt.figure()
        plt.axis("off")
        plt.axis("equal")
        anim = animation.FuncAnimation(fig, self.animate, frames=N)
        anim.save('figures/Plasma_ball3.mp4', fps=100) #, writer="imagemagick")
        

if __name__ == "__main__":
    r0, r1 = 10, 20
    nNodes = 4
    nParticles = 350

    ball = PlasmaBall(r0, r1)
    ball.initBall(nNodes, nParticles)
    #ball.show()
    
    dt = 0.1
    N = 100
    L = 101
    ball.initAnimation(dt, N, L)
"""
    ball.setupGrid(101)
    ball.setupGridBoundries()
    plt.imshow(ball.grid)
    plt.show()
    ball.show()
    plt.imshow(ball.grid)
    plt.show()
    #"""
