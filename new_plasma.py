import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
plt.style.use('seaborn-pastel')

class StartNode:
    def __init__(self, r0):
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

    #litt usikker om denne trengs
    """ 
    @source.setter
    def source(self, new_source):
        r0 = self.r0
        x_new, y_new = new_source
        old_angle = self.angle
        angle = np.arccos(x_new/r0)
        tol = 1e-10
        if abs(old_angle - angle) >= tol:
            angle = -angle

        self.angle = angle

        self._x, self._y = x_new, y_new
    """

    @property
    def spin(self):
        return self._spin

    @spin.setter
    def spin(self, spin):
        self._spin = spin

class Particle:
    speed = 0.5
    def __init__(self, r0, r1):
        angle = np.random.random()*2*np.pi
        r = np.random.randint(r0, r1)
        self.x, self.y = r*np.cos(angle), r*np.sin(angle)
        self.vx, self.vy = np.cos(angle)*self.speed, np.sin(angle)*self.speed

    
    @property
    def charge(self):
        return -1

    @property
    def position(self):
        return np.array([self.x, self.y])

    @property
    def velocity(self):
        return np.array([self.vx, self.vy])

    @velocity.setter
    def velocity(self, v_new):
        vx_new, vy_new = v_new
        self.vx, self.vy = vx_new, vy_new

    @position.setter
    def position(self, new_position):
        self.x, self.y = new_position

    def distanceParticles(self, other):
        x1, y1 = self.position
        if isinstance(other, Particle):
            x2, y2 = other.position
        
        else:
            x2, y2 = other

        return (x1 - x2)**2 + (y1 - y2)**2

    @property
    def distanceCenter(self):
        x, y = self.position
        return x**2 + y**2


def findEfieldinapoint(r, Q, R):
    x, y = r
    V = np.zeros(np.shape(x))
    for i in range(len(x.flat)):
        r = np.array([x.flat[i], y.flat[i]])
        V.flat[i] += sum(Q[j]/np.linalg.norm(r - R[j]) for j in range(len(Q)))
    E = -np.array(np.gradient(V))
    return E, V


class PlasmaBall:
    maxBoltJump = np.sqrt(10000)
    def __init__(self, r0, r1):
        self.r0, self.r1 = r0, r1

    def moveStartNodes(self, dt):
        startnodes = self.startnodes
        r0, r1 = self.r0, self.r1

        new_source = []
        for node in startnodes:
            node.angle += node.spin*dt
            node.spin = np.random.random() * r0/100 - r0/200
            if node.spin > 0.1:
                node.spin = 0.1
            elif node.spin > -0.1:
                node.spin = -0.1
            new_source.append(node.source)

        return np.array(new_source)

    def moveParticles(self, dt):
        particles = self.particles
        r0, r1 = self.r0, self.r1

        new_position = []
        for particle in particles:
            v = particle.velocity
            particle.position += v*dt

            x, y = particle.position

            if (x**2 + y**2) < r0 or (x**2 + y**2) > r1:
                particle.velocity = -v
            
            new_position.append(particle.position)

        return np.array(new_position)

    def drawCenter(self):
        r0 = self.r0
        theta = np.linspace(0, 2*np.pi, 100)

        center = plt.plot(r0*np.cos(theta), r0*np.sin(theta), c="black")
        return center

    def drawOuter(self):
        r1 = self.r1
        theta = np.linspace(0, 2*np.pi, 100)

        outer = plt.plot(r1*np.cos(theta), r1*np.sin(theta), c="blue")
        return outer

    def distanceEdge(self, particle):
        x, y = particle.position
        r1 = self.r1

        dist = 0

        if (x**2 + y**2) == r1:
            return dist

        else:
            return abs(x**2 + y**2 - r1)

    def distanceCenter(self, x, y):
        r1 = self.r1
        return np.sqrt((r1 - x)**2 + (r1 - y)**2)

    def checkIntersectCenter(self, X, Y):
        x1, x2 = X
        y1, y2 = Y
        r0, r1 = self.r0, self.r1

        dX = x2 - x1
        dY = y2 - y1

        if dX == 0 and dY == 0:
            return False
        
        dl = dX**2 + dY**2
        t = ((r1 - x1) * dX + (r1 - y1) * dY)/dl

        nearestX = x1 + t*dX
        nearestY = y1 + t*dY

        dist = self.distanceCenter(nearestX, nearestY)

        if dist <= r0:
            return True
        else:
            return False



    def drawBolt(self, startnode, particles):
        r0, r1 = self.r0, self.r1
        path = []
        boundryFound = False
        x, y = startnode.source
        #path.append(np.array([x, y]))
        start = True
        distanceCenter = 0
        distanceEdge = 10000000000
        distanceEdgeCur = 0
        distanceEdgeLast = distanceEdge
        """while True:
            lowestParticleDist = 1000000
            boundryFound = False

            for i, p in enumerate(particles):
                dist = p.distanceParticles([x, y])

                if dist < lowestParticleDist:
                    x1, y1 = p.position
                    distanceEdgeCur = self.distanceEdge(p)
                    if distanceEdgeCur <= distanceEdgeLast:
                        continue
                    if distanceEdgeCur > r1:
                        continue
                    if dist > self.maxBoltJump:
                        continue
                    if dist <= r0:
                        continue
                    if distanceEdgeCur < r1/r0:
                        if self.checkIntersectCenter([x, x1], [y, y1]):
                            continue

                    lowestParticleDist = dist
                    distanceEdge = distanceEdgeCur
                    nextParticle = p
                    nextIndex = i
                    boundryFound = True
            
            if not(boundryFound):
                break
            
            x, y = nextParticle.position
            path.append(np.array([x, y]))
            self.particles.pop(nextIndex)
            distanceEdgeLast = distanceEdge
        
        path = np.array(path)
        x, y = path[:, 0], path[:, 1]

        plt.plot(x, y, c="red")"""


        while not(boundryFound):
            lowestParticleDist = 1000000
            for i, p in enumerate(particles):
                x1, y1 = p.position
                if start:
                    dist = p.distanceParticles([x, y])
                    if dist < self.maxBoltJump and dist < lowestParticleDist and not(self.checkIntersectCenter([x, x1], [y, y1])):
                        start = False
                        lowestParticleDist = dist
                        path.append(np.array([x, y]))
                        x, y = x1, y1
                        self.particles.pop(i)
                        distanceCenter = p.distanceCenter
                        distanceEdge = self.distanceEdge(p)
                
                elif not(start) and distanceCenter < p.distanceCenter:
                    dist = p.distanceParticles([x, y])
                    if dist < self.maxBoltJump and dist < lowestParticleDist and not(self.checkIntersectCenter([x, x1], [y, y1])):
                        lowestParticleDist = dist
                        path.append(np.array([x, y]))
                        x, y = x1, y1
                        self.particles.pop(i)
                        distanceCenter = p.distanceCenter
                        distanceEdge = self.distanceEdge(p)
                
                else:
                    continue
            
            if distanceEdgeLast == distanceEdge:
                boundryFound = True
            else:
                distanceEdgeLast = distanceEdge
        
        path = np.array(path)
        x, y = path[:, 0], path[:, 1]

        bolt = plt.plot(x, y, c="red")
        return bolt

    def drawBall(self, animation = False):
        startnodes, particles = self.startnodes, self.particles
        bolts = []
        for startnode in startnodes:
            startnodes, particles = self.startnodes, self.particles
            bolts.append(self.drawBolt(startnode, particles))

        center = self.drawCenter()
        outer = self.drawOuter()
        if animation:
            return bolts, center, outer

    def initParticles(self, nParticles):
        r0, r1 = self.r0, self.r1
        particles = []
        for i in range(nParticles):
            particles.append(Particle(r0, r1))
        
        return particles
        

    def initStartNodes(self, nStartNodes):
        r0 = self.r0
        startnodes = []
        for i in range(nStartNodes):
            startnodes.append(StartNode(r0))

        return startnodes

    def initPlasma(self, nParticles, nStartnodes):
        self.startnodes = self.initStartNodes(nStartnodes)
        self.particles = self.initParticles(nParticles)

        self.drawBall(animation = False)
        plt.axis("off")
        plt.axis("equal")
        plt.show()

    def animate(self, i):
        plt.gca().clear()
        dt = self.dt
        bolts, center, outer = self.drawBall(animation=True)
        self.moveParticles(dt)
        self.moveStartNodes(dt)

        return bolts[0], center, outer,
        



    def initAnimation(self, nParticles, nStartnodes, dt, N):
        self.startnodes = self.initStartNodes(nStartnodes)
        self.particles = self.initParticles(nParticles)
        self.dt = dt

        fig = plt.figure()
        anim = animation.FuncAnimation(fig, self.animate, frames=N)
        plt.axis("off")
        plt.axis("equal")
        anim.save('figures/Plasma_ball.gif', writer='imagemagick')



if __name__ == "__main__":
    r0 = 10
    r1 = 150
    plasmaball = PlasmaBall(r0, r1)

    nParticles = 10000
    nStartNodes = 4  
    plasmaball.initPlasma(nParticles, nStartNodes)  

    dt = 0.001
    N = 100 
    plasmaball.initAnimation(nParticles, nStartNodes, dt, N)        









#når vi kommer til animering
#start på en partikkel, se rundt den og finn nærmeste partikkel,
#fortsett sånn til kanten er nærmere enn en partikkel.
#Deretter interpoler greier, slik at du får en kurve som går igjennom alle de punktene