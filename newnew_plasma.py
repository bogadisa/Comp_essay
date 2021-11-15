import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from numpy.random import f
plt.style.use('seaborn-pastel')

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
        plt.plot(bolt[:, 0], bolt[:, 1], c="red")

    def drawBall(self):
        outer = self.drawOuter()
        inner = self.drawInner()

        return inner, outer

    def drawParticles(self):
        particles = self.particles
        positions = []
        for p in particles:
            positions.append(p.position)

        positions = np.array(positions)
        return plt.scatter(positions[:, 0], positions[:, 1], s=0.2, c ="blue")

    def drawNodes(self):
        nodes = self.startnodes
        positions = []
        for node in nodes:
            positions.append(node.source)

        positions = np.array(positions)
        return plt.scatter(positions[:, 0], positions[:, 1])
        
    def drawPlasma(self):
        startnodes = self.startnodes
        particles = self.particles
        plasma = []
        for node in startnodes:
            plasma.append(self.drawBolt(node, particles))

        inner, outer = self.drawBall()
        return inner, outer, plasma

    def show(self):
        self.drawPlasma()
        self.drawParticles()
        self.drawNodes()
        plt.axis("equal")
        plt.axis("off")
        plt.show()

    def moveStartnodes(self, dt):
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

        self.startnodes = startnodes

        return np.array(new_source)
    
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
        

        inner, outer, bolts = self.drawPlasma()

        self.moveParticles(dt)
        nodes = self.moveStartnodes(dt)
        plt.scatter(nodes[:, 0], nodes[:, 1], c="orange")
        return inner, outer, bolts,

    def initAnimation(self, dt, N):
        self.dt = dt
        fig = plt.figure()
        plt.axis("off")
        plt.axis("equal")
        anim = animation.FuncAnimation(fig, self.animate, frames=N)
        anim.save('figures/Plasma_ball2.mp4', fps=100)

if __name__ == "__main__":
    r0, r1 = 10, 150
    nNodes = 10
    nParticles = 400

    ball = PlasmaBall(r0, r1)
    ball.initBall(nNodes, nParticles)
    #ball.show()
    
    dt = 0.1
    N = 1000
    ball.initAnimation(dt, N)