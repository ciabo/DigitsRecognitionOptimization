import numpy as np

class RBF():
    def __init__(self,X,F):
        self.X=X
        self.F=F
        # default epsilon is the "the average distance between nodes" based
        # on a bounding hypercube
        ximax = np.amax(self.X, axis=0)
        ximin = np.amin(self.X, axis=0)
        edges = ximax - ximin
        self.epsilon = np.power(np.prod(edges) / self.X.size, 1.0 / edges.size)


    def gaussian(self,xi,yi):
        r=np.sqrt(np.power( (xi-yi) , 2))#norm of xi-yi
        return np.exp(-(1.0 / self.epsilon * r) ** 2)

    def interpolate(self):
        n = self.X.size
        self.phi = np.zeros((n,n),dtype=float)
        for i in range(0,n):
            for j in range(0,n):
                v=self.gaussian(self.X[i],self.X[j])
                self.phi[i][j] = v
        self.multipliers = np.linalg.solve(self.phi,self.F)

    def getMultipliers(self):
        return self.multipliers

    def newxGivenf(self,fvalue):
        xcap=0 #value we want to minimize

        np.power(fvalue-self.s(xcap) , 2)*self.g(xcap)
    def g(self,xcap):
        a=np.linalg.det(self.phi)

    def s(self,xcap):
        for i in range(0,self.X.size):
            sxcap=self.multipliers[i]*self.gaussian(xcap,self.X[i])
        return sxcap