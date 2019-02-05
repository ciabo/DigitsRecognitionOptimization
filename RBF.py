import numpy as np
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint

class RBF():
    def __init__(self, X, F):
        self.X = np.absolute(np.log(X))
        self.F = F
        # default epsilon is the "the average distance between nodes" based
        # on a bounding hypercube
        ximax = np.amax(self.X, axis=0)
        ximin = np.amin(self.X, axis=0)
        edges = ximax - ximin
        self.epsilon = np.power(np.prod(edges) / self.X.size, 1.0 / edges.size)

    def gaussian(self, xi, yi):
        r = np.sqrt(np.power((xi - yi), 2))  # norm of xi-yi
        return np.exp(-(1.0 / self.epsilon * r) ** 2)

    def interpolate(self):
        n = self.X.size
        self.phi = np.zeros((n, n), dtype=float)
        for i in range(0, n):
            for j in range(0, n):
                v = self.gaussian(self.X[i], self.X[j])
                self.phi[i][j] = v
        self.multipliers = np.linalg.solve(self.phi, self.F)

    def getMultipliers(self):
        return self.multipliers

    def newxGivenf(self, fvalue):
        #xcap = np.array([1])
        # res = minimize(self.bumpiness, x0, method='BFGS', options = {'disp': True})
        points = np.linspace(0.3, 10, 1000)
        results=np.array([])
        bounds = [(0.3, 10)]

        for point in points:
            res = minimize(self.bumpiness, point, fvalue, method='L-BFGS-B', bounds=bounds)
            results=np.append(results,res.x[0])
        return np.amin(results)

    def bumpiness(self, xcap, fvalue):
        return np.power(fvalue - self.s(xcap), 2) * self.g(xcap)

    def g(self, xcap):
        a = np.linalg.det(self.phi)
        phivec = np.zeros((1, self.X.size), dtype=float)
        for i in range(0, self.X.size):
            phivec[0][i] = self.gaussian(xcap, self.X[i])
        mat = np.c_[self.phi, np.transpose(phivec)]
        v = np.c_[phivec, 0]
        matrix = np.r_[mat, v]
        b = np.linalg.det(matrix)
        return np.divide(a, b)

    def s(self, xcap):
        sxcap = 0
        for i in range(0, self.X.size):
            sxcap = sxcap + self.multipliers[i] * self.gaussian(xcap, self.X[i])
        return sxcap

    def getX(self):
        return self.X

    '''
    def newxGivenfComplete(self,fvalue):
        cons=[]
        for i in range(0,self.X.size):
            con={'type':'eq', 'fun': con}
            cons.append()
        multipliers = np.ones((1,self.X.size), dtype=float)
        res = minimize(self.functionToMinimize, multipliers, fvalue, method='SLSQP', constraints=cons)
        return res.x[0]

    def functionToMinimize(self,multipliers):
        return np.dot(np.dot(np.transpose(multipliers),self.phi),multipliers)

    def constraint(self,xcap,fvalue):
        sxcap = 0
        for i in range(0, self.X.size):
            sxcap = sxcap + self.multipliers[i] * self.gaussian(xcap, self.X[i])
        sxcap=sxcap+self.multipliers[i] * self.gaussian(xcap, self.X[i])
        return sxcap
    '''