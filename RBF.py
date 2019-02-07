import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

class RBF():
    def __init__(self, X, F):
        self.X = np.log10(X)
        self.F = F
        self.multipliers = np.zeros((1, self.X.size))
        self.phi = np.zeros((self.X.size, self.X.size), dtype=float)
        # default epsilon is the "the average distance between nodes" based
        # on a bounding hypercube
        ximax = np.amax(self.X, axis=0)
        ximin = np.amin(self.X, axis=0)
        edges = ximax - ximin
        self.epsilon = np.power(np.prod(edges) / self.X.size, 1.0 / edges.size)

    def gaussian(self, xi, yi):
        r = np.sqrt(np.power((xi - yi), 2))  # norm of xi-yi
        return np.exp(-1.0 / self.epsilon * np.power(r, 2))

    def interpolate(self):
        n = self.X.size
        # create phi matrix
        for i in range(0, n):
            for j in range(0, n):
                v = self.gaussian(self.X[i], self.X[j])
                self.phi[i][j] = v
        # find multipliers solving the system
        self.multipliers = np.linalg.solve(self.phi, self.F)

    def getMultipliers(self):
        return self.multipliers

    def minimizeInterpolant(self):
        points = np.linspace(-6, -0.3, 500)
        results = np.array([])
        bounds = [(-6, -0.3)]
        for point in points:
            # minimize the bumpiness
            res = minimize(self.s, point, method='L-BFGS-B', bounds=bounds)
            results = np.append(results, res.fun[0])
        # return the min value of all the optimization
        best = np.amin(results)
        return best-1


    def newxGivenf(self, explore=False):
        fvalue=-100000
        if explore is False:
            fvalue=self.minimizeInterpolant()
        # create points to to start the minimization
        points = np.linspace(-6, -0.3, 500)
        results = np.array([])
        xvalues = np.array([])
        bounds = [(-6, -0.3)]
        for point in points:
            # minimize the bumpiness
            res = minimize(self.bumpiness, point, fvalue, method='L-BFGS-B', bounds=bounds)
            results = np.append(results, res.fun[0])
            xvalues = np.append(xvalues, res.x[0])
        # return the min value of all the optimization
        bestIndex=np.where(results == np.amin(results))
        idx=np.take(bestIndex,0,axis=0)[0]
        return xvalues[idx]

    def bumpiness(self, xcap, fvalue):
        return np.power(fvalue - self.s(xcap), 2) * self.g(xcap)

    def g(self, xcap):
        # nominator determinant
        a = np.linalg.det(self.phi)
        phivec = np.zeros((1, self.X.size), dtype=float)
        for i in range(0, self.X.size):
            phivec[0][i] = self.gaussian(xcap, self.X[i])
        mat = np.c_[self.phi, np.transpose(phivec)]
        v = np.c_[phivec, 0]
        matrix = np.r_[mat, v]
        # denominator determinant
        b = np.linalg.det(matrix)
        return np.divide(a, b)

    # return the value of the interpolant for a given xcap
    def s(self, xcap):
        sxcap = 0
        for i in range(0, self.X.size):
            sxcap = sxcap + self.multipliers[i] * self.gaussian(xcap, self.X[i])
        return sxcap

    def getX(self):
        return self.X

    def plotBumpiness(self):
        xnew = np.linspace(-6, -0.3, 1000)
        fval = []
        for i in range(1000):
            fval.append(self.bumpiness(xnew[i],0))
        plt.figure(3)
        #plt.scatter(self.X, self.F, c='r', marker='o')
        plt.plot(xnew, fval, label="bumpiness")
        plt.title('Bumpiness')
        plt.legend()
        plt.show()
