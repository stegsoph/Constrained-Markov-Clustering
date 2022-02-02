import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from models.baseCoMaC import baseCoMaC
# from source.utils.helperFunctions import randomSampling

class dataGen(baseCoMaC):

    def __init__(self, knns=20):
        
        super().__init__()
        self.knns = knns
        
    def generateCircles(self, M=3, N_vec=[60,60,60], R=[0.5,7,15], eps_vec=[0.3,0.3,0.3]):
        
        V_true = np.vstack([np.eye(M)[i, :] for i in range(M)
                            for j in range(N_vec[i])])
        X = np.vstack([[R[i]*np.cos(2*np.pi*phase), R[i]*np.sin(2*np.pi*phase)]
                       for i in range(M)
                       for phase in np.random.uniform(0, 1, N_vec[i])])
        N = np.vstack([eps_vec[i]*np.random.randn(N_vec[i], 2)
                       for i in range(M)])
        X = X + N

        P = self.generateTransitionProbability(X)

        return P, V_true, X
    
    def generateClouds(self, M=3, N_vec=[60,60,60], x=[-10,0,10], eps_vec=[2,2,2]):

        V_true = np.vstack([np.eye(M)[i, :]
                            for i in range(M) for j in range(N_vec[i])])
        X = np.array([[x[i], 0] for i in range(M) for j in range(N_vec[i])])
        N = np.vstack([eps_vec[i]*np.random.randn(N_vec[i], 2)
                       for i in range(M)])
        X = X + N

        P = self.generateTransitionProbability(X)

        return P, V_true, X

