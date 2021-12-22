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


    def gen_rand_constraints(labels_true, percentage=1, number_sample=None):

        N_samples = len(labels_true)
        l1, l2 = np.arange(N_samples), np.arange(N_samples)
        output = list(product(l1, l2))
        output = np.array(output)

        delete_rows =  []
        for i in range(output.shape[0]): 
            if output[i][0]==output[i][1]:
                delete_rows.append(i)
        output = np.delete(output, delete_rows, axis=0)
        output_sampled = randomSampling(output, percentage=percentage, number_sample=number_sample)

        M_must = np.empty((0, 2))    # initialization
        M_cannot = np.empty((0, 2))  # initialization

        for i in range(output_sampled.shape[0]):
            idx_0 = output_sampled[i][0]
            idx_1 = output_sampled[i][1]
            label_0 = labels_true[idx_0]   # get cluster index
            label_1 = labels_true[idx_1]   # get cluster index
            if label_0 == label_1:
                M_must = np.append(M_must, [[idx_0, idx_1]], axis=0)
            else:
                M_cannot = np.append(M_cannot, [[idx_0, idx_1]], axis=0)

        return M_must, M_cannot