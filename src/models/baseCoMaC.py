import numpy as np
import math
import sys
from scipy.sparse import csr_matrix


class baseCoMaC():
    
    def __init__(self, knns=20):
        
        self.knns=20
        
    def randomSampling(self, array, percentage=1, number_sample=None):
        '''
        Parameters
        ----------
        array : float array
            input array to be sampled
        percentage : float, optional
            percentage of samples to be selected randomly
        number_sample : int, optional
            number of samples to be selected randomly

        Returns
        -------
        array : float array
            random samples of the input array

        '''
        if number_sample is None:
            n_sample = math.ceil(array.shape[0]*percentage)
        else:
            n_sample = np.min((number_sample, array.shape[0]))

        index = np.sort(np.random.choice(array.shape[0], n_sample, replace=False))

        return array[index]
    
    def generateTransitionProbability(self,X):

        EDM = np.array([[np.linalg.norm(X[ind1, :] - X[ind2, :])**2
                        for ind1 in range(X.shape[0])]
                        for ind2 in range(X.shape[0])])
        dummy = np.sort(EDM, 0)
        sigma = 1 / np.mean(np.mean(dummy[0:self.knns, :], axis=0))
        P = np.exp(-sigma*EDM)
        # P, *ignore = np.linalg.lstsq( np.diag( np.sum(P,axis=1) ), P, rcond=None)
        P = np.linalg.solve(np.diag(np.sum(P, axis=1)), P)

        return P
    
    def partition_to_labels(self,V):
        temp_states = np.tile(np.arange(V.shape[1]), [V.shape[0], 1])
        labels = temp_states[V.astype(bool)]

        return labels
    
    def stationaryDistribution(self, P):
        n = P.shape[0]
        A = P.conj().T - np.eye(n)
        A[n-1, :] = np.ones((1, n))
        b = np.zeros((n, 1))
        b[n-1] = 1
        mu, *ignore = np.linalg.lstsq( A, b , rcond=None)
        
        return mu


    def jointDistribution(self, mu, P):
        P_stat = np.vstack([mu[i]*P[i, :] for i in range(P.shape[0])])

        return P_stat


    def mutualInformation(self, P):

        P_docs = np.sum(P, axis=1)
        P_words = np.sum(P, axis=0)

        # to avoid zero division, when taking the log it is zero anyways
        P_docs[P_docs == 0] = 1
        P_words[P_words == 0] = 1

        P_temp = (P/P_docs[np.newaxis].T)/P_words[np.newaxis]

        if np.any(P_temp < 0):
            a = 1
        Inf = P * np.log(P_temp, out=np.zeros_like(P_temp), where=(P_temp != 0))

        MI = np.sum(Inf) / np.log(2)

        return MI
    
    def pairwiseConstraints(self,TruePartition):
        
        '''
        Function to return matrices containing the pairwise constraints

        Parameters
        ----------
        TruePartition : array
            matrix containing the true clustering

        Returns
        -------
        M_must : float array
            matrix containing pairs in the same cluster
        M_cannot : float array
            matrix containing pairs in separate clusters

        '''
        M_must = np.empty((0, 2))    # initialization
        M_cannot = np.empty((0, 2))  # initialization
        for i in range(TruePartition.shape[0]):
            index = TruePartition[i, 1]   # get cluster index
            for j in range(i+1, TruePartition.shape[0]):
                if (TruePartition[j, 1] == index):
                    # all other samples which are in the same cluster
                    M_must = np.append(M_must, [[TruePartition[i, 0], TruePartition[j, 0]]], axis=0)
                else:
                    # the rest must be in a separate cluster
                    M_cannot = np.append(M_cannot, [[TruePartition[i, 0], TruePartition[j, 0]]], axis=0)

        return M_must, M_cannot
        