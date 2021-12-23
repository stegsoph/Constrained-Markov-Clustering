from models.baseCoMaC import baseCoMaC
from utils.k_coloring_greedy_v1 import (possible_partition_greedy,checkConstraints)

import numpy as np
import math
from scipy.sparse import csr_matrix
import sys


class CoMaC(baseCoMaC):
    
    def __init__(self, knns=20, M=3, restarts=5):
        
        super().__init__()
        self.knns = knns
        self.M = M
        self.restarts = restarts
        
        
    def cluster_seq(self, X, beta=0.5, g_init=None, max_iter=25):
                
        A = self.generateTransitionProbability(X)
        # -------------------------------------------------------------------------
        # parameters:
        factor_1 = 1-2*beta
        factor_2 = 1-beta
        n = A.shape[1]  # number of samples
        # -------------------------------------------------------------------------

        # preparing
        if np.abs(np.sum(A) - 1) < 1e-9:
            P = A
        else:
            mu = self.stationaryDistribution(A)
            P = self.jointDistribution(mu, A)

        MI = self.mutualInformation(P)

        # Loop over restarts
        runningMin = -math.inf

        # -------------------------------------------------------------------------
        for cnt in range(self.restarts):
            #print('~'*25+f'Restart {cnt}'+'~'*25)

            if g_init is not None:
                # if initial partition is given
                V = g_init
                # first, check if this partition fullfills all requirements
                is_okay = checkConstraints(self.adj_must, self.adj_cannot, V)
                if ~is_okay:  # if there is an error
                    print("V initialized with ERROR")
                    is_okay = checkConstraints(self.adj_must, self.adj_cannot, V)
            else:
                # no initial partition: create it from cannot-&-must-links
                # result: 1d array with values in [0 to M-1]
                # adj_cannot_temp = create_graph(M_cannot, len(A))
                result = possible_partition_greedy(self.adj_must, self.adj_cannot, len(A), self.M)
                # V: 2d array with dimensions nxM with values in {0,1}
                V = np.array([[1 if i == result[j] else 0 for i in range(self.M)]
                              for j in range(len(A))])

                # again check if this partition fullfills all requirements
                # can be omitted, just if there is an error in the code...
                is_okay = checkConstraints(self.adj_must, self.adj_cannot, V)
                if ~is_okay:
                    print("V initialized with ERROR")
                    is_okay = checkConstraints(self.adj_must, self.adj_cannot, V)

            # Loop over iterations
            k = 1
            tol = 1
            old_cost = -math.inf

            # ---------------------------------------------------------------------
            while tol > 0 and k < max_iter:
                #print('Iteration {:.0f}'.format(k))

                ClusterCost = np.zeros((self.M, 1))

                # -----------------------------------------------------------------
                # temp = np.arange(n)
                # np.random.shuffle(temp)
                # temp = np.hstack((np.ravel(M_cannot), temp))
                # _, idx = np.unique(temp, return_index=True)
                # list_iterations = temp[np.sort(idx)].astype(int)

                for i in range(n): 
                    
                    # indices that have to be in the same cluster
                    index_must = np.array(self.adj_must[i]).astype(int)

                    # indices that cannot be in the same cluster
                    # also check all indices in index_must, for the case that we
                    # do not have all pairwise constraints, but only a subset
                    #    e.g. 1,2,3 in the same cluster and cannot be with 7
                    #         but due to random sampling some pairs are not given
                    #         3 / 7 is given, but 1 / 7 or 2 / 7 is not given
                    # this can occur when we generate all pairwise constraints and
                    # select random samples from there
                    index_cannot = []
                    for idx_connected in index_must:
                        index_cannot = np.append(index_cannot,
                                                 self.adj_cannot[idx_connected])
                    index_cannot = np.unique(index_cannot)

                    # indices that cannot be in the same cluster
                    index_cannot = index_cannot.astype(int)

                    # check in which cluster the other samples are located and
                    # look for "free" clusters
                    # array with unit vectors of length M
                    index_states = np.all([np.full(self.M, True, dtype=bool),
                                           ~np.any(V[index_cannot, :], axis=0)],
                                          axis=0)
                    # create array with elements in [0 to M-1]
                    temp_states = np.arange(self.M)
                    vector_states = temp_states[index_states]

                    # if there are no "free" clusters, select the one with the
                    # least contradiction and print an error message
                    if ~index_states.any():
                        print("Cannot-link-constraint not satisfied")
                        number_cannot = np.sum(V[index_cannot, :], axis=0)
                        vector_states = np.where(number_cannot ==
                                                  np.min(number_cannot))[0]
                        # vector_states = np.arange(self.M)

                    # set the row of the index and all connected rows with
                    # must-link-constraints to zero
                    V[index_must, :] = np.zeros((1, self.M))

                    ClusterCost = np.zeros((self.M, 1)) - math.inf

                    # iterate only over "free" cluster states
                    for j in vector_states:
                        V[index_must, j] = 1
                        
                        Q_V = csr_matrix(V)
                        Ponered = csr_matrix.dot(P, Q_V)
                        Ptwored = csr_matrix.dot(Q_V.T,Ponered)
                        '''
                        Ponered = np.dot(P, V)
                        Ptwored = np.dot(V.T, Ponered)
                        '''
                        I1 = self.mutualInformation(Ptwored)
                        I3 = self.mutualInformation(Ponered)

                        ClusterCost[j] = -(factor_1*I3 - factor_2*I1)
                        V[index_must, j] = 0

                    # compute where the loss is the lowest and set this state to 1
                    index = np.argmax(ClusterCost)
                    V[index_must, index] = 1

                # -----------------------------------------------------------------
                
                Q_V = csr_matrix(V)
                Ponered = csr_matrix.dot(P, Q_V)
                Ptwored = csr_matrix.dot(Q_V.T,Ponered)
                '''
                Ponered = np.dot(P, V)
                Ptwored = np.dot(V.T, Ponered)
                '''
                I1 = self.mutualInformation(Ptwored)
                I3 = self.mutualInformation(Ponered)
                new_cost = - (factor_1*I3 - factor_2*I1)
                tol = np.abs(new_cost - old_cost) / np.abs(new_cost)
                old_cost = new_cost
                k += 1
            # ---------------------------------------------------------------------

            if new_cost > runningMin:
                runningMin = new_cost
                V_best = V

        V = V_best
        cost = beta*MI - runningMin
        
        print('*'*25+'Clustering Finished!'+'*'*25)

        return cost, V
    
    
    def cluster_ann(self, X, final_beta=0, step_size=-0.5, max_iter=25):
                        
        frac = (final_beta-1) / step_size
        beta = np.arange(frac, -sys.float_info.min, -1) / frac
        if (beta[-1] != final_beta):
            beta = np.append(beta, final_beta)

        n = len(X)
        cost = np.zeros((len(beta), 1))
        V_ann = np.zeros((len(beta), n, self.M))

        print(f'Starting with beta = 1')
        (cost_temp, V_temp) = self.cluster_seq(X, beta=1, max_iter=max_iter)
        cost[0] = cost_temp
        V_ann[0, :, :] = V_temp

        for i in range(1, len(beta)):
            print(f'Starting with beta = {beta[i]}')
            (cost_temp, V_temp) = self.cluster_seq(X, beta=beta[i], max_iter=max_iter)
            cost[i] = cost_temp
            V_ann[i, :, :] = V_temp
            
        return (cost, V_ann, beta)
    
