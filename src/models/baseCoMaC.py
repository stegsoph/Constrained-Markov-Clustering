import numpy as np
import math
import random
import sys
from scipy.sparse import csr_matrix
from itertools import product

from utils.k_coloring_greedy_v2 import create_graph
from utils.connected_graph import findConnectedComp


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
        
           
    def sample_from_k_classes(self, labels, labels_enum, percentage, k):
        
        # number of samples needed to achieve 30%
        number_sample_min30 = np.round(labels_enum.shape[0]*0.3).astype(int)
        number_sample = np.round(labels_enum.shape[0]*percentage).astype(int)
        # number of elements in each class
        sum_elements = np.array([np.sum(labels_enum[:,1] == i) for i in range(self.M)])
        # randomly choose class indices
        cluster_idx_sampling = np.unique(random.sample(range(self.M),
                                         k = k)).astype(int)
        N_max_iter = 0
        while (np.sum(sum_elements[cluster_idx_sampling]) < number_sample_min30):
            cluster_idx_sampling = np.unique(random.choices(range(self.M),
                                             k=k))
            N_max_iter += 1
            print('Repeat: not enough samples in the clusters')
            if N_max_iter > 20:
                # two largest clusters
                cluster_idx_sampling = np.argpartition(sum_elements, -k)[-k:]
                print("Stopped: not enough samples in the clusters")
                break

        labels_idx = np.array([(labels == x) for x in cluster_idx_sampling]).any(axis=0)
        labels_temp = labels_enum[labels_idx, :]
        labels_red_enum = self.randomSampling(labels_temp, number_sample=number_sample)
        return labels_red_enum
    
    def gen_rand_constraints(self, labels, percentage=1, number_sample=None):
        '''
        Randomly select pairs of data points - based on their class label, 
        they generate cannot (different labels) or must (same label) constraints
        '''

        N_samples = len(labels)
        l1, l2 = np.arange(N_samples), np.arange(N_samples)
        output = list(product(l1, l2))
        output = np.array(output)

        delete_rows =  []
        for i in range(output.shape[0]): 
            if output[i][0]==output[i][1]:
                delete_rows.append(i)
        output = np.delete(output, delete_rows, axis=0)
        
        if number_sample == None:
            number_sample = np.round(len(labels)*percentage).astype(int)
        output_sampled = self.randomSampling(output, number_sample=number_sample)

        M_must = np.empty((0, 2))    # initialization
        M_cannot = np.empty((0, 2))  # initialization

        for i in range(output_sampled.shape[0]):
            idx_0 = output_sampled[i][0]
            idx_1 = output_sampled[i][1]
            label_0 = labels[idx_0]   # get cluster index
            label_1 = labels[idx_1]   # get cluster index
            if label_0 == label_1:
                M_must = np.append(M_must, [[idx_0, idx_1]], axis=0)
            else:
                M_cannot = np.append(M_cannot, [[idx_0, idx_1]], axis=0)

        return M_must, M_cannot
    
    def constraint_graph(self, X, M_must, M_cannot):
        #-------------------------------------------------------------------------------------
        # Initialization if we do or do not have cannot-link-constraints
        
        if M_cannot is None:
            adj_cannot = [[]]*len(X)  # empty list of length A (number of samples)
        else:
            # create graph fullfilling the constraints
            adj_cannot = create_graph(M_cannot, len(X))

        #-------------------------------------------------------------------------------------
        # Initialization if we do or do not have must-link-constraints
        
        if M_must is None:
            # list where row i has value i
            adj_must = [[i] for i in range(len(X))]
        else:
            # create graph fullfilling the constraints
            temp_list_graph = create_graph(M_must, len(X))
            # list where row i has value i
            temp_list = [[i] for i in range(len(X))]
            # concatenate both lists
            adj_must = [a + b for a, b in zip(temp_list, temp_list_graph)]
            
        self.adj_must = adj_must
        self.adj_cannot = adj_cannot
        self.M_must = M_must
        self.M_cannot = M_cannot
    
    def random_constraints(self, X, labels, percentage=1, number_sample=None,
                           flag_no_cannot=False, flag_connected=True):
        
        (M_must, M_cannot) = self.gen_rand_constraints(labels, percentage=percentage, number_sample=number_sample)
        M_must = M_must.astype(int)
        M_cannot = M_cannot.astype(int)
        if flag_no_cannot:
            M_cannot = []           
            
        self.constraint_graph(X, M_must, M_cannot)
        
        if flag_connected:
            adj_must_connected = findConnectedComp(M_must.astype(int), len(X))
            added_elem = np.sum( [[-len(self.adj_must[i]), len(adj_must_connected[i])] for i in range(len(X))]  )
            print('Number of additional constraints due to constraint propagation: ', added_elem)
            self.adj_must = adj_must_connected
        
    def constraints(self, X, labels, percentage=1, number_sample=None,
                    wrong_percentage=0, ClassLabels='All'):
        
        labels_enum = np.array([np.arange(X.shape[0]), labels]).T

        #-------------------------------------------------------------------------------------
        # Randomly sample from the labels
        
        if ClassLabels == 'All':
            labels_red_enum = self.randomSampling(labels_enum, percentage=percentage, number_sample=number_sample)
        else: 
            labels_red_enum = self.sample_from_k_classes(labels, labels_enum, percentage, ClassLabels)
               
        #-------------------------------------------------------------------------------------
        # Add some wrong labels
        
        labels_wrong = self.randomSampling(labels_red_enum, percentage=wrong_percentage)
        
        index_list = np.arange(self.M)
        for i in range(len(labels_wrong)):
            K = labels_wrong[i, 1] # true label for data point x_i
            # select another label which is wrong
            res = random.choice([ele for ele in index_list if ele != K])
            # overwrite the true label with the wrong one
            labels_red_enum[labels_red_enum[:, 0] == labels_wrong[i, 0], 1] = res
                    
        #-------------------------------------------------------------------------------------
        # Generate the pairwise constraints
        
        M_must, M_cannot = self.pairwiseConstraints(labels_red_enum)
        
        self.constraint_graph(X, M_must, M_cannot)
        