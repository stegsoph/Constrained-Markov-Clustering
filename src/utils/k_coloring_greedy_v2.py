import numpy as np
import random
# np.random.seed(2)

# ----------------------------------------------------------------------------
# This code is contributed by mohit kumar 29

# Python3 program to implement greedy
# algorithm for graph coloring


def addEdge(adj, v, w):

    adj[v].append(w)

    # Note: the graph is undirected
    adj[w].append(v)
    return adj

# Assigns colors (starting from 0) to all
# vertices and prints the assignment of colors


def greedyColoring(adj_must, adj, V, M):

    result = [-1] * V

    # Assign the first color to first vertex
    result[0] = 0

    # A temporary array to store the available colors.
    # True value of available[cr] would mean that the
    # color cr is assigned to one of its adjacent vertices
    
    available = [False] * V
    # Assign colors to remaining V-1 vertices
    for u in range(1, V):
        
        idx_cannot = []
        idx_cannot = [adj[i] for i in adj_must[u]]

        states_temp = []
        # Process all adjacent vertices and
        # flag their colors as unavailable
        for i_can in idx_cannot:
            for i in i_can:
                states_temp.append(result[i])
                if (result[i] != -1):
                    available[result[i]] = True
         
        ### Variante 2: ###
        # all the states of points connected with cannot-link constraints
        states_temp = [x for x in states_temp if x >= 0]
        states_vec = np.arange(0,M)
        # random.shuffle(states_vec)

        # find first empty state
        cr = -1
        for state in states_vec:
            if state not in states_temp:
                cr = state
                break
            
        # if no state is available, select the one with least frequency
        if cr == -1:
            print("Coloring with M colors not possible")
            cr = np.bincount(states_temp).argmin()
                    
        # ## Variante 1: ###
        # # Find the first available color

        # temp_list = []
        # for i in range(len(adj_must)):
        #     temp_list.append(sorted(adj_must[i]))
        # temp_list = np.array(temp_list, dtype=object)
        # temp_list = np.unique(temp_list)

        # cr = 0
        # ## Variante 1.1
        # # while cr < V:
        # ## Variante 1.2
        # for idx in temp_list:
        #     if isinstance(idx, list):
        #         cr = idx[0]
        #     else:
        #         cr = idx
        #     if (available[cr] is False):
        #         break
        #     cr += 1

        # Assign the found color
        for i in np.array(adj_must[u]).astype(int):
            result[i] = cr

        # to obtain some randomness when it does not matter
        random_num = np.random.randint(M)
        index_must = np.array(adj_must[u]).astype(int)

        flag_empty = True
        
        for idx_must in index_must:
            if len(np.array(adj, dtype=object)[idx_must]) != 0:
                flag_empty = False

        if flag_empty:
            for i in np.array(adj_must[u]).astype(int):
                result[i] = random_num

        # Reset the values back to false
        # for the next iteration
        for i_can in idx_cannot:
            for i in i_can:
                if (result[i] != -1):
                    available[result[i]] = False

    return result


def create_graph(M_graph, N_node):
    '''
    creates a list from an array with constraints

    Parameters
    ----------
    M_graph: array
        contains pairwise constraints, not necessarily all pairwise constraints
    N_node: int
        total number of samples

    Returns
    -------
    g1: list
        contains information given in M_graph but in a list
    '''
    g1 = [[] for i in range(N_node)]
    for i in range(len(M_graph)):
        g1 = addEdge(g1, M_graph[i, 0].astype(int), M_graph[i, 1].astype(int))

    return g1


def checkConstraints(g1_must, g1_cannot, V):
    '''
    checks if a certain partition fulfills all must-&-cannot-link constraints

    Parameters
    ----------
    g1_must: list
        must-link constraits
    g1_cannot: list
        cannot-link constraits
    V: array
        partition to be analyzed

    Returns
    -------
    is_okay: bool
        True: everything is okay, False: error in the graph
    '''
    for u in range(len(V)):
        i_must = np.array(g1_must[u]).astype(int)
        is_okay_must = np.any(np.all(V[i_must, :], axis=0))
        is_okay_1 = True
        is_okay_2 = True

        if ~is_okay_must:
            # print("Must-link constraint not fulfilled")
            is_okay_1 = False

        is_okay_cannot = np.all(np.any(V[i_must, :], axis=0))
        if is_okay_cannot:
            # print("Cannot-link constraint not fulfilled")
            is_okay_2 = False

        is_okay = np.all((is_okay_1, is_okay_2))

    return is_okay


def possible_partition_greedy(g1_must, g1_cannot, N_node, M):
    '''
    creates a possible partition given some constraints

    Parameters
    ----------
    g1_must: list
        must-link constraits
    g1_cannot: list
        cannot-link constraits
    N_node: int
        total number of samples
    M: int
        number of clusters

    Returns
    -------
    result_M: array
        possible partition
    '''
    result = greedyColoring(g1_must, g1_cannot, N_node, M)

    result_M = np.array(result)

    # if the partition is not possible with M clusters, give a warning and
    # assign the states >= M with a random state in [0,M-1]
    if np.any(result_M >= M):
        print("Coloring with M colors not possible")
        result_M[result_M >= M] = np.random.randint(M)

    return result_M
