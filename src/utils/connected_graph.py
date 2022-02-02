def DFSUtil(adj, temp, v, visited):
 
        # Mark the current vertex as visited
        visited[v] = True
 
        # Store the vertex to list
        temp.append(v)
 
        # Repeat for all vertices adjacent
        # to this vertex v
        for i in adj[v]:
            if visited[i] == False:
 
                # Update the list
                temp = DFSUtil(adj, temp, i, visited)
        return temp

def connectedComponents(adj, V):
        visited = []
        cc = [[] for i in range(V)]
        for i in range(V):
            visited.append(False)
        for v in range(V):
            if visited[v] == False:
                temp = []
                temp_states = DFSUtil(adj, temp, v, visited)
                for state in temp_states:
                    cc[state] = temp_states
        return cc

def addEdge(adj, v, w):
        adj[v].append(w)
        adj[w].append(v)

def findConnectedComp(graph, N_states):
    adj = [[] for i in range(N_states)]
    for link in graph:
        addEdge(adj, link[0], link[1])
        
    graph_connected = connectedComponents(adj,len(adj))
    return graph_connected
