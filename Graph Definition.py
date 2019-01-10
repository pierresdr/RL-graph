# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 16:09:23 2019

@author: ekgia
"""

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt



class DAG:
    def __init__(self, N_nodes=10, max_weight=100):
        self.N_nodes = N_nodes
        self.max_weight = max_weight
        self.DAG = self.generate_DAG(N_nodes, max_weight)
    
    
    '''Implementing the general elements of a directed acyclic graph with a topological order representing paths from node 1 to node N'''
    
    def generate_DAG(self, N, max_weight):
        '''generate a Directed Acyclic Graph based on numpy : scalars of the returned matrix represent the weights betweens the nodes
        The DAG computed can be represent as several path between node 1 and node N'''
        dist_mat = np.zeros((N, N))
        node_mat = np.zeros((N, N))
    
        for i in range(N):
            for j in range(N):
                dist_mat[i][j] = npr.randint(1, max_weight, size=1)[0]
        
        #generating strict upper half matrix (generate the children of each nodes)
        for i in range(N-1):
            for j in range(i+1, N):
                node_mat[i][j] = npr.randint(2, size=1)[0]
        
        #generating strict lower half matrix (generate the parent of each nodes)
        for i in range(1, N-1):
            for j in range(i+1, N-1):
                if node_mat[i][j] == 0:         #We generate a DIRECTED graph therefore if node_mat[i][j] == 1 then node_mat[j][i] == 0 
                    node_mat[j][i] = npr.randint(2, size=1)[0]
        
        
        for i in range(N-1):                #verify that node i (except N) has always at most one child, garanteeing that there's at most path between nodes1 and N
            if sum(node_mat[i][:]) == 0:         
                child_i = npr.randint(1, N, size=1)[0]
                while(child_i == i or child_i == 0):                          #we want an ACYCLIC graph : node_mat[i][i] must be zero
                    child_i = npr.randint(1, N, size=1)[0]
                node_mat[i][child_i] = 1
        
        for i in range(1, N-1):         #verify that node i (except 1) has always at most one parent
            if sum(node_mat[:][i]) == 0:
                parent_i = npr.randint(0, N-1, size=1)[0]
                while(parent_i == i or parent_i == N-1 or node_mat[i][parent_i]==1):
                    parent_i = npr.randint(0, N-1, size=1)[0]
                node_mat[parent_i][i] = 1
        
        
        random_DAG = node_mat * dist_mat

        return random_DAG
    
    
    def parent_of(self, node):
        '''return the parents of specified node in a numpy array of lenght N_nodes, it means the column of indice 'node' of A.DAG'''
        parent_of_node = A.DAG[:, node]
        return list(parent_of_node)
    
    
    def indices_parent_of(self, node):
        '''return the indices of specified node in a list'''
        indices_of_parent_of_node = [int(k) for k in np.argwhere(self.parent_of(node))]
        return indices_of_parent_of_node
    
    
    def children_of(self, node):
        '''return the parents of specified node numpy array of lenght N_nodes'''
        children_of_node = A.DAG[node, :]
        return list(children_of_node)
    
    
    def indices_children_of(self, node):
        '''return the indices of specified node in a list'''
        indices_of_children_of_node = [int(k) for k in np.argwhere(self.children_of(node))]
        return indices_of_children_of_node
    
    
    '''Implementing algorithms used on graphs'''
    
    def smooth_max(x_input, gamma=1, omega='negentropy'):
        if omega == 'negentropy':
            return gamma * np.log( sum( list(map(lambda x: np.exp(x / gamma), x_input)) ) )
    
    
    def grad_smooth_max(x_input, gamma=1, omega='negentropy'):
        if omega == 'negentropy':
            sum_exp = sum( list(map(lambda x: np.exp(x / gamma), x_input)) )
            return [ np.exp(x / gamma) / sum_exp for x in x_input ]
    
    
    def mat_sym_canonic(q):
        '''return the canonic symetric matrix of a vector q'''
        Nq = len(q)
        q_mat = np.zeros((Nq, Nq))
        
        for i in range(Nq):
            for j in range(Nq):
                q_mat[i][j] = q[i] * q[j]
        return q_mat
    
    
    def lapl_smooth_max(x_input, gamma=1, omega='negentropy'):
        if omega == 'negentropy':
            q = grad_smooth_max(x_input, gamma, omega=omega)
            
            return (np.diag(q) - mat_sym_canonic(q)) / gamma
    
    def sum_weight_vector_parent(self, node, vector, conserve_size=False):
        '''return the list of the sum of weights with a vector of parent of node : [theta(i,j) + v(j) for j in parent of i] for node=i '''
        if conserve_size == False:
            list_sum = [self.parent_of(node)[k] + vector[k] for k in self.indices_parent_of(node)]
            return list_sum         #Only have non nul value, do not have 0 
        else:
            list_sum = np.zeros(self.N_nodes)
            for k in self.indices_parent_of(node):
                list_sum[k] = self.parent_of(node)[k] + vector[k]
            return list_sum         #Keep the placeholder with 0 value
        
    def DP(self):
        
        N = len(self.DAG[0])
        
        v = np.zeros(N)
        v[0] = 0
        
        for i in range(1, N):
            v[i] = max(A.sum_weight_vector_parent(i, v))
        
        return v[N-1]
        
    
    
    def smooth_DP(self):
        '''Return a 4-tuple: -first = DP(self.DAG)   -second = grad_DP(self.DAG) {-third, fourth = e_bar, Q used in gradient_computing}'''
        #Initialization
        N = len(self.DAG[0])
        
        v = np.zeros(N)
        v[0] = 0
        
        e_bar = np.zeros(N)
        e_bar[N-1] = 1
    
        Q = np.zeros((N, N))
        E = np.zeros((N, N))
        
        #Topological order
        for i in range(1, N):
            v[i] = smooth_max(A.sum_weight_vector_parent(i, v), gamma=1)
            #for j in self.parent_of(i):
             #   if j != 0:
            Q[i] = grad_smooth_max(A.sum_weight_vector_parent(i, v, conserve_size=True), gamma=1)
        
        
        #Reverse topological order
        E[:, N-1] = Q[:, N-1]
        for j in range(N-1):
            for i in self.indices_children_of(N-2-j):
                E[i, N-2-j] = e_bar[i] * Q[i, N-2-j]
            e_bar[N-2-j] = sum([E[i, N-2-j] for i in self.indices_children_of(N-2-j)])
            
        #Reverse topological order
        #for j in range(N-1):
        #   for i in self.indices_children_of(N-2-j):
        #        #if i != 0:
        #        E[i][N-2-j] = E[i][N-2-j] * e_bar[i]
        #    e_bar[N-2-j] = sum([E[k][N-2-j] for k in self.indices_children_of(N-2-j)])    #Attention ne pas indenter il faut sortir de la boucle for children_of_j
        #
        return (v[N-1], E, e_bar, Q)
    
        
    
#    def directional_derivative_computing(self, perturbation, gamma=1):
#        #Initialization
#        N = len(self.DAG)
#        
#        v_bis = np.zeros(N)
#        v_bis[0] = 0
        
#        e_bar_bis = np.zeros(N)
#        e_bar_bis[N-1] = 0
        
#        Q_bis = np.zeros((N, N))
#        E_bis = np.zeros((N, N))
        
#        e_bar, Q = DP_computing(self.DAG)[2], DP_computing(self.DAG)[3]
        
        
        #Topological order
#        Q_bis_duplicate = np.zeros(Q_bis.shape)
        
#        for i in range(1, N):
#            v_bis[i] = sum([Q[i][j] * (perturbation[i][j] + v_bar[j]) for j in self.children_of(i) if j != 0])
            
#            Q_bis_duplicate[i] = ( (np.diag(Q[i]) - mat_sym_canonic(Q[i])) / gamma) @ (perturbation[i][:] + v_bis[:])
#            for j in self.parent_of(i):
#                if j != 0:
#                    Q_bis[i][j] = Q_bis_duplicate[i][j]
        
        
        #Reverse topological order
#        for j in range(1, N-1):
#            for i in self.children_of(j):
#                if i != 0:
#                    E_bis[i][N-1-j] = Q_bis[i][N-1-j] * e_bar[i] + Q[i][N-1-j] * e_bar_bis[i]
#            e_bar_bis[N-1-j] = sum([E_bis[i][N-1-j] for i in self.children_of(j) if i != 0]) #Attention ne pas indenter il faut sortir de la boucle for children_of_j
        
        
        return (v_bis[N-1], E_bis)


#%%
npr.seed(1000)
A = DAG(22, 100)

b = list(npr.randint(0, 300, size=10))
print(A.DAG)
print(b)
#print([j, np.argwhere(A.children_of(1)==j) for j in A.children_of(1)])
#print([float(k) for k in np.argwhere(A.children_of(5))])
#print(A.children_of(0))
#print(A.indices_children_of(0))
#print('\n')
#print([float(k) for k in np.argwhere(A.parent_of(5))])
#print(A.indices_parent_of(1))
#print(float(np.argwhere(A.children_of(2))[0]))
#print(float(np.argwhere(A.children_of(2))[1]))
#print(A.parent_of(2))
#print(A.indices_parent_of(2))
#print([k for k in A.indices_parent_of(1)])
#print(A.sum_weight_vector_parent(2, b))
#print(A.sum_weight_vector_parent(2, b, conserve_size=True))
#d = A.sum_weight_vector_parent(2, b, conserve_size=True)
#print(grad_smooth_max(d, gamma=1))

#print(A.DP_computing()[2])
#print(A.DP_computing()[0])


#%%

print(A.DP())
print(A.smooth_DP()[0])