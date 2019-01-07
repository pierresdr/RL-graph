# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 21:44:47 2019

@author: ekgia
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def DP_computing(weights):
    N = len(weights[0])
    v = np.zeros(N)
    v[0] = 0
    
    E_bar = np.zeros(N)
    E_bar[N] = 1

    Q = np.zeros(N, N)
    E = np.zeros(N, N)
    
    for i in range(1, N):
        pass #compute v[i] = max_omega_j_in_Pi(theta_ij + v_j)
        pass #compute Q[i][j] = grad_max_omega_j_in_Pi(theta_ij + v_j)
    
    for j in range(1, N-1):
        #for i in children_of_j
            E[i][N-1-j] = E[i][N-1-j] * E_bar[i]
            E_bar[j] = sum(E[i][j])
    
    return v[N], E
    
    
    
    

#%%

for i in range(1, 10):
    print(i)
    print(10-i)
