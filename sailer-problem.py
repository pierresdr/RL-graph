# -*- coding: utf-8 -*-

import numpy as np
import sklearn.datasets as skd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import random
import itertools
import math
import copy
import sklearn as sk

#%% CREATE STOPS TO VISIT IN 2D

dim = 2
n_obs = 10
np.random.seed(1000)
points = np.random.rand(n_obs*dim).reshape(n_obs,dim)
plt.scatter(points[:,0],points[:,1])


#%% COMPUTE DISTANCE MATRIX

dist_mat = np.zeros((n_obs,n_obs))
for i in range(n_obs-1):
    dist_mat[i,i+1:] = np.linalg.norm(points[i+1:,:]-points[i,:],axis=1)
dist_mat  = dist_mat + dist_mat.T


#%% HELD–KARP ALGORITHM

def getSubsets(x,i):
    """ Returns all the subsets of size i in x
    """
    return list(map(list, itertools.combinations(x, i)))

def findMini(S,k,dist_mat,C_subset,C_dist):
    """ k  has already been removed from S
    """
    distances = list()
    for i in S:
        sub = C_subset.index([S,i])
        distances.append(C_dist[sub]+ dist_mat[i,k])
    return min(distances)
    
def findOptPath(opt,C_subset,n):
    opt_path = np.zeros(n)
    opt_path[0] = np.argmin(opt)+1
    memory_path = list(range(1,n))
    
    for i in range(n-2):
        print(memory_path)
        sub = C_subset.index([memory_path,int(opt_path[i])])
        opt_path[i+1] = C_subset[sub][1]
        memory_path.remove(C_subset[sub][1])
    
    opt_path[n] = 0
    return opt_path

def heldCarp(dist_mat):
    n = np.shape(dist_mat)[0]
    C_dist = list()
    C_subset = list()
    for i in range(1,n):
        C_subset.append([[i],i])
        C_dist.append(dist_mat[0,i])
    for i in range(2,n):
        print(i)
        for j in getSubsets(range(1,n),i):
            for k in j:
                C_subset.append([j,k])
                j_copy = copy.deepcopy(j)
                j_copy.remove(k)
                mini = findMini(j_copy,k,dist_mat,C_subset,C_dist)
                C_dist.append(mini)
    
    opt = np.zeros(n-1)
    for i in range(1,n):
        opt[i-1] = findMini(list(range(1,n)),i,dist_mat,C_subset,C_dist)

    
    return findOptPath(opt,C_subset,n)
            
            
"""subsets = np.arange(2**(n-1))
subsets = np.delete(subsets,2**(np.arange(n-1)))"""

#%%

heldCarp(dist_mat)





#%%



class state_action:
    def __init__(self):
        self.r = list()

    def bellman_action(self,V,gamma):
        #   For QUESTION 2
        #   Computes Bellman Operator for a given state and action
        return(self.r+gamma*np.dot(self.P,V))
        
    def addObsOfR(self,r):
        self.r.append(r)
    
    def
    


class state:
    #   Class which contains the returns and probabilities for a given state
    def __init__(self,nb_actions):
        self.nb_actions = nb_actions
        self.actions = list()
        for i in range(self.nb_actions):
            self.actions.append(state_action())
    
    def bellman_opt(self,V,gamma):
        #   For QUESTION 2
        #   Computes the optimal Bellman operator over the possible actions for a given state
        temp = 0
        for j in range(self.nb_actions):
            temp = max(temp,self.actions[j].bellman_action(V,gamma))
        return (temp)

    def make_trajectory(self):
        #   For QUESTION 3
        """temp1 = np.cumsum(self.policy)
        random1 = np.random.rand()
        chosen_action = np.where(temp1>=random1)[0][0]"""
        chosen_action = np.where(self.policy==1)[0][0]
        temp2 = np.cumsum(self.actions[chosen_action].P)
        random2 = np.random.rand()
        new_state = np.where(temp2>=random2)[0][0]
        return self.actions[chosen_action].r, new_state
    
    def update_policy(self,V,gamma):
        #   For QUESTION 3
        temp = 0
        sub = 0
        for j in range(self.nb_actions):
            temp2 = self.actions[j].bellman_action(V,gamma)
            if(temp2>temp):
                temp = temp2
                sub = j
        new_policy = np.zeros(self.nb_actions)
        new_policy[sub] = 1
        if(np.array_equal(self.policy,new_policy)):
            return(False)
        else :      
            self.policy = new_policy
            return(True)
    
    def set_policy(self,init_policy):
        self.policy = init_policy
    


class MDP_IRL:
    #   Class which contains all the states, and their return and probabilities
    def __init__(self,action_by_state,gamma):
        self.nb_states = len(action_by_state)
        self.states=list()
        self.gamma = gamma
        
        #build the states contained in MDP
        self.action_by_state=np.cumsum(action_by_state)
        for i in range(self.nb_states-1 ):
            self.states.append(state(action_by_state[i]))
    
    def trajectory(self,j,T):
        #   For QUESTION 3
        result, new_state = self.states[j].make_trajectory()
        for i in range(T):
            temp, new_state = self.states[new_state].make_trajectory()
            result += gamma**(i+1)*temp
        return result
    
        
#%%
        

class state_action:
    def __init__(self,r,P):
        self.r = r
        self.P = P

    def bellman_action(self,V,gamma):
        #   For QUESTION 2
        #   Computes Bellman Operator for a given state and action
        return(self.r+gamma*np.dot(self.P,V))



class state:
    #   Class which contains the returns and probabilities for a given state
    def __init__(self,r,P):
        self.nb_actions = len(r)
        self.actions = list()
        for i in range(self.nb_actions):
            self.actions.append(state_action(r[i],P[i,:]))
        self.policy = np.ones(self.nb_actions)/self.nb_actions
    
    def bellman_opt(self,V,gamma):
        #   For QUESTION 2
        #   Computes the optimal Bellman operator over the possible actions for a given state
        temp = 0
        for j in range(self.nb_actions):
            temp = max(temp,self.actions[j].bellman_action(V,gamma))
        return (temp)

    def make_trajectory(self):
        #   For QUESTION 3
        """temp1 = np.cumsum(self.policy)
        random1 = np.random.rand()
        chosen_action = np.where(temp1>=random1)[0][0]"""
        chosen_action = np.where(self.policy==1)[0][0]
        temp2 = np.cumsum(self.actions[chosen_action].P)
        random2 = np.random.rand()
        new_state = np.where(temp2>=random2)[0][0]
        return self.actions[chosen_action].r, new_state
    
    def set_policy(self,init_policy):
        self.policy = init_policy
    
    def selectAction(self):
        actions = np.cumsum(self.policy)
        return  np.where(actions>=np.random.rand())[0][0]
    
    def getNewState(self,chosen_action):
        new_states = np.cumsum(self.actions[chosen_action].P)
        return np.where(new_states>=np.random.rand())[0][0]
        
    


class MDP:
    #   Class which contains all the states, and their return and probabilities
    def __init__(self,r,P,gamma):
        self.nb_states = len(r)
        self.states=list()
        self.V = np.zeros(self.nb_states)
        self.gamma = gamma
        
        #build the states contained in MDP
        self.action_by_state=np.cumsum(action_by_state)
        sub = range(self.action_by_state[0])
        self.states.append(state(r[sub],P[sub,:]))
        for i in range(self.nb_states-1 ):
            sub = range(self.action_by_state[i],self.action_by_state[i+1])
            self.states.append(state(r[sub],P[sub,:]))
    
    def trajectory(self,j,T):
        #   For QUESTION 3
        result, new_state = self.states[j].make_trajectory()
        for i in range(T):
            temp, new_state = self.states[new_state].make_trajectory()
            result += gamma**(i+1)*temp
        return result
    
    def createDR(self,n):
        states = np.random.randint(0,nb_states,n)
        DR = np.zeros(3,n)
        for i in range(n):
            DR[0,i] = i
            DR[1,i] = self.states[i].selectAction()
            DR[2,i] = self.states[i].getNewState(DR[1,i])
        return DR
            
    
    def get_policy(self):
        #   For QUESTION 2 and QUESTION 3
        policy = np.zeros(self.action_by_state[len(self.action_by_state)-1])
        sub = range(self.action_by_state[0])
        policy[sub] = self.states[0].policy
        for i in range(self.nb_states-1 ):
            sub = range(self.action_by_state[i],self.action_by_state[i+1])
            policy[sub] = self.states[i+1].policy
        return policy


gamma = 0.95



r = np.array([0,0,5/100,0,0,0,0,1,9/10]) #the returns of each (x,a), in the right order
P = np.array([[0.55,0.45,0], #the probabilities p(y|x,a) for each (x,a) in the right order
             [0.3,0.7,0],
             [1,0,0],
             [1,0,0],
             [0,0.4,0.6],
             [0,1,0],
             [0,1,0],
             [0,0.6,0.4],
             [0,0,1]])

MDP_opt = MDP(r,P,gamma)
    
MDP_IRL = MDP_IRL(action_by_state,gamma)   
    

def CSI(DC,DR,action_by_state,gamma):
    """DC contains a list of states in its first lign and a list of corresponding
    actions given expert policy in the second lign
    """
    classifier = sk.svm.SVC(gamma='auto')
    classifier.fit(DC[0,:],DC[1,:])
    
    r = MDP(action_by_state,gamma)
    
    for i in range(np.shape(DR)[1]):
        r.state[DR[0,i]] = classifier.score(DR[0,i],DR[1,i]) - gamma * classifier.score(DR[2,i],classifier)
        
    return r
