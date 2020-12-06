# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 04:19:20 2020

@author: Vishnu
"""
import numpy as np

def compute_p(x, theta, n_nodes):
    
    s = 0
    for i in range (0, n_nodes):
        for j in range (0, n_nodes):
            if (i != j):
                s = s + theta[i, j]*x[i]*x[j]
            elif (i == j):
                s = s + theta[i, i]*x[i]
    
    return np.exp(s)

def from_space(n, n_nodes):
    
    bin_rep = bin(n)
    x = []
    
    for i in range (0, n_nodes):
        x.append(-1)
    
    for i in range (0, len(bin_rep) - 2):
        if(bin_rep[len(bin_rep) - i - 1] == '1'):
            x[n_nodes - i - 1] = 1
        
    return x         
        
def normalizing_const(n_nodes, theta):
    
    s = 0
    for i in range (0, 2**(n_nodes)):
        x = from_space(i, n_nodes)
        s = s + compute_p(x, theta, n_nodes)
    
    c = 1/s
    return c

def compute_cdf(n_nodes, theta):
    
    cdf = []
    c = normalizing_const(n_nodes, theta)
    
    x = from_space(0, n_nodes)
    cdf.append(c * compute_p(x, theta, n_nodes))
    
    for i in range (1, 2**(n_nodes)):
        x = from_space(i, n_nodes)
        cdf.append(c * compute_p(x, theta, n_nodes) + cdf[i - 1])
        
    return cdf

def sampling(theta, n_nodes, cdf):    
    u = np.random.uniform(0, 1)
    
    left = 0
    right = 2**(n_nodes) - 1
    
    while ((right - left) > 1):
        mid = int(round((left + right)/2))
        if(u > cdf[mid]):
            left = mid
        if(u < cdf[mid]):
            right = mid
    
    return from_space(left, n_nodes)
        
def compute_theta(n_nodes, alpha, beta, max_deg):
    theta = np.zeros([n_nodes,n_nodes])
    
    for row in range(n_nodes-1):    # nth row deg depends on n-1 rows
        # randomly select max_deg-row_deg indices from [row+1:n_nodes]
        row_deg = np.count_nonzero(theta[row])
        neighbors = np.random.choice(np.arange(row+1,n_nodes), min(n_nodes-row-1,max_deg-row_deg), replace=False)
        for _nb in neighbors:
            self_corr = np.random.uniform(0,1)
            theta[row][row]=self_corr
            if np.count_nonzero(theta[_nb])>=max_deg:   # Condition to restrict deg in unvisited rows.
                continue
            corr = np.random.uniform(alpha,beta)
            theta[row][_nb]=corr
            theta[_nb][row]=corr
    theta = np.asarray(theta)
    # print(f'Post processing - ')
    # print(f'Alpha : {np.min(theta[np.nonzero(theta)])}; Beta : {np.max(theta)}; Max_deg : {np.max(np.count_nonzero(theta,axis=0))-1}')
    return theta 

def ising_samples(n_nodes, n_samples, alpha, beta, max_deg):
    theta = compute_theta(n_nodes, alpha, beta, max_deg)
    cdf = compute_cdf(n_nodes,theta)
    sample_set = []
    for _sample in range(n_samples):
        sample_set.append(sampling(theta,n_nodes,cdf))

    alpha = np.min(theta)
    beta = np.max(theta)
    max_deg = None
    return theta,np.asarray(sample_set)

