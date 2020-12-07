from itertools import combinations
import numpy as np
import pandas as pd
from typing import List
import copy

from Experiments.sampling_from_gaussian import generate_psd_cov_mat
from utils import plot_error_vs_samples

def generate_samples(num_nodes, num_samples):
  adj_mat, cov_mat = generate_psd_cov_mat(num_nodes, num_samples)
  mean = [0 for i in range(cov_mat.shape[0])]
  T_samples = np.random.multivariate_normal(np.array(mean), cov_mat, num_samples)
  return adj_mat, T_samples

"""**Phase 1** - Estimation of conditional variance.
Elements of the inverse of the precision matrix are computed. A loss function is minimized to achieve the same.
"""

def compute_empirical_covariance(x_samples):
  n = x_samples.shape[0]
  emp_mat = []
  for i in range(0, x_samples.shape[1]):
    emp_list_i = []
    for j in range(0, x_samples.shape[1]):
      emp_ij = 0
      for k in range(0, n):
        x_i = x_samples[k][i]
        x_j = x_samples[k][j]
        emp_ij += x_i*x_j
      emp_ij = emp_ij/n
      emp_list_i.append(emp_ij)
    emp_mat.append(emp_list_i)
  return np.array(emp_mat)

from numpy.linalg import inv
def compute_conditional_var(combo, node_i, emp_matrix, x_samples):
  emp_ii = emp_matrix[node_i][node_i]
  emp_iA = []
  for node in combo:
    emp_iA.append(emp_matrix[node_i][node])
  emp_iA = np.array(emp_iA)
  emp_AA = []
  for node_i in combo:
    emp_AA_inner = []
    for node_j in combo:
      emp_AA_inner.append(emp_matrix[node_i][node_j])
    emp_AA.append(emp_AA_inner)
  emp_AA = np.array(emp_AA)
  emp_AA_inv = inv(emp_AA)
  emp_Ai = []
  for node in combo:
    emp_Ai.append(emp_matrix[node][node_i])
  emp_Ai = np.array(emp_Ai)
  product = np.dot(np.dot(emp_iA,emp_AA_inv),emp_Ai)
  return emp_ii- product

def compute_optimal_conditional_var(x_samples, max_deg):
  # x_samples. Each sample 
  # for each submatrix of dim A(max_deg) compute conditional_var. The min is returned.\
  conditional_vars =  []
  num_samples = x_samples.shape[0]
  num_nodes = x_samples.shape[1]
  emp_matrix = compute_empirical_covariance(x_samples)
  for node_i in range(num_nodes):
    # select j indices from [num_samples]
    node_excluded_list = list(range(num_nodes))

    node_excluded_list.remove(node_i)
    all_combos = list(combinations(node_excluded_list,max_deg))
    candidate_conditional_vars = []
    for _combo in all_combos:
      # candidate_conditional_vars.append(compute_conditional_var(x_samples[:,_combo]),x_samples[:,node_i])
      candidate_conditional_vars.append(compute_conditional_var(_combo, node_i, emp_matrix, x_samples))
    candidate_conditional_variance = min(candidate_conditional_vars)
    conditional_vars.append(1/candidate_conditional_variance)

  return conditional_vars

# conditional_vars = compute_optimal_conditional_var(x_samples,2)

"""**Phase 2** - Iterative support testing.

"""

def compute_beta(node, A, emp_matrix):
  emp_AA = []
  for node_i in A:
    emp_AA_inner = []
    for node_j in A:
      emp_AA_inner.append(-1*emp_matrix[node_i][node_j])
    emp_AA.append(emp_AA_inner)
  emp_AA = np.array(emp_AA)
  emp_AA_inv = inv(emp_AA)
  emp_Ai = []
  for node_i in A:
    emp_Ai.append(emp_matrix[node_i][node])
  emp_Ai = np.array(emp_Ai)
  return np.dot(emp_AA_inv, emp_Ai)

def compute_normalized_edge_strength(node, combo, emp_matrix,getmax=True):
  k_ijs = []
  for node_j in combo:
    beta_ij = compute_beta(node, [node_j], emp_matrix)
    beta_ji = compute_beta(node_j, [node], emp_matrix)
    k_ijs.append(beta_ij*beta_ji)
  if getmax:
    return max(k_ijs)
  else:
    return min(k_ijs)

def support_testing(x_samples, max_deg, emp_matrix):
  B_nodes = []

  for node in range(x_samples.shape[1]):
    node_excluded_list = list(range(x_samples.shape[1]))
    node_excluded_list.remove(node)
    B1 = list(combinations(node_excluded_list,max_deg))
    for b1_combo in B1:
      passed = True
      k = compute_normalized_edge_strength(node, b1_combo,emp_matrix, False)
      new_node_excluded_list = copy.deepcopy(node_excluded_list)
      for value in b1_combo:
        new_node_excluded_list.remove(value)
      B2 = list(combinations(new_node_excluded_list, max_deg))
      for b2_combo in B2:
        beta_b1b2 = compute_beta(node, b1_combo+b2_combo, emp_matrix)
        k_node = compute_normalized_edge_strength(node, b2_combo,emp_matrix, True)
        if k_node > k/2:
          passed = False
          break
      if passed:
        B_nodes.append(list(b1_combo))
        break
  
  return B_nodes

"""**Phase 3** - Pruning."""

def pruning(B_nodes, num_nodes, max_deg, emp_matrix):
  import copy
  nodes_list = list(range(num_nodes))
  pruned_neighborhoods = []
  for _i in range(len(B_nodes)):
    candidates = []
    k = compute_normalized_edge_strength(_i, B_nodes[_i],emp_matrix, False)
    _node_neighborhood = []
    node_excluded_list = copy.deepcopy(nodes_list).remove(_i)
    node_excluded_list = list(set(copy.deepcopy(nodes_list)) - set(B_nodes[_i]))
    B2 = np.random.choice(node_excluded_list, max_deg,replace=False)
    for j in B2:
      candidate = compute_normalized_edge_strength(j, [_i],emp_matrix, True)
      candidates.append(candidate)
      if candidate>k/2:
        _node_neighborhood.append(j)

    pruned_neighborhoods.append(_node_neighborhood)
  
  return np.asarray(pruned_neighborhoods)

def compute_error(label,pred):
  return np.count_nonzero(label!=pred)/label.shape[0]**2

def neighborhood_to_adj_mat(neighborhoods,num_nodes):
  adj_mat = np.zeros((num_nodes,num_nodes))
  for _node in range(len(neighborhoods)):
    for _neighbor_i in neighborhoods[_node]:
      adj_mat[_node][_neighbor_i]=1
  
  return adj_mat

def exp_misra(num_nodes: List[int], num_samples: List[int], zeta: float):
  for p in num_nodes:
    errors = []
    max_deg = int(0.2*p)
    for n in num_samples:
      true_adj_mat, sample_set = generate_samples(p,n)
      conditional_var = compute_optimal_conditional_var(sample_set,max_deg)
      emp_matrix = compute_empirical_covariance(sample_set)
      B_nodes = support_testing(sample_set,max_deg,emp_matrix)
      neighborhoods_list = pruning(B_nodes,p,max_deg,emp_matrix)
      pred_prec_mat = neighborhood_to_adj_mat(neighborhoods_list,p)
      err = compute_error(true_adj_mat,pred_prec_mat)
      print(f'Sample complexity bound holds? Pred_err:{err}, Zeta:{zeta} ')
      errors.append(err)
    plot_error_vs_samples('Misra',errors,num_samples,p)