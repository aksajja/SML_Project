import numpy as np

def is_symm(x):
  return np.all(x==x.T)

def is_pos_def(x):
  return np.all(np.linalg.eigvals(x) > 0)

def is_invertible(a):
  return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def generate_psd_cov_mat(num_nodes,num_obs,using_R=True,force=True):
  if using_R:
    import subprocess
    import pandas as pd
    from os import path
    if not path.exists('data/cov_mat.csv') or force:    
      subprocess.check_call(['/usr/bin/Rscript', \
          '--vanilla', \
          'Experiments/ggm.R', \
          '--args', \
          f'{num_nodes}', \
          f'{num_obs}'], shell=False)
    cov_mat = None
    cov_mat_df = pd.read_csv('data/cov_mat.csv', header=None)
    cov_mat = cov_mat_df.to_numpy()
    adj_mat_df = pd.read_csv('data/adj_mat.csv', header=None)
    adj_mat = adj_mat_df.to_numpy()

    assert is_invertible(cov_mat)==True
    assert is_pos_def(cov_mat)==True
    assert is_symm(cov_mat)==True
    
    return adj_mat, cov_mat
  else:
    # Can try to write a Python script to mimic the R script.
    pass

