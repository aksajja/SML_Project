library(MASS)
library(Matrix)
args <- commandArgs(trailingOnly = TRUE)

set.seed(2019)
# c(args)
# Problem parameters
# p = 2000    # number of covariates
# n = 400     # number of observations
# k = 60      # number of non-zero regression coefficients
# A = 60      # signal amptitude 
p = strtoi(args[2])    # number of covariates
n = strtoi(args[3])     # number of observations
k = 6      # number of non-zero regression coefficients
A = 6      # signal amptitude 
nonzero = sample(p, k) 
beta = 1 * (1:p %in% nonzero) * sample(c(-1,1),p,rep = T) 

`%diag*%` <- function(d, X)
  d * X  #diag(d) %*% X
`%*diag%` <- function(X, d)
  t(t(X) * d)

# setup the banded precison matrix
nn=10            # bandwidth
rho0=-0.05       # the non-zero partial correlation
Omega0 = (1 - rho0) * diag(p) + rho0
Omega0[abs(outer(1:p, 1:p, '-')) > nn] = 0
S=solve(Omega0)
# renormalize so that the marginal variance = 1
ds=diag(S)
Omega=sqrt(ds)%diag*%Omega0%*diag%sqrt(ds)
Sigma=sqrt(1/ds)%diag*%S%*diag%sqrt(1/ds)
Sigma.chol = chol(Sigma)

# Generate the covariate
X = matrix(rnorm(n * p), n) %*% Sigma.chol
# Known graphical information: adjacent matrix
Adj = 1*(Omega!=0 )
cat('For ',p,' nodes ',n, ' samples ', ' -- sparsity of adj mat is: ',nnzero(Adj)/(p**2), '\n')
write.matrix(Sigma,'data/cov_mat.csv',',')
write.matrix(Adj,'data/adj_mat.csv',',')
