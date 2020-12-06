library(knockoff)
library(MASS)
source('cknockoff/src/knockoff_measure.R')
source('cknockoff/src/util.R')

set.seed(2019)
# Problem parameters
width  = 15   # the width of the 2D lattice
height = 15   # the height of the 2D lattice
p = width * height    # number of covariates
Graph = SpatialGM(width,height) 
Theta = 0.2           #  equal entries  

n = 200     # number of observations
k = 60      # number of non-zero regression coefficients
A = 20      # signal amptitude 
nonzero = sample(p, k) 
beta = 1 * (1:p %in% nonzero) * sample(c(-1,1),p,rep = T) 

# Generate the covariate by Coupling from the past algorithm
X = c()
for( tt in 1: n){
    newx = CFPT.Ising(width,  temperature =  1 / Theta) 
    # CFPT.Ising() returns one covariate data point as a matrix 
    X = rbind(X,  c(newx))
}

c(X)
# Known graphical information: adjacent matrix
# Graph = SpatialGM(width,height)

write.matrix(X,'ising_mat.csv',',')
write.matrix(Graph,'ising_adj_mat.csv',',')