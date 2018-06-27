from baccus import BACCUS
import numpy as np
from numpy.random import normal
import theano
import theano.tensor as T

#################
# Generate data #
#################

#2d gaussian: 
mean = [0., 1.]
cov = [[0.25, 0.02], [0.02, 0.04]]  # diagonal covariance
x1, y1 = np.random.multivariate_normal(mean, cov, 500).T
sx1 = np.ones(500)*0.3
sy1 = np.ones(500)*0.2

#2d gaussian: 
mean = [-0.2, .6]
cov = [[0.04, -0.005], [-0.005, 0.01]]  # diagonal covariance
x2, y2 = np.random.multivariate_normal(mean, cov, 200).T
sx2 = np.ones(200)*0.2
sy2 = np.ones(200)*0.05

#2d gaussian: 
mean = [2., -2]
cov = [[0.04, -0.005], [-0.005, 0.01]]  # diagonal covariance
x3, y3 = np.random.multivariate_normal(mean, cov, 300).T
sx3 = np.ones(300)*0.02
sy3 = np.ones(300)*0.05

data1 = np.stack((x1,y1,sx1,sy1),axis=1)
data2 = np.stack((x2,y2,sx2,sy2),axis=1)
data3 = np.stack((x3,y3,sx3,sy3),axis=1)
DATA = [data1,data2,data3]

##############
# Likelihood #
##############

def lkl_norescale(theta,DATA):
    x = theta[0]
    y = theta[1]
    
    count = 0
    npar = 2
    
    log_lkl = 0
    for i in range(0,len(DATA)):
        shift_x = theta[npar+count]
        count += 1
        shift_y = theta[npar+count]
        count += 1
        
        xpar = x+shift_x
        ypar = y+shift_y

        dat = DATA[i]
        xdat = dat[:,0]
        ydat = dat[:,1]
        sx = dat[:,2]
        sy = dat[:,3]
        
        log_lkl -= 0.5*(np.sum(((xdat-xpar)/sx)**2.) + np.sum(((ydat-ypar)/sy)**2.))
        
    return log_lkl
    
    
    
    
###############
# CALL BACCUS #
###############

#bounds for the priors of the model parameters
prior_bounds_model = [(-10.,10.),(-10.,10.)]
#bounds for the priors of the shifts
prior_bounds_shifts = [(0,-8.,8.),(1,-8.,8.)]
#bounds for the priors of the variances of the shifts
prior_bounds_var = [(0,0,8),(1,0,8)]
#which shifts in each data set
kind_shifts = [(0,1),(0,1),(0,1)]
#prior for the variances of the shifts - a lognormal
b = -2
xi = 10
y = T.dscalar('y')
s = -0.5*((T.log(y)-b)/2./xi)**2.
logprior_sigma = theano.function([y],s)
prior_var = []
for ivar in range(0,2):
    prior_var += [logprior_sigma]
    
    
    
want_rescaling = False
model = BACCUS(prior_bounds_model=prior_bounds_model,prior_bounds_shifts=prior_bounds_shifts,prior_bounds_var=prior_bounds_var,
              lkl = lkl_norescale, kind_shifts = kind_shifts,prior_var=prior_var,want_rescaling=want_rescaling)
	      


#Set initial position, steps and walkers
nwalkers = 250
steps = 4000
ndata = len(DATA)
pos = []

for i in range(0,nwalkers):
    #Model parameters
    pos += [np.array([normal(0,1),normal(0,1)])]
    #Rescaling parameters, if wanted
    if want_rescaling:
        for j in range(0,ndata):
            pos[i] = np.append(pos[i],normal(1,0.2))
    #shift_hyperparams
    for j in range(0,ndata):
        pos[i] = np.append(pos[i],normal(0.,1.))
        pos[i] = np.append(pos[i],normal(0.,1.))
                
    #Std for shifts
    pos[i] = np.append(pos[i],normal(1.,0.2))
    pos[i] = np.append(pos[i],normal(1.,0.2))
        
    #correlation of shifts
    pos[i] = np.append(pos[i],normal(0.,0.2))




## RUN the CHAIN
model.run_chain(DATA, stepsize=2.4, pos=pos, nwalkers=nwalkers, steps=steps)

## Get the results
model.get_results()
