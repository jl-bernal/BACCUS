BACCUS
=======

BACCUS: BAyesian Conservative Constraints and Unknown Systematics.
J. L. Bernal and J. A. Peacock, 2018
arXiv: 1803.04470
JCAP

This code allows to apply BACCUS to any likelihood given by the user.
The likelihood only needs to take the corresponding shift or rescaling
parameter and apply as corresponds (as shown in the example).

For more information about BACCUS, read https://arxiv.org/pdf/1803.04470.pdf

How to use BACCUS
==================

BACCUS(prior_bounds_model=prior_bounds_model,prior_bounds_shifts=prior_bounds_shifts,
	prior_bounds_var=prior_bounds_var,lkl = lkl, kind_shifts = kind_shifts,
	prior_var=prior_var,want_rescaling=want_rescale,extras=extras)

Initiate the object which prepare the priors and the run the chain


Parameters
-----------
        
prior_var: log(prior) of the diagonal elements of the covariance matrix
        of the shift parameters (var). list of theano functions
        Each function should take sigma_ii^2 as input
        
prior_corr: log(prior) of the off-diagonal elements of the covariance matrix
        of the shift parameters (correlations). theano function
        This function should take a vector with Ndim*(Ndim-1)/2 rho_ij values
        indexed counting the elements in the upper triangular part row by row
        
prior_model: log(prior) of the model parameters (if other than uniform). 
        format: list of tuples, with the tuple being (int,theano function)::
        (index of the model parameter, prior)
        
prior_bounds_model: list of tuples with log(prior) bounds for model parameters
        example [(prior_min1,prior_max1),(prior_min2,prior_max2)]
    
prior_bounds_shifts: list of tuples with (kindshift,bound_min,bound_max)
    
prior_bounds_var: list of tuples with (kindshift,bound_min,bound_max)
    
lkl: function which returns the log of the likelihood of your model
        
kind_shifts: type of shifts for each class of datasets. List of tuples of int values
        example: [(1,2),(2),(1,2),(1)]. Must be in order. If there are only shifts in 1d: 
        kind_shifts = [(0,)]*ndata
    
extras: extra quantities for the lkl - tuple. 
    example: (par1,par2,par3,..)
    if only 1 parameter to the lkl: extras = (par1,)
    It enters as *args in lkl function
    
want_rescaling: whether to include or not rescaling parameters. True or False
        
    
Returns
---------
A BACCUS object. It has a method run_chain to run the mcmc chain:

run_chain: Initialize the sampler and run the MCMC chain
                
        parameters::DATA, stepsize=2.4, pos=None, nwalkers=300, steps=5000
        - - - - - - -
        
        DATA: a list of length ndata. Each entry of the ndata should 
            be a matrix with the independent variable in the first column 
            and the measuarements in subsequent columns
            
        stepsize: size of the jump for the affine sampling in emcee
        
        pos: Initial position of each parameters in each walker:
            Example for the pos parameter:
            pos = []
            for i in range(0,nwalkers):
                pos += [np.array([pos P1,pos P2,...])]
                if want_rescaling:
                for j in range(0,ndata):
                    pos[i] = np.append(pos[i],pos Resc_i)
                for j in range(0,ndata):
                pos[i] = np.append(pos[i],Pos Shift1)
                pos[i] = np.append(pos[i],Pos Shift2)
                ...
                pos[i] = np.append(pos[i],Pos var1)
                pos[i] = np.append(pos[i],Pos var2)
                ...
                pos[i] = np.append(pos[i],Pos Rho1)
                pos[i] = np.append(pos[i],Pos Rho2)
                ...
                
        nwalkers: Number of walkers in the mcmc
        
        steps: Number of steps for each walker in the mcmc
            Total of points: nwalkers*steps
       


To get the output and the chain, you can use the method get_results:

get_results: Output the mcmc, a log of the results and the behaviour of the mcmc
    
        parameters:: want_print=True,want_plots=False,root='output_'
        
        want_print: If True, prints a log of the results and behaviour of the mcmc
        
        want_plots: If True, saves plots (3 png) of the mean value of each parameter at each 
            step in all the walkers, a histogram of the accepted fraction (in each walker) 
            and all the values of the params at each step in each walker
            
        root: path to save the output



Examples
---------
Examples can be found in the examples folder. There are both a python code and an ipython notebook


tests
-----
```
python tests/baccus_test.py
```

installation
-------------
```
git clone <DIRECTION>
cd BACCUS 
python setup.py install

# or in a user-defined prefix
python setup.py install --prefix=/some/path
```

Test installation: python tests/baccus_test.py

Required dependencies
----------------------
numpy, emcee, matplotlib, pymc3, theano


Usage
-----
When used, please refer to the github page and cite arXiv:1803.04470/JCAP


