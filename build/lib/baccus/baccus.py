# -*- coding: utf-8 -*-
"""
BACCUS: BAyesian Conservative Constraints and Unknown Systematics.
J. L. Bernal and J. A. Peacock, 2018
arXiv: ?????
"""

import numpy as np
import emcee
import matplotlib.pyplot as plt
from numpy.random import normal
from numpy.linalg import inv
from numpy.linalg import det
import pymc3 as pm
import theano
import theano.tensor as T

class BACCUS(object):
    """
    A class to perform a MCMC following the formalism of BACCUS introduced in
    arxiv:1803.04470 to obtain conservative constraints allowing for systematics
        
        
    Parameters
    -----------
        
    prior_var: log(prior) of the diagonal elements of the covariance matrix
        of the shift parameters (variance). list of theano functions
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
        
    
    Methods
    --------
	check_input: Check mandatory input is included
    
    
    
	prepare_prior: Prepare the prior_min and prior_out of all parameters 
	    and the prior functions for var and rhos in case is none.
	
	
	
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
			    pos[i] = np.append(pos[i],Pos Var1)
			    pos[i] = np.append(pos[i],Pos Var2)
			    ...
			    pos[i] = np.append(pos[i],Pos Rho1)
			    pos[i] = np.append(pos[i],Pos Rho2)
			    ...
			    
		nwalkers: Number of walkers in the mcmc
		
		steps: Number of steps for each walker in the mcmc
		    Total of points: nwalkers*steps
		    
		    
	
	lnprob: Compute the log_lkl calling to lkl, log_prior and including the
	    priors of each parameters.
	    Called by sampler.
	    
	    parameters:: theta, DATA,prior_bounds,priors,nrescaling,nshifts,nsigmas,
                nrhos,kind_shifts,want_rescaling,extras,lkl
		
		
		theta: list with the sampled parameters
		
		DATA: inherited from run_chain
		
		prior_bounds: inherited from prepared priors. Bounds for each sampled
		    parameter
		    
		priors: log(prior) for any sampled parameter other than uniform.
		    List with [model_priors,var_priors,corr_priors]
		    
		nrescaling, nshifts, nsigmas, nrhos: Number (int) of rescaling (0 if
		    want_rescaling is False), shifts, var and corr parameters, respectively
		    
		kind_shifts: inherited from BACCUS class
		
		want_rescaling: inherited from BACCUS class
		
		extras: inherited from BACCUS class
		
		lkl: inherited from BACCUS class
		
		
		
	get_results: Output the mcmc, a log of the results and the behaviour of the mcmc
	
	    parameters:: want_print=True,want_plots=False,root='output_'
	    
		want_print: If True, prints a log of the results and behaviour of the mcmc
		
		want_plots: If True, saves plots (3 png) of the mean value of each parameter at each 
		    step in all the walkers, a histogram of the accepted fraction (in each walker) 
		    and all the values of the params at each step in each walker
		    
		root: path to save the output
    """
    
    def __init__(self, prior_var=None, prior_corr=None, prior_model = [],prior_bounds_model = [],prior_bounds_shifts = [], prior_bounds_var = [], lkl = None,kind_shifts = [],extras=(),want_rescaling=False):
        
        
        
        self.prior_var = prior_var
        self.prior_corr = prior_corr
        self.prior_model = prior_model
        self.prior_bounds_model = prior_bounds_model
        self.prior_bounds_shifts = prior_bounds_shifts
        self.prior_bounds_var = prior_bounds_var
        self.lkl = lkl
        self.kind_shifts = kind_shifts
        self.extras = extras
        self.want_rescaling = want_rescaling
        
        ndata = len(kind_shifts)
        self.ndata = ndata
        nshifts = 0
        for i_nd in range(0,ndata):
            nshifts += len(kind_shifts[i_nd])
        
        self.nshifts = nshifts
        if want_rescaling:
            self.nrescaling = ndata
        else:
            self.nrescaling = 0
            
        self.npars_model = len(prior_bounds_model)
        self.check_input()


    def check_input(self):
        '''
        Check mandatory input is included
        '''
        prior_bounds_model = self.prior_bounds_model
        prior_bounds_shifts = self.prior_bounds_shifts
        prior_bounds_var = self.prior_bounds_var
        lkl = self.lkl
        kind_shifts = self.kind_shifts
        
        if len(prior_bounds_model) == 0:
            raise ValueError("You must set the bounds of the prior of model parameters")
        if len(prior_bounds_shifts) == 0:
            raise ValueError("You must set the bounds of the prior of shift parameters")
        if len(prior_bounds_var) == 0:
			raise ValueError("You must set the bounds of the prior of variance of shift parameters")
        if lkl is None:
            raise ValueError("You must specify the likelihood of your model")
        if len(kind_shifts) == 0:
            raise ValueError("You must specify what shift is used \
            in each data set")
        return
            
            
    def prepare_prior(self):
        '''
        Prepare the prior_min and prior_out of all parameters and parameters
        and the prior functions for var and rhos in case is none.
        '''
        
        #Prepare the prior bounds
        
        nshifts = self.nshifts
        nrescaling = self.nrescaling
        npars_model = self.npars_model
        prior_bounds_model = self.prior_bounds_model
        prior_bounds_shifts = self.prior_bounds_shifts
        prior_bounds_var = self.prior_bounds_var
        kind_shifts = self.kind_shifts
                
        
        nsigmas = len(prior_bounds_var)
        nrhos = int(nsigmas*(nsigmas-1.)/2.)
        ntot = int(nshifts+nrescaling+npars_model+nsigmas+nrhos)
        
        self.nsigmas = nsigmas
        self.nrhos = nrhos
        
        prior_min = np.zeros(ntot)
        prior_max = np.zeros(ntot)
        
        #Structure of priors: prior_model
        for i in range(0,npars_model):
            prior_min[i] = prior_bounds_model[i][0]
            prior_max[i] = prior_bounds_model[i][1]
        
        #prior_rescaling
        if self.want_rescaling:
            prior_min[npars_model:npars_model+nrescaling] = np.zeros(nrescaling)
            prior_max[npars_model:npars_model+nrescaling] = np.ones(nrescaling)*1000.

		#prior_shift
        which_shift = []
        var_min = np.zeros(nsigmas)
        var_max = np.zeros(nsigmas)
        
        ind = npars_model + nrescaling
        
        for i in range(0,len(kind_shifts)):
            ll = len(kind_shifts[i])
            for il in range(0,ll):
                which_shift += [kind_shifts[i][il]]   

        
        for i in range(0,nshifts):
            prior_min[ind+i] = prior_bounds_shifts[which_shift[i]][1]
            prior_max[ind+i] = prior_bounds_shifts[which_shift[i]][2]
        
        #prior var
        ind += nshifts
        
        for ivar in range(0,nsigmas):
            var_min[ivar] = prior_bounds_var[ivar][1]
            var_max[ivar] = prior_bounds_var[ivar][2]                    
        var_min[var_min == 0] += 1e-8
        
        prior_min[ind:ind+nsigmas] = var_min
        prior_max[ind:ind+nsigmas] = var_max
        
        #prior rhos    
        ind += nsigmas
        
        prior_min[ind:ind+nrhos] = -1+1e-8
        prior_max[ind:ind+nrhos] = 1-1e-8
        
        
        self.prior_bounds = [prior_min,prior_max]
        
        #Check prior distributions for var and rho
        prior_var = self.prior_var
        prior_corr = self.prior_corr
        
        #if None, prior_var -> Lognormal (log(s) = N(b,xi))
        if prior_var == None:
            prior_var = []
            b = -6.
            xi = 200.
            y = T.dscalar('y')
            s = -((T.log(y)-b)/2./xi)**2.
            logprior_sigma = theano.function([y],s)
            for ivar in range(0,nsigmas):
                prior_var += [logprior_sigma]
                
            self.prior_var = prior_var
        
        #if None, prior_corr -> LKJ with eta = 1.
        if prior_corr == None:
            dim = nsigmas
            distri = pm.LKJCorr.dist(eta=1.,n=dim)
            y = T.dvector('y')
            s = distri.logp(y)
            LKJ = theano.function([y],s)
            
            self.prior_corr = LKJ
            
        return
        
                
    def run_chain(self, DATA, stepsize=2.4, pos=None, nwalkers=300, steps=5000):
        """
        Initialize the sampler and run the MCMC chain
        
        DATA: a list of length ndata. Each entry of the ndata should be a matrix with
        the independent variable in the first column and the measuarements in subsequent
        columns
        
        Example for the pos parameter:
        pos = []
        for i in range(0,nwalkers):

            pos += [np.array([pos P1,pos P2,...])]
            for j in range(0,ndata):
                pos[i] = np.append(pos[i],pos Resc_i)
            for j in range(0,ndata):
                pos[i] = np.append(pos[i],Pos Shift1)
                pos[i] = np.append(pos[i],Pos Shift2)
                ...

            pos[i] = np.append(pos[i],Pos Std1)
            pos[i] = np.append(pos[i],Pos Std2)
            ...
            
            pos[i] = np.append(pos[i],Pos Rho1)
            pos[i] = np.append(pos[i],Pos Rho2)
            ...
            
        """
        #Prepare the priors
        self.prepare_prior()
        
        self.nwalkers = nwalkers
        self.steps = steps
        
        if pos == None:
            raise ValueError("You need to specify the initial position of each parameter")
        
        prior_bounds = self.prior_bounds
        ndim = len(prior_bounds[1])
        
        prior_model = self.prior_model
        prior_var = self.prior_var
        prior_corr = self.prior_corr
        
        priors = [prior_model,prior_var,prior_corr]
        
        extras = self.extras
        nsigmas = self.nsigmas
        nrhos = self.nrhos
        nshifts = self.nshifts
        nrescaling = self.nrescaling
        kind_shifts = self.kind_shifts
        want_rescaling = self.want_rescaling
        
        
        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnprob, a = stepsize, args=
                                        (DATA,prior_bounds,priors,nrescaling,nshifts,
                                        nsigmas,nrhos,kind_shifts,want_rescaling,extras,self.lkl))
        
        print 'Total steps = ', nwalkers*steps
        
        # Clear and run the production chain.
        print("Running MCMC...")
        # synt: initial position, number of steps
        sampler.run_mcmc(pos, steps, rstate0=np.random.get_state())

        print("Done.")  
        
        self.INFO = sampler
        
        return
        
                
    @staticmethod
    def lnprob(theta, DATA,prior_bounds,priors,nrescaling,nshifts,nsigmas,
                nrhos,kind_shifts,want_rescaling,extras,lkl):
        """
        Compute the log_lkl calling to lkl, log_prior and inlcuding the
        priors of each parameters.
        Called by sampler.
        """
        prior_min = prior_bounds[0]
        prior_max = prior_bounds[1]
        
        prior_model = priors[0]
        prior_var = priors[1]
        prior_corr = priors[2]
        
        
        if (prior_min >= theta).any() == True or (theta >= prior_max).any() == True:
			return -np.inf
       
       
        #Apply the prior on the correlations
        lp_corr = 0
        if nrhos != 0:
            rhos = theta[-nrhos:]
            lp_corr = prior_corr(np.asarray(rhos))
        
        #Apply the prior on the variances
        ind = -nsigmas - nrhos
        if nrhos == 0:
            sigmas = theta[-nsigmas:]
        else:
			sigmas = theta[ind:ind+nsigmas]
        
        lp_var = 0
        for isig in range(0,nsigmas):
            sig = theta[ind+isig]
            lp_var += prior_var[isig](sig)

        #Check that params are inside the distributions
        if not np.isfinite(lp_var) or not np.isfinite(lp_corr):
			return -np.inf
            
        #Build the covariance matrix of the prior of the shifts
        n_elem = nrhos
        tri_index = np.zeros([nsigmas, nsigmas], dtype=int)
        tri_index[np.triu_indices(nsigmas, k=1)] = np.arange(n_elem)
        tri_index[np.triu_indices(nsigmas, k=1)[::-1]] = np.arange(n_elem)
        tri_index -= np.diag(np.ones(nsigmas)).astype(int)
        corr_matrix = np.zeros([nsigmas,nsigmas])
        for i in range(0,n_elem):
            corr_matrix[tri_index == i] = rhos[i]
        corr_matrix += np.diag(np.ones(nsigmas))
        var_matrix = np.diag(sigmas)
        
        COV = np.dot(var_matrix,np.dot(corr_matrix,var_matrix))
        iCOV = inv(COV)

        #Apply the prior on the shifts
        ndata = len(kind_shifts)
        ind -= nshifts
        count = 0
        lp_shift = 0
        for ik in range(0,ndata):
            #which shifts are applied in this data set?
            kind = kind_shifts[ik]
            nkind = len(kind)
            if nkind > 0:
                #Take correspondent shifts
                shifts_ik = theta[ind+count:ind+count+nkind]
                count += nkind
                #Take the correspondent submatrix of the inv. covmat
                iCOV_ik = iCOV[np.ix_(kind,kind)]
                lp_shift += -0.5*np.dot(shifts_ik,np.dot(iCOV_ik,shifts_ik))+0.5*np.log(det(iCOV_ik))

        #Apply the prior on the rescaling
        lp_rescale = 0
        if want_rescaling:
            ind -= nrescaling
            for ik in range(0,ndata):
                lendata = DATA[ik].shape[0]
                alpha = theta[ind+ik]
                lp_rescale += lendata*np.log(alpha)/2. - alpha
                
        
        #Apply priors in model parameters
        npriors = len(prior_model)  
        lp_model = 0
        if npriors != 0:
            for ip in range(0,npriors):
                par = theta[prior_model[ip][0]]
                lp_model += prior_model[ip][1](par)
                        
    
        #Compute log lkl
        log_lkl = lkl(theta, DATA,*extras)
        

        if not np.isfinite(log_lkl):
			return -np.inf
            
        log_post = log_lkl +lp_corr + lp_var + lp_shift + lp_rescale + lp_model
        
        return log_post
        
    
    def get_results(self,want_print=True,want_plots=False,root='output_'):
        """
        Output a log of the results and the behaviour of the mcmc
        
        root: path to save the output
        """
        INFO = self.INFO
        nwalkers = self.nwalkers
        steps = self.steps
        
        #Get the chain
        chain = INFO.chain
        
        #Get the bestfit
        best = INFO.flatchain[np.where(INFO.flatlnprobability == np.max(INFO.flatlnprobability))]
        
        nshifts = self.nshifts
        nrescaling = self.nrescaling
        npars_model = self.npars_model
        nsigmas = self.nsigmas
        nrhos = self.nrhos
        ntot = int(nshifts+nrescaling+npars_model+nsigmas+nrhos)
        
        bestfit = np.zeros(ntot + 1)
        bestfit[0] = - np.max(INFO.flatlnprobability)
        bestfit[1:] = np.atleast_2d(best)[0,:]

        #Get the chain
        mc_chain = np.zeros([nwalkers * steps, ntot + 1])
        mc_chain[:,0] = - INFO.flatlnprobability
        mc_chain[:,1:] = INFO.flatchain

        #Get the mean value of each parameter of the whole secquence 
            #of walkers at each step 
        mean_seq = np.zeros([steps,ntot])

        for i in range(0,steps):
            for j in range(0,ntot):
                mean_seq[i,j] = np.sum(chain[:,i,j]) / nwalkers

        #save the files
        np.savetxt(root + 'chain.txt', mc_chain, header = 'Flat chain of \
                    the walkers:\nC_0: -logLkl; The rest are the n parameters\
                    \nEach row is a position of a walker (i.e. a step)')

        f1 = open(root + 'log.txt', 'w')
        f1.write('####Log file of the chain####\n\nChain with ')
        f1.write(repr(nwalkers) + ' walkers and ' + repr(steps) + ' steps: Total steps = ' + repr(nwalkers * steps) + '\n\n')
        f1.write('bestfit of parameters:\n' + repr(bestfit[1:]))
        f1.write('\n\nAutocorrelation times for each parameter:\n' + repr(INFO.acor))
        f1.write('\n\nMean acceptance fraction' + repr(np.sum(INFO.acceptance_fraction) / nwalkers))
        f1.write('\n\nAnd the **unnormalized** minimum of the -Loglkl = ' + repr(bestfit[0]))
        f1.close()

        if want_print:
            print '\nChain with %i walkers and %i steps: Total steps = %i'%(nwalkers,steps,steps*nwalkers)
            print '\nbestfit = ', bestfit[1:]
            print '\nwith **unnormalized** min -logLkl = ', bestfit[0]
            print '\nMean acceptance fraction = %f'%(np.sum(INFO.acceptance_fraction) / nwalkers)

        if want_plots:
            #plot value of chain vs step
            f,ax = plt.subplots(ntot,1,figsize=(14,12))
            for i in range(0,ntot):
                ax[i].plot(chain[:,:,i].T,color ='k',alpha = 0.05)
                ax[i].tick_params(axis='x',labelcolor='white')
                ax[i].set_xlabel(r'$steps$', fontsize=14,labelpad=15)

            ax[-1].tick_params(axis='x',labelcolor='black')
            f.subplots_adjust(hspace = .2)
            f.subplots_adjust(bottom=0.18)
            f.subplots_adjust(left=0.18)
            plt.savefig(root + 'steps.png')

            #Plot acceptance fraction per chain (hist)
            f2,ax = plt.subplots()
            ax = plt.hist(INFO.acceptance_fraction, bins = 30, normed = True, histtype = 'step')
            plt.xlabel(r'Acceptance fraction', fontsize=24,labelpad=15)
            f2.subplots_adjust(bottom=0.18)
            f2.subplots_adjust(left=0.18)
            plt.savefig(root + 'Acc_frac_hist.png')

            #Plot mean sequence of all walkers vs steps
            f,ax = plt.subplots(ntot,1,figsize=(14,12))
            for i in range(0,ntot):
                ax[i].plot(np.linspace(0,steps-1,steps),mean_seq[:,i])
                ax[i].tick_params(axis='x',labelcolor='white')
                ax[i].set_xlabel(r'$steps$', fontsize=14,labelpad=15)

            ax[-1].tick_params(axis='x',labelcolor='black')
            f.subplots_adjust(hspace = .2)
            f.subplots_adjust(bottom=0.18)
            f.subplots_adjust(left=0.18)
            plt.savefig(root + 'mean_seq.png')


        return
