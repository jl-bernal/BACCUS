"""
Trivial example to test the code
"""

import numpy as np
import matrix_interp
from sklearn.gaussian_process.kernels import *

def test():
    
    #Create a Simulation Design Points
    a = np.arange(10)
    DESIGN = np.zeros([len(a),2])
    DESIGN[:,0] = a
    np.random.shuffle(a)
    DESIGN[:,1] = a
    
    #Create positive definite random matrices for each point
    TARGET = [0]*DESIGN.shape[0]
    for i in range(0,len(TARGET)):
        TARGET[i] = np.random.rand(30,30)
        TARGET[i] = TARGET[i]+TARGET[i].T
        TARGET[i] += np.eye(30)*30
        
    m = matrix_interp.matrix_interp(DESIGN,TARGET,kernel=Matern(),n_restarts=0)
    
    m.print_info()
    
    p = np.array([1.3,2.64])
    
    predicted = m.predict(p)
    
    print predicted
    
    
if __name__=="__main__":
    test()