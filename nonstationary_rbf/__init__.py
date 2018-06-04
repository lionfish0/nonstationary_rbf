from __future__ import division
import GPy
import numpy as np
from GPy.kern import Kern
from GPy.core.parameterization import Param
from paramz.transformations import Logexp
import math

class NonstationaryRBF(Kern): #todo do I need to inherit from Stationary
    """
    Integral kernel between...
    """

    def __init__(self, input_dim, variances=1.0, lengthscale=1.0, ARD=False, active_dims=None, lengthscalefun=None, name='nonstatRBF'):
        super(NonstationaryRBF, self).__init__(input_dim, active_dims, name)

        if lengthscale is None:
            lengthscale = np.ones(1)
        else:
            lengthscale = np.asarray(lengthscale)

        if lengthscalefun is None:
            lengthscalefun = lambda x: lengthscale
            
        self.lengthscalefun = lengthscalefun
        self.lengthscale = Param('lengthscale', lengthscale, Logexp()) #Logexp - transforms to allow positive only values...
        self.variances = Param('variances', variances, Logexp()) #and here.
        self.link_parameters(self.variances, self.lengthscale) #this just takes a list of parameters we need to optimise.
        

    def update_gradients_full(self, dL_dK, X, X2):
        pass
    
    def kxx(self,X1,X2):
        
        assert isinstance(self.lengthscalefun(X1,0),float), "The lengthscalefunction should return a single, scalar, positive value." #this should only return one lengthscale
        p = 1.0
        s = 0.0
        for i,(x1,x2) in enumerate(zip(X1,X2)):
            s+=((x1-x2)**2/(self.lengthscalefun(X1,i)**2+self.lengthscalefun(X2,i)**2))
            p*= np.sqrt((2.0*self.lengthscalefun(X1,i)*self.lengthscalefun(X2,i))/(self.lengthscalefun(X1,i)**2+self.lengthscalefun(X2,i)**2))
        return self.variances*p*np.exp(-s)

    
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        k = np.zeros([X.shape[0],X2.shape[0]])
        for i,x1 in enumerate(X):
            for j,x2 in enumerate(X2):
                k[i,j] = self.kxx(x1,x2)
        return self.variances*k

    def Kdiag(self, X):
        return np.diag(self.K(X))
