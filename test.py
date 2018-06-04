from nonstationary_rbf import NonstationaryRBF
import numpy as np
import GPy

def test_constant_1d():    
    def lengthscalefun(X,dim):
        return 4.0
    k = NonstationaryRBF(1,lengthscalefun=lengthscalefun)
    X = np.arange(0.0,20.0,1)[:,None]
    Y = 10.0*np.sin((50*X[:,0:1])**0.4)+1*np.random.randn(X.shape[0],1)
    m1 = GPy.models.GPRegression(X,Y,k)
    m1.Gaussian_noise = 0.1
    k = GPy.kern.RBF(1,lengthscale=4.0)
    m2 = GPy.models.GPRegression(X,Y,k)
    m2.Gaussian_noise = 0.1
    assert m1.predict(np.array([[10.0]]))==m2.predict(np.array([[10.0]])), "Result doesn't match simple EQ kernel"

def test_1d():
    def lengthscalefun(X,dim):
        if X<25:
            return 4.0
        else:
            return 1.0

    k = NonstationaryRBF(1,lengthscalefun=lengthscalefun)
    X = np.arange(0.0,50.0,1)[:,None]
    Y = 10.0*np.sin((50*X[:,0:1])**0.4)+1*np.random.randn(X.shape[0],1)
    m1 = GPy.models.GPRegression(X,Y,k)
    m1.Gaussian_noise = 0.1
    k = GPy.kern.RBF(1,lengthscale=4.0)
    m2 = GPy.models.GPRegression(X,Y,k)
    m2.Gaussian_noise = 0.1
    assert np.abs(m1.predict(np.array([[0.0]]))[0]-m2.predict(np.array([[0.0]]))[0])<0.01, "Result doesn't match simple EQ kernel"
    assert np.abs(m1.predict(np.array([[50.0]]))[0]-m2.predict(np.array([[50.0]]))[0])>0.01, "Result unexpectedly matches simple EQ kernel"
    
def test_2d():
    def lengthscalefun(X,dim):
        if dim==0:
            return 4.0
        else:
            return 0.5

    k = NonstationaryRBF(2,lengthscalefun=lengthscalefun)
    Xa = np.arange(0.0,20.0,1)[:,None]
    Xb = 20*np.random.rand(len(Xa),1)
    X = np.c_[Xa,Xb]
    Y = 10.0*np.sin((50*X[:,0:1])**0.4)+1*np.random.randn(X.shape[0],1)
    m1 = GPy.models.GPRegression(X,Y,k)
    m1.Gaussian_noise = 0.1
    k = GPy.kern.RBF(2,ARD=True,lengthscale=[4.0,0.5])
    m2 = GPy.models.GPRegression(X,Y,k)
    m2.Gaussian_noise = 0.1
    assert np.abs(m1.predict(np.array([[10.0,10.0]]))[0]-m2.predict(np.array([[10.0,10.0]]))[0])<1e-10, "Result does't match ARD EQ kernel"

