import sys
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from cvxopt import matrix
from cvxopt import solvers
from docplex.mp.model import Model

class logreg_gcd():
    def __init__(self, maxiter = 1000, lamb=1.0, delta=1.0, L=0.25, ini = 'zero', rule = 1, verbose = False ):
        self.maxiter = maxiter
        self.lamb = lamb
        self.delta = delta
        self.L = L
        self.ini = ini
        self.rule = rule
        self.verbose = verbose
        self.sol = None
        self.counter = None
        self.obj_list = None
        self.W = None
        self.idx_list = []
        self.nnz_sol = []
        self.L1dist_to_sol = []
        self.L2dist_to_sol = []
        self.n_inactive = []
        
    def select_rule(self, x, g):
        assert np.ndim(x) == 1
        assert np.ndim(g) == 1
        m = len(x)
        x_new = x - g/self.L
        for i in range(m):
            if np.abs( x_new[i] ) <= self.lamb/self.L:
                x_new[i] = 0
            else:
                x_new[i] = x_new[i] - np.sign(x_new[i])*self.lamb/self.L
        d = x_new - x
        # choose the largest, set Q = -Q
        Q = -( 0.5*self.L*d**2 + g*d + self.lamb*np.abs( x_new ) - self.lamb*np.abs(x) )
        assert np.ndim(Q) == 1
        if np.min(Q) < -1e-5:
            ii = np.argmin(Q)
            print("x: ", x[ii], " x_new: ", x_new[ii], "g: ", g[ii])
            print("min Q: ", np.min(Q))
        assert np.min(Q) >= -1e-5
        return Q
    
    def get_obj(self, X, y, w):
        n, m = X.shape
        res = self.lamb * norm( w, 1 )
        Xw = np.dot(X, w)
        yXw = y * Xw
        exp_myXw = np.exp( -yXw )
        res += np.sum( np.log(1 + exp_myXw) )
        return res
    
    def get_grad(self, X, y, w):
        n, m = X.shape
        Xw = np.dot(X, w)
        yXw = y * Xw
        sigm = 1 / ( 1 + np.exp( -yXw ) )
        tmp = (sigm - 1.) * y
        g = np.dot( X.T, tmp )
        return g
        
    def fit(self, X, y):
        # minimize 0.5 \| Ax -b \|_2^2 + lamb |x|_1
        n, m = X.shape
        counter = np.zeros( m )
        if self.ini == 'zero':
            w = np.zeros( m )
        else:
            w = np.random.randn( m )
        W = []
        obj_list = []
        for it in range(self.maxiter):
            g = self.get_grad(X, y, w)
            idx = -1
            # selection
            Vx = self.select_rule( w, g )
            max_idx = np.argmax( Vx )
            max_Vx = Vx[max_idx]
            # delta-selection rule
            max_Vx_W = -1.0
            max_idx_W = -1
            if len(W) > 0:
                max_idx_W = np.argmax( Vx[W] )
                max_idx_W = W[ max_idx_W ]
                max_Vx_W = Vx[max_idx_W]
            if max_Vx_W >= self.delta*max_Vx:
                idx = max_idx_W
            else:
                idx = max_idx
         
            if counter[idx] ==0:
                W.append( idx )
                counter[idx] += 1
                
            self.idx_list.append( idx )
            W = list( np.sort(W) )
            
            # update
            w_old = w[idx]
            w[idx] -= 1/self.L * g[idx]
            if np.abs( w[idx] ) <= self.lamb/self.L:
                w[idx] = 0
            else:
                w[idx] -= np.sign( w[idx] )*self.lamb/self.L
                
            if(w[idx] == w_old):
                print("optimal solution foud!")
                break
            
            
            if( it % 10 == 0 ):
                obj = self.get_obj(X, y, w)
                if not self.verbose:
                    print( "iter: %4.1i,  obj %4.8f\n" % (it, obj) )
                obj_list.append( obj )
                # update nnz_sol
                self.nnz_sol.append( sum( w != 0 ) )
        
        self.sol = w
        self.counter = counter
        self.obj_list = obj_list
        self.W = W
        print( "optimization finished!" )

        
# a better implementation with heap
class logreg_fastgcd():
    def __init__(self, maxiter = 1000, lamb=1.0, delta=1.0, L=0.25, ini = 'zero', rule = 1, verbose = False ):
        self.maxiter = maxiter
        self.lamb = lamb
        self.delta = delta
        self.L = L
        self.ini = ini
        self.rule = rule
        self.verbose = verbose
        self.sol = None
        self.counter = None
        self.obj_list = None
        self.W = None
        self.idx_list = []
        self.nnz_sol = []
        self.L1dist_to_sol = []
        self.L2dist_to_sol = []
        self.n_inactive = []
        
    def select_rule(self, x, g):
        assert np.ndim(x) == 1
        assert np.ndim(g) == 1
        m = len(x)
        x_new = x - g/self.L
        for i in range(m):
            if np.abs( x_new[i] ) <= self.lamb/self.L:
                x_new[i] = 0
            else:
                x_new[i] = x_new[i] - np.sign(x_new[i])*self.lamb/self.L
        d = x_new - x
        # choose the largest, set Q = -Q
        Q = -( 0.5*self.L*d**2 + g*d + self.lamb*np.abs( x_new ) - self.lamb*np.abs(x) )
        assert np.ndim(Q) == 1
        if np.min(Q) < -1e-5:
            ii = np.argmin(Q)
            print("x: ", x[ii], " x_new: ", x_new[ii], "g: ", g[ii])
            print("min Q: ", np.min(Q))
        assert np.min(Q) >= -1e-5
        return Q
    
    def get_obj(self, X, y, w):
        n, m = X.shape
        res = self.lamb * norm( w, 1 )
        Xw = np.dot(X, w)
        yXw = y * Xw
        exp_myXw = np.exp( -yXw )
        res += np.sum( np.log(1 + exp_myXw) )
        return res
    
    def get_grad(self, X, y, w):
        n, m = X.shape
        Xw = np.dot(X, w)
        yXw = y * Xw
        sigm = 1 / ( 1 + np.exp( -yXw ) )
        tmp = (sigm - 1.) * y
        g = np.dot( X.T, tmp )
        return g
        
    # maintain g_W and g_q
    def fit(self, X, y):
        # minimize 0.5 \| Ax -b \|_2^2 + lamb |x|_1
        n, m = X.shape
        counter = np.zeros( m )
        if self.ini == 'zero':
            w = np.zeros( m )
        else:
            w = np.random.randn( m )
        W = []
        obj_list = []
        for it in range(self.maxiter):
            g = self.get_grad(X, y, w)
            idx = -1
            # selection
            Vx = self.select_rule( w, g )
            max_idx = np.argmax( Vx )
            max_Vx = Vx[max_idx]
            # delta-selection rule
            max_Vx_W = -1.0
            max_idx_W = -1
            if len(W) > 0:
                max_idx_W = np.argmax( Vx[W] )
                max_idx_W = W[ max_idx_W ]
                max_Vx_W = Vx[max_idx_W]
            if max_Vx_W >= self.delta*max_Vx:
                idx = max_idx_W
            else:
                idx = max_idx
         
            if counter[idx] ==0:
                W.append( idx )
                counter[idx] += 1
                
            self.idx_list.append( idx )
            W = list( np.sort(W) )
            
            # update
            w_old = w[idx]
            w[idx] -= 1/self.L * g[idx]
            if np.abs( w[idx] ) <= self.lamb/self.L:
                w[idx] = 0
            else:
                w[idx] -= np.sign( w[idx] )*self.lamb/self.L
                
            if(w[idx] == w_old):
                print("optimal solution foud!")
                break
            
            
            if( it % 10 == 0 ):
                obj = self.get_obj(X, y, w)
                if not self.verbose:
                    print( "iter: %4.1i,  obj %4.8f\n" % (it, obj) )
                obj_list.append( obj )
                # update nnz_sol
                self.nnz_sol.append( sum( w != 0 ) )
        
        self.sol = w
        self.counter = counter
        self.obj_list = obj_list
        self.W = W
        print( "optimization finished!" )