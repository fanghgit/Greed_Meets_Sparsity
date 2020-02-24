import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from sklearn.linear_model import Lasso


# naive implementation
class lasso_gcd():
    def __init__(self, maxiter = 1000, lamb=1.0, delta=1.0, L=1.0, ini = 'zero', ini_level = 1.0, rule = 1, verbose = False, interval = 10 ):
        self.maxiter = maxiter
        self.lamb = lamb
        self.delta = delta
        self.L = L
        self.ini = ini
        self.ini_level = ini_level
        self.rule = rule
        self.verbose = verbose
        self.sol = None
        self.counter = None
        self.obj_list = None
        self.W = None
        self.interval = interval
        self.L1dist_to_sol = []
        self.grad_L1dist_to_sol = []
        self.grad_Linfdist_to_sol = []
        self.dual_angle = []
        self.n_inactive = []
        self.n_inactive_grad = []
        self.screening_n_inactive = []
        self.idx_list = []
        self.nnz_sol = []
        self.max_g_W = []
        self.bound = []
        
    def check_inactive(self, x, g):
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
        return sum(d == 0)
    
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
    
    def get_delta(self, g_opt):
        return self.lamb - np.abs( g_opt )
        
        
    def fit(self, A, b, x_ = None, delta = None):
        # minimize 0.5 \| Ax -b \|_2^2 + lamb |x|_1
        n, m = A.shape
        counter = np.zeros( m )
        if self.ini == 'zero':
            x = np.zeros( m )
        elif self.ini == 'ls':
            #x = np.linalg.lstsq(A, b)
            #x = x[0]
            x = np.linalg.solve( np.dot(A.T, A) + self.lamb*np.eye(m), np.dot(A.T, b) )
        else:
            x = np.random.randn( m )*self.ini_level
        W = []
        obj_list = []
        
        # dual variable theta
        #theta = np.dot(A, x) - b
        
        for it in range(self.maxiter):
            if( it % self.interval == 0 ):
                obj = 0.5*norm( A.dot(x) - b, 2 )**2 + self.lamb*norm(x, 1)
                if not self.verbose:
                    print( "iter: %4.1i,  obj %4.3f\n" % (it, obj) )
                obj_list.append( obj )
                # update nnz_sol
                self.nnz_sol.append( sum( x != 0 ) )
            
            #Ax = np.dot(A, x)
            Ax = A.dot(x)
            g =  (A.T).dot( Ax - b )
            #print("# inactive: %4.1i\n" % ( check_inactive(x, g) ))
            
            
            #n_active = np.sum( np.abs(g) > self.lamb )
            #print( "num active: %4.1i\n" % (n_active) )
            # calculate angle
            #theta_new = Ax - b
            #angle = np.dot( theta, theta_new ) / ( norm(theta, 2)*norm(theta_new, 2) )
            #self.dual_angle.append( angle )
            
            idx = -1
            # selection
            Vx = self.select_rule( x, g )
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
            x[idx] -= 1/self.L * g[idx]
            if np.abs( x[idx] ) <= self.lamb/self.L:
                x[idx] = 0
            else:
                x[idx] -= np.sign( x[idx] )*self.lamb/self.L
                
            if x_ is not None:
                self.L1dist_to_sol.append( norm(x - x_, 1) )
                g_opt = (A.T).dot( A.dot(x_) - b )
                dd = self.lamb - np.abs(g_opt)
                self.grad_Linfdist_to_sol.append( norm( g - g_opt , np.inf) ) 
                self.n_inactive_grad.append(  sum(  np.abs(g - g_opt) < dd  )  )
                self.bound.append( sum(norm( x - x_,1 ) > dd) + it )
            self.n_inactive.append( self.check_inactive(x, g) )
            if delta is not None:
                g_opt = A.T.dot( A.dot(x_) - b )
                self.screening_n_inactive.append(  sum(np.abs(g) < self.lamb - delta )  )
                
                
            g_max_idx = np.argmax( np.abs(g) )
            if g_max_idx not in self.max_g_W:
                self.max_g_W.append( g_max_idx )
            
            
        self.sol = x
        self.counter = counter
        self.obj_list = obj_list
        self.W = W
        print( "optimization finished!" )

# totally corrective greedy algorithm implementation
class lasso_omp():
    def __init__(self, maxiter = 1000, lamb=1.0, delta=1.0, L=1.0, ini = 'zero', rule = 1, verbose = False ):
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
        self.L1dist_to_sol = []
        self.dual_angle = []
        self.n_inactive = []
        self.idx_list = []
        self.nnz_sol = []
        
    def select_rule(self, x, g):
        assert np.ndim(x) == 1
        assert np.ndim(g) == 1
        m = len(x)
        x_new = x - g/self.L
        for i in range(m):
            if np.abs( x_new[i] ) <= self.lamb:
                x_new[i] = 0
            else:
                x_new[i] = x_new[i] - np.sign(x_new[i])*self.lamb
        d = x_new - x
        # choose the largest, set Q = -Q
        Q = -( 0.5/self.L*d**2 + g*d + self.lamb*np.abs( x_new ) - self.lamb*np.abs(x) )
        assert np.ndim(Q) == 1
        if np.min(Q) < -1e-5:
            ii = np.argmin(Q)
            print("x: ", x[ii], " x_new: ", x_new[ii], "g: ", g[ii])
            print("min Q: ", np.min(Q))
        assert np.min(Q) >= -1e-5
        return Q
    
    def fit(self, A, b):
        n, m = A.shape
        W = []
        x = np.zeros( m )
        Ax = np.dot(A, x)
        g = np.dot( A.T, Ax - b )
        Vx = self.select_rule( x, g )
        max_idx = np.argmax( Vx )
        W.append( max_idx )
        for i in range(m):
            A_sub = A[:,W]
            model = Lasso(alpha=self.lamb/n, fit_intercept=False, max_iter=10000, tol=1e-12, selection='cyclic')
            model.fit(A_sub, b)
            x[W] = model.coef_
            
            Ax = np.dot(A, x)
            g = np.dot( A.T, Ax - b )
            Vx = self.select_rule( x, g )
            max_idx = np.argmax( Vx )
            if Vx[max_idx] < 1e-6:
                break
            elif max_idx in W:
                print("max_idx in W")
                sys.exit(1)
            elif i > self.maxiter:
                break
            else:
                print( max_idx )
                print( Vx[max_idx] )
                W.append( max_idx )
        self.sol = x
        self.W = W
        
# naive implementation
class elastic_net_gcd():
    def __init__(self, maxiter = 1000, alpha = 1.0, l1_ratio=1.0, delta=1.0, L=1.0, ini = 'zero', rule = 1, verbose = False ):
        self.maxiter = maxiter
        self.l1_ratio = l1_ratio
        self.alpha = alpha
        self.delta = delta
        self.L = L + alpha*(1 - l1_ratio)
        self.ini = ini
        self.rule = rule
        self.verbose = verbose
        self.sol = None
        self.counter = None
        self.obj_list = None
        self.W = None
        self.L1dist_to_sol = []
        self.dual_angle = []
        self.n_inactive = []
        self.idx_list = []
        self.nnz_sol = []
        
    
    def select_rule(self, x, g):
        assert np.ndim(x) == 1
        assert np.ndim(g) == 1
        m = len(x)
        x_new = x - g/self.L
        for i in range(m):
            if np.abs( x_new[i] ) <= self.alpha*self.l1_ratio:
                x_new[i] = 0
            else:
                x_new[i] = x_new[i] - np.sign(x_new[i])*self.alpha*self.l1_ratio
        d = x_new - x
        # choose the largest, set Q = -Q
        Q = -( 0.5/self.L*d**2 + g*d + self.alpha*self.l1_ratio*np.abs( x_new ) - self.alpha*self.l1_ratio*np.abs(x) )
        assert np.ndim(Q) == 1
        if np.min(Q) < -1e-5:
            ii = np.argmin(Q)
            print("x: ", x[ii], " x_new: ", x_new[ii], "g: ", g[ii])
            print("min Q: ", np.min(Q))
        assert np.min(Q) >= -1e-5
        return Q
        
        
    def fit(self, A, b):
        # minimize 0.5 \| Ax -b \|_2^2 + lamb |x|_1
        n, m = A.shape
        counter = np.zeros( m )
        if self.ini == 'zero':
            x = np.zeros( m )
        elif self.ini == 'ls':
            #x = np.linalg.lstsq(A, b)
            #x = x[0]
            x = np.linalg.solve( np.dot(A.T, A) + self.alpha*self.l1_ratio*np.eye(m), np.dot(A.T, b) )
        else:
            x = np.random.randn( m )
        W = []
        obj_list = []
        
        # dual variable theta
        #theta = np.dot(A, x) - b
        
        for it in range(self.maxiter):
            if( it % 10 == 0 ):
                obj = 0.5*norm( A.dot(x) - b, 2 )**2 + self.alpha*self.l1_ratio*norm(x, 1) + 0.5*self.alpha*(1 - self.l1_ratio)*norm(x, 2)**2
                if not self.verbose:
                    print( "iter: %4.1i,  obj %4.3f\n" % (it, obj) )
                obj_list.append( obj )
                # update nnz_sol
                self.nnz_sol.append( sum( x != 0 ) )
            
            #Ax = np.dot(A, x)
            Ax = A.dot(x)
            g =  (A.T).dot( Ax - b ) + self.alpha*(1 - self.l1_ratio)*x
            
            #n_active = np.sum( np.abs(g) > self.lamb )
            #print( "num active: %4.1i\n" % (n_active) )
            # calculate angle
            #theta_new = Ax - b
            #angle = np.dot( theta, theta_new ) / ( norm(theta, 2)*norm(theta_new, 2) )
            #self.dual_angle.append( angle )
            
            idx = -1
            # selection
            Vx = self.select_rule( x, g )
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
            x[idx] -= 1/self.L * g[idx]
            if np.abs( x[idx] ) <= self.alpha*self.l1_ratio:
                x[idx] = 0
            else:
                x[idx] -= np.sign( x[idx] )*self.alpha*self.l1_ratio
                
            #self.n_inactive.append( self.check_inactive(x, g) )
            
        self.sol = x
        self.counter = counter
        self.obj_list = obj_list
        self.W = W
        print( "optimization finished!" )
        
        
        