import sys
import numpy as np
from numpy.linalg import norm
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import linear_kernel
from cvxopt import matrix
from cvxopt import solvers
from docplex.mp.model import Model

class svm_gcd():
    def __init__(self, maxiter = 1000, C=1.0, delta=1.0, L=1.0, ini = 'zero', rule = 1, gamma=1.0, r=1.0, kernel="linear", verbose = False ):
        self.maxiter = maxiter
        self.C = C
        self.delta = delta
        self.L = L
        self.ini = ini
        self.rule = rule
        self.gamma = gamma
        self.r = r
        self.kernel = kernel
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
    
    def min_exp_set(self, X, y, S = []):
        n, m = X.shape
        # linear kernel
        if self.kernel == "linear":
            Q = np.dot( X, X.T )
            Q = Q.todense()
            #print(Q.shape)
        elif self.kernel == "rbf":
            self.gamma = 1.0/m
            Q = rbf_kernel( X, X, gamma = self.gamma )
        else:
            print("invalid kernel!")
            sys.exit(0)
        for i in range(n):
            Q[:,i] *= ( y[i] * y )
        Q = Q.tolist()
        if len(S) == 0:
            print("need warm start!")
            sys.exit(0)
        # variables [x_S, d, z, r]
        # S+1 variables
        S_size = len(S)
        S_exp = []
        N = [i for i in range(0, n)]
        for i in range(0, n):
            if i in S:
                continue 
            Other_idx = [j for j in range(0,n) if j not in S and j != i]
            mdl = Model("expand")
            x = mdl.continuous_var_dict(S, lb=0.01, ub=self.C, name='x')
            #d = mdl.continuous_var_dict(N, name='d')
            #z = mdl.binary_var_dict(N, name='z')
            r = mdl.continuous_var( name='r' )
            # set objective
            mdl.maximize( r )
            # add constraints
            mdl.add_constraint( (mdl.sum( Q[i][j]*x[j] for j in S )-1.0)/Q[i][i] <= -r )
            mdl.add_constraints( (mdl.sum( Q[k][j]*x[j] for j in S )-1.0)/Q[k][k] >= -self.C+1e-2 for k in Other_idx )
            solution=mdl.solve(log_output=False)
            res = solution.get_objective_value()
            if res > self.C:
                print( str(i) + ": " + str(res) )
                S_exp.append( i )
        return S_exp
            # add constraints
            # z = 1 -> g > 0, z = 0 -> g <= 0
            #mdl.add_indicator(   )
            #mdl.add_indicator_constraints( mdl.indicator_constraint(z[k], \ 
            #                                d[k] >= (mdl.sum( Q[k][j]*x[j] for j in S )-1.0)/Q[k][k] ) \
            #                              for k in N )

    def get_delta(self, X, y, alpha):
        n, m = X.shape
        # linear kernel
        if self.kernel == "linear":
            Q = np.dot( X, X.T )
            Q = Q.todense()
            #print(Q.shape)
        elif self.kernel == "rbf":
            self.gamma = 1.0/m
            Q = rbf_kernel( X, X, gamma = self.gamma )
        else:
            print("invalid kernel!")
            sys.exit(0)
        for i in range(n):
            Q[:,i] *= ( y[i] * y )
        delta = np.dot(Q, alpha) - np.ones( n )
        return delta
        
    def exact_fit(self, X, y):
        n, m = X.shape
        # linear kernel
        if self.kernel == "linear":
            Q = np.dot( X, X.T )
            Q = Q.todense()
            #print(Q.shape)
        elif self.kernel == "rbf":
            self.gamma = 1.0/m
            Q = rbf_kernel( X, X, gamma = self.gamma )
        else:
            print("invalid kernel!")
            sys.exit(0)
        for i in range(n):
            Q[:,i] *= ( y[i] * y )
        #Q = matrix(Q)
        q = -np.ones( n )
        G1 = np.eye( n )
        G2 = -np.eye( n )
        G = np.vstack( (G1, G2) )
        h = np.zeros( 2*n )
        h[:n] = self.C
        Q = matrix(Q)
        G = matrix(G)
        q = matrix(q)
        h = matrix(h)
        opts = {'maxiters' : 50, 'reltol':1e-12}
        sol = solvers.qp(Q,q,G,h, options = opts)
        return np.array(sol['x']), sol['primal objective']
        
        
        
    def get_obj(self, X, y, alpha):
        n, m = X.shape
        # linear kernel
        if self.kernel == "linear":
            Q = np.dot( X, X.T )
            Q = Q.todense()
        elif self.kernel == "rbf":
            self.gamma = 1.0/m
            Q = rbf_kernel( X, X, gamma = self.gamma )
        else:
            print("invalid kernel!")
            sys.exit(0)
        for i in range(n):
            Q[:,i] *= ( y[i] * y )
            
        z = np.dot(Q, alpha)
        obj = 0.5*np.dot( alpha, z ) - sum(alpha)
        return obj
        
        
    def fit(self, X, y, alpha_opt = None):
        n, m = X.shape
        # linear kernel
        if self.kernel == "linear":
            Q = linear_kernel(X, X)
        elif self.kernel == "rbf":
            self.gamma = 1.0/m
            Q = rbf_kernel( X, X, gamma = self.gamma )
        else:
            print("invalid kernel!")
            sys.exit(0)
        for i in range(n):
            Q[:,i] *= ( y[i] * y )
        
        W = []
        if self.ini == 'zero':
            alpha = np.zeros( n )
            z = np.zeros( n )
        else:
            alpha = np.random.randn( n )
            z = np.dot( Q, alpha )
        d = np.zeros( n )
        #grad = np.zeros( n )
        counter = np.zeros( n )
        obj_list = []
        # test delta
        if alpha_opt is not None:
            g_star = np.dot(Q, alpha_opt) - np.ones( n )
            Z = [i for i in range(0, n) if alpha_opt[i] == 0]
            
        
        for it in range(self.maxiter):
            idx = -1
            max_g_W = -float("inf")
            # find the best in W
            for i in range(n):
                d[i] = max( -alpha[i], min( self.C - alpha[i],  (1 - z[i])/Q[i,i]  ) )
                #grad[i] = z[i] - 1
            # delta-selection rule
            max_idx = np.argmax( np.abs(d) )
            max_Vx = d[max_idx]
            # delta-selection rule
            max_Vx_W = -1.0
            max_idx_W = -1
            if len(W) > 0:
                max_idx_W = np.argmax( np.abs(d[W]) )
                max_idx_W = W[ max_idx_W ]
                max_Vx_W = d[max_idx_W]
            if max_Vx_W >= self.delta*max_Vx:
                idx = max_idx_W
            else:
                idx = max_idx            
            
            # update
            alpha[idx] += d[idx]
            #if( alpha[idx] == 0 ):
            #    print("0 occurs!s")
            z += d[idx]*Q[:,idx]
            counter[idx] += 1
            if idx not in W:
                W.append( idx )
                
            if alpha_opt is not None:
                self.L1dist_to_sol.append(  sum( np.abs(alpha - alpha_opt) )  )
                self.L2dist_to_sol.append(  np.sqrt( sum( (alpha - alpha_opt)**2 ) )  )
            inactive_idx = [ ii for ii in range(n) if alpha[ii]==0 and 1-z[ii] < 0 ]
            self.n_inactive.append( len(inactive_idx) )
                #tmp = np.abs( z - np.ones(n) - g_star )
                #inactive_idx = [i for i in Z if np.abs(tmp[i]) <= g_star[i]/2 ]
                #self.n_inactive.append( len(inactive_idx) )
            
            if( it % 1 == 0 ):
                obj = 0.5*np.dot( alpha, z ) - sum(alpha)
                if not self.verbose:
                    print( "iter: %4.1i,  obj %4.3f\n" % (it, obj) )
                obj_list.append( obj )
                self.nnz_sol.append( sum( alpha != 0 ) )
                    
        self.W = W
        self.sol = alpha
        self.counter = counter
        self.obj_list = obj_list
        print( "optimization finished!" )
                    
                    
                    