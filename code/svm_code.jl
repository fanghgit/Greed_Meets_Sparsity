using SparseArrays
using Printf
using LinearAlgebra

function svm_greedy(K, y, C , maxiter = 1e4)
    Q = copy(K)
    n = size(X, 1)
    
    counter = zeros(n)
    
    for i = 1:n
        for j = 1:n
            Q[i,j] = Q[i,j]*y[i]*y[j] 
        end
    end
    
    alpha = zeros(n)
    z = zeros(n)
    delta = zeros(n)
    
    @printf "preprocess complete! \n"
    
    for iter = 1:maxiter
        
        if( (iter-1) % 1e2 == 0 )
            obj = 0.5* dot( alpha, z ) - sum(alpha)   
            @printf "iter: %4.1i,  obj %4.3f\n" iter obj
        end
        
        # find coordinate with largest gradient
        idx = -1
        max_g = -Inf
        for i = 1:n
            delta[i] = max( -alpha[i], min( C - alpha[i],  (1 - z[i])/Q[i,i]  ) ) 
            if( abs(delta[i] ) > max_g )
                idx = i
                max_g = abs(delta[i])
            end
        end
        
        # update
        alpha[idx] += delta[idx]
        z = z + delta[idx]*Q[:,idx]
        
        # counter
        counter[idx] += 1
    
        
    end
    
    return alpha, counter
     
end



