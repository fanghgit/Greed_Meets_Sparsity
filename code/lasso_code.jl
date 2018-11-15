using SparseArrays

# to be optimized (using heap)
function lasso_greedy(A, b, lambda, maxiter = 1e4)
    n, m = size(A)
    x = zeros(m)
    counter = zeros(m)
    for iter = 1:maxiter
        Ax = A*x
        g = A'*(Ax - b)
        for j = 1:length(g)
            if( x[j] == 0 && g[j] >= -lambda && g[j] <= lambda )
                g[j] = 0
            else
                g[j] = abs( g[j] + sign(x[j])*lambda )
            end
        end
        # greedy coordinate
        r = sortperm(g, rev = true)
        idx = r[1]
        counter[idx] += 1
        Atmp = A[:,idx]*x[idx]
        aTa = dot(A[:,idx], A[:,idx])
        if(aTa == 0)
            x[idx] = 0
            continue
        end
        c = b - (Ax - Atmp)
        cTa = dot(c, A[:,idx])
        if( cTa >= -lambda && cTa <= lambda )
            x[idx] = 0
        else
            x1 = (cTa - lambda)/(aTa)
            x2 = (cTa + lambda)/(aTa)
            if(x1 > 0)
                x[idx] = x1
            elseif(x2 < 0)
                x[idx] = x2
            else
                error("bug!")
            end
        end
        
        if(iter % 100 == 0)
            obj = 0.5*norm( A*x - b )^2 + lambda*norm(x, 1)
            @printf "iter: %4.1i,  obj %4.3f\n" iter obj
        end
        
    end
    
    return x, counter
end

# to be optimized 
function lasso_random(A, b, lambda, maxiter = 1e4)
    n, m = size(A)
    x = zeros(m)
    counter = zeros(m)
    for iter = 1:maxiter
        Ax = A*x
        g = A'*(Ax - b)
        for j = 1:length(g)
            if( x[j] == 0 && g[j] >= -lambda && g[j] <= lambda )
                g[j] = 0
            else
                g[j] = abs( g[j] + sign(x[j])*lambda )
            end
        end
        # random coordinate
        idx = rand(1:m)
        counter[idx] += 1
        Atmp = A[:,idx]*x[idx]
        aTa = dot(A[:,idx], A[:,idx])
        if(aTa == 0)
            x[idx] = 0
            continue
        end
        c = b - (Ax - Atmp)
        cTa = dot(c, A[:,idx])
        if( cTa >= -lambda && cTa <= lambda )
            x[idx] = 0
        else
            x1 = (cTa - lambda)/(aTa)
            x2 = (cTa + lambda)/(aTa)
            if(x1 >= 0)
                x[idx] = x1
            elseif(x2 <= 0)
                x[idx] = x2
            else
                error("bug!")
            end
        end
        
        if(iter % 100 == 0)
            obj = 0.5*norm( A*x - b )^2 + lambda*norm(x, 1)
            @printf "iter: %4.1i,  obj %4.3f\n" iter obj
        end
        
    end
    
    return x, counter
end

