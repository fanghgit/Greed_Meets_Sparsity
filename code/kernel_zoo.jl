# linear kernel
function linear_kernel(X)
    return full( X*X' ) 
end


# polynominal kernel
function poly_kernel(X, gamma, c, d)
    n = size(X, 1)
    K = full( X*X' )
    K =  (gamma*K .+ c).^d
    return K
end


# gaussian kernel (rbf kernel)
function gaussian_kernel(X, sigma)
    n = size(X, 1)
    K = zeros(n, n)
    for i = 1:n
        for j = 1:n
            K[i,j] = exp( -norm( X[:,i] - X[:,j] )^2/(2*sigma^2) )
        end
    end
    return K
end


