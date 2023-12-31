#=
# Based on the file JacobiP.m from the book:
# Nodal Discontinuous Galerkin Method - Hesthaven, Warburton
#  https://link.springer.com/book/10.1007/978-0-387-72067-8
=#

# Evaluate Nth order Jacobi Polynomial of type (alpha,beta) at points x.
# Note   : They are normalized to be orthonormal.
function jacobi_polynomial(x, alpha::Int, beta::Int, N::Int; polynomialStartIdx::Int = -1)
    PL = zeros(N+1,length(x));
    
    # Initial values P_0(x) and P_1(x)
    gamma0 = 2^(alpha+beta+1)/(alpha+beta+1)*factorial(alpha)*
                factorial(beta)/factorial(alpha+beta);
    PL[1,:] = 1.0/sqrt(gamma0) .* ones(length(x));
    if N==0
        return PL[1,:];
    end
    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3) * gamma0;
    PL[2,:] = ((alpha+beta+2)/(2*sqrt(gamma1))) .* x .* ones(length(x)) .+ (alpha-beta)/(2*sqrt(gamma1));
    if N==1

        if polynomialStartIdx == -1
            return PL[2,:];
        else  
            return PL[polynomialStartIdx:end, :];
        end
    end
    
    # Higher orders are generated by this recurrence.
    # P(i+1) = -(a(i-1)/a(i))P(i-1) + ((x-b(i))/a(i))P(i)
    aold = 2/(2+alpha+beta)*sqrt((alpha+1)*(beta+1)/(alpha+beta+3));
    for i=1:N-1
      h1 = 2*i+alpha+beta;
      anew = 2/(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*
          (i+1+beta)/((h1+1)*(h1+3)));
      bnew = - (alpha^2-beta^2)/(h1*(h1+2));
      PL[i+2,:] = (-aold/anew) .* PL[i,:] + ((x.-bnew)./anew) .* PL[i+1,:];
      aold =anew;
    end;
    
    if polynomialStartIdx == -1
        return PL[N+1,:];
    else  
        return PL[polynomialStartIdx:end, :];
    end
end

# sqrt(N*(N+alpha+beta+1))*JacobiP(r(:),alpha+1,beta+1, N-1);
function grad_jacobi_polynomial(x, alpha::Int, beta::Int, N::Int)
    if N < 1
        return zeros(size(x));
    end
    
    return sqrt(N*(N+alpha+beta+1)) .* jacobi_polynomial(x, alpha+1, beta+1, N-1);
end