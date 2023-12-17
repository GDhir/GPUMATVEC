import Base
using LinearAlgebra
import LinearAlgebra: mul!

struct SizedStrangMatrix
    size::Tuple{Int,Int}
end

Base.eltype(A::SizedStrangMatrix) = Float64
Base.size(A::SizedStrangMatrix) = A.size
Base.size(A::SizedStrangMatrix,i::Int) = A.size[i]

B = ones( 20 )

A = SizedStrangMatrix((length(B),length(B)))

function mul!(C,A::SizedStrangMatrix,B, val1, val2)
    for i in 2:length(B)-1
        C[i] = B[i-1] - 2B[i] + B[i+1]
    end
    C[1] = -2B[1] + B[2]
    C[end] = B[end-1] - 2B[end]
    C
end
LinearAlgebra.:*(A::SizedStrangMatrix,B::AbstractVector) = (C = similar(B); mul!(C,A,B, 1, 1))

using IterativeSolvers
U = gmres(A,B,abstol=1e-14)
norm(A*U - B)