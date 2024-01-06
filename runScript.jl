include("FinchStructs.jl")
include("mesh_read.jl")
include("mesh_data.jl")
include("refel.jl")
include("grid.jl")
include("finch_constants.jl")
include("geometric_factors.jl")
include("serialMATVECVersions.jl")
include("globalAssemblyRoutines.jl")
include("gpuMATVECVersions.jl")
include("feUtils.jl")

using LinearAlgebra
import LinearAlgebra: mul!
using SparseArrays
using IterativeSolvers
using Test
using PyPlot
using LaTeXStrings
using CUDA
using Adapt
using StaticArrays
using BenchmarkTools

globalFloatType = Float32;

function initFunction(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat
    return 1.0; 
end

function exactSol(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat   
    return (sin(pi*x*1)*sin(pi*y*1)*sin(pi*z*1));
end

function rhsNodalFunction(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat   
    return (3 * (pi ^ 2) * sin( pi * x * 1 ) * sin( pi * y * 1 ) * sin( pi * z * 1 )); 
end


function testFullComputation( A, fValsRHS  )

    U = gmres( A, fValsRHS; reltol = 1e-14, verbose = true )

    return U;

end

function checkConvergence( gmshFolderName, configObj )

    meshvals = readdir( gmshFolderName, join = true )

    errorValsL2 = Array{globalFloatType}(undef, size( meshvals, 1 ));
    errorValsLinf = Array{globalFloatType}(undef, size( meshvals, 1 ));
    numNodeVals = Array{globalFloatType}(undef, size( meshvals, 1 ));
    hVals = Array{globalFloatType}(undef, size( meshvals, 1 ));

    for (index, meshval) in enumerate(meshvals)

        val = match( r"lvl", meshval )

        if !isnothing( val )
            lvlstr =  match( r"lvl", meshval )
            lvloffset = lvlstr.offset + 3
            lvlval = match( r"[0-9]+", meshval[ lvloffset:end ] )
            lvlval = lvlval.match
        end

        fileVal = open( meshval, "r" );
        println(meshval)

        gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
        nnodes = size( gridDataVal.allnodes, 2 );

        numNodeVals[index] = nnodes;
        A = SizedStrangMatrix((length( fValsRHS ),length( fValsRHS ) ) );
        U = testFullComputation( A, fValsRHS );
        
        ### Residual Error test
        @test isapprox( A* U, fValsRHS, atol = 1e-6 )

        ### Solution Error to check correct solution and convergence
        solutionErrorVec = U - exactvals;
        solutionErrorL2 = norm( solutionErrorVec, 2 );
        solutionErrorL2 = sqrt( (solutionErrorL2 ^ 2 ) / nnodes );
        solutionErrorLinf = norm( solutionErrorVec, Inf );

        errorValsL2[index] = solutionErrorL2;
        errorValsLinf[index] = solutionErrorLinf;

        hVals[index] = 1/(nnodes^(2/3));

    end

    figure(1)
    loglog( numNodeVals, errorValsL2[:], label = L"L^2" * " Error" )
    loglog( numNodeVals, hVals[:], label = L"h^2" )
    legend()

    figure(2)
    loglog( numNodeVals, errorValsLinf[:], label = L"L_{\infty}" * " Error" )
    loglog( numNodeVals, hVals[:], label = L"h^2" )
    legend()
    
end

function testMATVEC( fileVal, configObj )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    matVals, globalVec = createGlobalMatrix([0.2, 0.3], gridDataVal, refelVal, geoFacs, configObj ) 

    ### Global Matrix-Vector multiply versus Sparse MATVEC Product Test
    solutionValuesMATVEC = performMATVEC_version3( configObj.float_type, gridDataVal.allnodes, gridDataVal.loc2glb, fValsRHS,
    gridDataVal.nodebid, geoFacs.detJ, geoFacs.J, refelVal.Nqp, refelVal.Np, 
    refelVal.Qr, refelVal.Qs, refelVal.Qt, refelVal.wg );
    
    solutionValuesMatMul = matVals * fValsRHS;
    @test isapprox( solutionValuesMatMul, solutionValuesMATVEC, atol = 1e-6 )

end

function profileMATVEC( fileVal, configObj )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    ### Serial MATVEC performance
    # Finch version
    matvecBenchFinch = @benchmarkable solutionValuesMATVEC = performMATVEC( $configObj, $geoFacs, $refelVal, $gridDataVal, $fValsRHS )

    # version 1
    matvecBenchv1 = @benchmarkable solutionValuesMATVEC = performMATVEC_version1_Modified( $configObj.float_type, $gridDataVal.allnodes, $gridDataVal.loc2glb, $fValsRHS,
        $gridDataVal.nodebid, $geoFacs.detJ, $geoFacs.J, $refelVal.Nqp, $refelVal.Np, $refelVal.Qr, $refelVal.Qs, $refelVal.Qt, $refelVal.wg );

    # version 2 
    matvecBenchv2 = @benchmarkable solutionValuesMATVEC = performMATVEC_version2( $configObj.float_type, $gridDataVal.allnodes, $gridDataVal.loc2glb, $fValsRHS,
        $gridDataVal.nodebid, $geoFacs.detJ, $geoFacs.J, $refelVal.Nqp, $refelVal.Np, $refelVal.Qr, $refelVal.Qs, $refelVal.Qt, $refelVal.wg );

    # version 3
    matvecBenchv3 = @benchmarkable solutionValuesMATVEC = performMATVEC_version3( $configObj.float_type, $gridDataVal.allnodes, $gridDataVal.loc2glb, $fValsRHS,
        $gridDataVal.nodebid, $geoFacs.detJ, $geoFacs.J, $refelVal.Nqp, $refelVal.Np, $refelVal.Qr, $refelVal.Qs, $refelVal.Qt, $refelVal.wg );

    tune!(matvecBenchFinch);
    tune!(matvecBenchv1);
    tune!(matvecBenchv2);
    tune!(matvecBenchv3);

    mFinch = median( run(matvecBenchFinch) )
    m1 = median( run(matvecBenchv1) )
    m2 = median( run(matvecBenchv2) )
    m3 = median( run(matvecBenchv3) )
    # println(BenchmarkTools.DEFAULT_PARAMETERS)

    println( mFinch )
    println( m1 )
    println( m2 )
    println( m3 )

end

function testGPUMATVEC( fileVal, configObj )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    matVals, globalVec = createGlobalMatrix(Float32[0.2, 0.3], gridDataVal, refelVal, geoFacs, configObj ) 

    allnodes_d = CuArray( gridDataVal.allnodes )
    loc2glb_d = CuArray( gridDataVal.loc2glb )
    fValsRHS_d = CuArray( fValsRHS )
    nodebid_d = CuArray( gridDataVal.nodebid )
    detJ_d = CuArray( geoFacs.detJ )

    nnodes = size( gridDataVal.allnodes, 2 );
    nElements = size( gridDataVal.loc2glb, 2 );
    qnodes_per_element = refelVal.Nqp

    rx = zeros( qnodes_per_element, nElements );
    ry = zeros( qnodes_per_element, nElements );
    rz = zeros( qnodes_per_element, nElements );
    tx = zeros( qnodes_per_element, nElements );
    ty = zeros( qnodes_per_element, nElements );
    tz = zeros( qnodes_per_element, nElements );
    sx = zeros( qnodes_per_element, nElements );
    sy = zeros( qnodes_per_element, nElements );
    sz = zeros( qnodes_per_element, nElements );

    for eid = 1:nElements

        rx[ :, eid ] = geoFacs.J[eid].rx[:];
        ry[ :, eid ] = geoFacs.J[eid].ry[:];
        rz[ :, eid ] = geoFacs.J[eid].rz[:];
        sx[ :, eid ] = geoFacs.J[eid].sx[:];
        sy[ :, eid ] = geoFacs.J[eid].sy[:];
        sz[ :, eid ] = geoFacs.J[eid].sz[:];
        tx[ :, eid ] = geoFacs.J[eid].tx[:];
        ty[ :, eid ] = geoFacs.J[eid].ty[:];
        tz[ :, eid ] = geoFacs.J[eid].tz[:];

    end

    rx_d = CuArray( rx );
    ry_d = CuArray( ry );
    rz_d = CuArray( rz );
    tx_d = CuArray( tx );
    ty_d = CuArray( ty );
    tz_d = CuArray( tz );
    sx_d = CuArray( sx );
    sy_d = CuArray( sy );
    sz_d = CuArray( sz );

    Qr_d = CuArray( refelVal.Qr )
    Qs_d = CuArray( refelVal.Qs )
    Qt_d = CuArray( refelVal.Qt )
    wg_d = CuArray( refelVal.wg )

    solutionValuesMATVEC = CuArray( zeros( configObj.float_type, nnodes ) );

    numthreads = 256
    numblocks = ceil(Int, nElements / numthreads )

    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks performGPUMATVEC_version3( allnodes_d, loc2glb_d, fValsRHS_d, 
        nodebid_d, nnodes, nElements, detJ_d, rx_d, ry_d, rz_d, sx_d, sy_d, sz_d, tx_d, ty_d, tz_d, refelVal.Nqp, refelVal.Np,
        Qr_d, Qs_d, Qt_d, wg_d, solutionValuesMATVEC );
    end
    
    ### Global Matrix-Vector multiply versus Sparse MATVEC Product Test
    solutionValuesMatMul = matVals * fValsRHS;

    cpuSolutionValues = Array( solutionValuesMATVEC );

    filename = "saveValsMatMul.txt";
    fileval1 = open( filename, "w" );
    println( fileval1, solutionValuesMatMul )
    close(fileval1)

    filename = "saveValsMatVec.txt";
    fileval2 = open( filename, "w" );
    println( fileval2, cpuSolutionValues )
    close(fileval2)
    @test isapprox( solutionValuesMatMul, cpuSolutionValues, atol = 1e-6 )

end

struct SizedStrangMatrix
    size::Tuple{Int,Int}
end

Base.eltype(A::SizedStrangMatrix) = globalFloatType;
Base.size(A::SizedStrangMatrix) = A.size;
Base.size(A::SizedStrangMatrix,i::Int) = A.size[i];

gmshFolderName = "/home/gaurav/CS6958/Project/Code/Mesh3D/";

configObj = FinchConfig();

function mul!( C,A::SizedStrangMatrix,B )
    C[:] = performMATVEC_version3( configObj.float_type, gridDataVal.allnodes, gridDataVal.loc2glb, B,
     gridDataVal.nodebid, geoFacs.detJ, geoFacs.J, refelVal.Nqp, refelVal.Np, 
     refelVal.Qr, refelVal.Qs, refelVal.Qt, refelVal.wg );
end

LinearAlgebra.:*(A::SizedStrangMatrix,B::AbstractVector) = mul!(ones( length(B) ), A, B);        

# checkConvergence( gmshFolderName, configObj )
Adapt.@adapt_structure JacobianDevice

gmshFileName = gmshFolderName * "regularMesh3D_lvl2.msh";
fileVal = open( gmshFileName )
# testGPUMATVEC( fileVal, configObj )
profileMATVEC( fileVal, configObj )