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
using Adapt
using StaticArrays
using BenchmarkTools
# using BenchmarkPlots
# using StatsPlots

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

    U = gmres( A, fValsRHS; reltol = globalFloatType( 1e-14 ), verbose = true, restart = 1000 )

    return U;

end

function checkConvergence( gmshFolderName, configObj )

    meshvals = readdir( gmshFolderName, join = true )

    errorValsL2 = Array{configObj.float_type}(undef, size( meshvals, 1 ));
    errorValsLinf = Array{configObj.float_type}(undef, size( meshvals, 1 ));
    numNodeVals = Array{configObj.float_type}(undef, size( meshvals, 1 ));
    hVals = Array{configObj.float_type}(undef, size( meshvals, 1 ));

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
    figname = "Plots/SerialMATVECConvergenceL2.png";
    savefig( figname );

    figure(2)
    loglog( numNodeVals, errorValsLinf[:], label = L"L_{\infty}" * " Error" )
    loglog( numNodeVals, hVals[:], label = L"h^2" )
    legend()
    figname = "Plots/SerialMATVECConvergenceLinf.png";
    savefig( figname );
    
end

function testMATVEC( fileVal, configObj )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    matVals, globalVec = createGlobalMatrix(gridDataVal, refelVal, geoFacs, configObj ) 

    ### Global Matrix-Vector multiply versus Sparse MATVEC Product Test
    solutionValuesMATVEC = performMATVEC_version3( configObj.float_type, gridDataVal.allnodes, gridDataVal.loc2glb, fValsRHS,
    gridDataVal.nodebid, geoFacs.detJ, geoFacs.J, refelVal.Nqp, refelVal.Np, 
    refelVal.Qr, refelVal.Qs, refelVal.Qt, refelVal.wg );
    
    solutionValuesMatMul = matVals * fValsRHS;
    @test isapprox( solutionValuesMatMul, solutionValuesMATVEC, atol = 1e-6 )

end

function getSerialMATVECBenchmarkGroup( params, funcVal )

    return @benchmarkable solutionValuesMATVEC = $(funcVal)( $params... )
end

function profileMATVEC( fileVal, configObj, matvecBench )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    funcVals = [ performMATVEC, performMATVEC_version1_Modified, performMATVEC_version2, performMATVEC_version3 ];

    matvecVersionParams = [configObj.float_type, gridDataVal.allnodes, gridDataVal.loc2glb, fValsRHS,
        gridDataVal.nodebid, geoFacs.detJ, geoFacs.J, refelVal.Nqp, refelVal.Np, refelVal.Qr, refelVal.Qs,
        refelVal.Qt, refelVal.wg ]

    matvecFinchVersionParams = [configObj, geoFacs, refelVal, gridDataVal, fValsRHS]

    paramVals = [ matvecFinchVersionParams, matvecVersionParams, matvecVersionParams, matvecVersionParams ]

    tagVals = [ "VectorizedVersion_NoLoops", "QuadratureLoop_DerivativeVectorized", "QuadratureLoopWithDerivative_NoFusion",
    "QuadratureLoopWithDerivative_Fused" ]

    for (idx, funcVal) in enumerate( funcVals )
        # matvecBench[ tagVals[idx] ] = @benchmarkable solutionValuesMATVEC = $(funcVal)( $paramVals[$idx]... )
        matvecBench[ tagVals[idx] ] = getSerialMATVECBenchmarkGroup( paramVals[idx], funcVal ) 
    end

    tune!(matvecBench)
    results = run(matvecBench, verbose = true )

    return results, tagVals

end

function serialMATVECScaling( gmshFolderName, configObj )

    meshVals = readdir( gmshFolderName, join = true )

    matvecBench = BenchmarkGroup()

    tagVals = []
    lvlVals = zeros( length( meshVals ) )
    perfVals = zeros( length( meshVals ) )

    for (meshIdx, meshVal) in enumerate( meshVals )
        val = match( r"lvl", meshVal )

        if !isnothing( val )
            lvlStr =  match( r"lvl", meshVal )
            lvlOffset = lvlStr.offset + 3
            lvlVal = match( r"[0-9]+", meshVal[ lvlOffset:end ] )
            lvlVal = parse( Int64, lvlVal.match) + 1
        end

        fileVal = open( meshVal )
        lvlVals[ lvlVal ] = lvlVal

        ( matvecBench[lvlVal], tagVals ) = profileMATVEC( fileVal, configObj, BenchmarkGroup() )
    
    end

    figure(1)

    for tagVal in tagVals
        matvecBenchGroups = matvecBench[@tagged (tagVal)]

        for (lvlVal, benchVal) in matvecBenchGroups

            medianVal = median(benchVal[ tagVal ]);
            medianTimeVal = medianVal.time;
            perfVals[ lvlVal ] = medianTimeVal * (10 ^ (-6));

        end

        plot( lvlVals, perfVals, label = tagVal, "-o" )
    end

    legend()
    xlabel("Levels")
    ylabel("Time (ms)")
    figname = "Plots/SerialMATVECScaling.png";
    savefig( figname );

end

function profileBestSerialMATVEC( fileVal, configObj, matvecBench, serialFuncVal = performMATVEC, serialTagVal = "BestSerialVersion" )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    matvecFinchVersionParams = [configObj, geoFacs, refelVal, gridDataVal, fValsRHS]

    matvecBench[ serialTagVal ] = getSerialMATVECBenchmarkGroup( matvecFinchVersionParams, serialFuncVal ) 

    tune!(matvecBench)
    results = run(matvecBench, verbose = true )

    return results

end

function benchGPU( funcVal, params, numthreads, numblocks )

    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks funcVal( params... )
    end

end

function profileGPUMATVEC( fileVal, configObj, matvecBench )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    funcVals = [ performGPUMATVEC_version1, performGPUMATVEC_version2 ];
    tagVals = [ "gpuVersion1", "gpuVersion2" ]

    paramVals = [getGPUVersionParams( refelVal, geoFacs, gridDataVal, fValsRHS, configObj.float_type, 1 ), 
    getGPUVersionParams( refelVal, geoFacs, gridDataVal, fValsRHS, configObj.float_type, 2 ) ]

    nElements = size( gridDataVal.loc2glb, 2 );

    numthreads = 256
    numblocks = ceil(Int, nElements / numthreads )

    for (idx, funcVal) in enumerate( funcVals )
        matvecBench[ tagVals[idx] ] = @benchmarkable $benchGPU( $funcVal, $paramVals[$idx], $numthreads, $numblocks )
    end

    # CUDA.@profile benchGPU( funcVals[1], paramVals[1], numthreads, numblocks );

    tune!(matvecBench)
    results = run(matvecBench, verbose = true )

    return results, tagVals

end

function gpuMATVECGridScaling( gmshFolderName, configObj, figname )

    meshVals = readdir( gmshFolderName, join = true )

    gpuMatvecBench = BenchmarkGroup()
    bestSerialMATVECBench = BenchmarkGroup()

    tagVals = []
    lvlVals = zeros( length( meshVals ) )
    perfVals = zeros( length( meshVals ) )

    for (meshIdx, meshVal) in enumerate( meshVals )
        val = match( r"lvl", meshVal )

        if !isnothing( val )
            lvlStr =  match( r"lvl", meshVal )
            lvlOffset = lvlStr.offset + 3
            lvlVal = match( r"[0-9]+", meshVal[ lvlOffset:end ] )
            lvlVal = parse( Int64, lvlVal.match) + 1
        end

        lvlVals[ lvlVal ] = lvlVal

        fileVal = open( meshVal )
        ( gpuMatvecBench[lvlVal], tagVals ) = profileGPUMATVEC( fileVal, configObj, BenchmarkGroup() )
        close(fileVal)

        fileVal = open( meshVal )
        bestSerialMATVECBench[lvlVal] = profileBestSerialMATVEC( fileVal, configObj, BenchmarkGroup() )
        close(fileVal)
    end

    serialTagVal = "BestSerialVersion";
    figure(1)

    for tagVal in tagVals
        gpuMatvecBenchGroups = gpuMatvecBench[@tagged (tagVal)]

        for (lvlVal, benchVal) in gpuMatvecBenchGroups

            gpuMedianVal = median(benchVal[ tagVal ]);
            gpuMedianTimeVal = gpuMedianVal.time;

            serialMedianVal = median( bestSerialMATVECBench[lvlVal][serialTagVal] )
            serialMedianTimeVal = serialMedianVal.time;

            speedupVal = serialMedianTimeVal / gpuMedianTimeVal;
            perfVals[ lvlVal ] = speedupVal;

        end

        plot( lvlVals, perfVals, label = tagVal, "-o" )
    end

    legend()
    xlabel("Levels")
    ylabel("Speedup")
    savefig( figname );

end

function getIndexTuple( dxn, qnodes_per_element, eid )

    if dxn == 1
        return ( 1:qnodes_per_element, eid )
    elseif dxn == 2
        return ( eid, 1:qnodes_per_element )
    else
        println("Invalid dxn")
        return
    end

end

function getInitializerTuple( dxn, qnodes_per_element, nElements )

    if dxn == 1
        return ( qnodes_per_element, nElements )
    elseif dxn == 2
        return ( nElements, qnodes_per_element )
    else
        println("Invalid dxn")
        return
    end

end

function getGPUVersionParams( refelVal, geoFacs, gridDataVal, fValsRHS, float_type, dxn )

    nnodes = size( gridDataVal.allnodes, 2 );
    nElements = size( gridDataVal.loc2glb, 2 );
    qnodes_per_element = refelVal.Nqp

    initTuple = getInitializerTuple( dxn, qnodes_per_element, nElements )

    rx = zeros( initTuple... );
    ry = zeros( initTuple... );
    rz = zeros( initTuple... );
    tx = zeros( initTuple... );
    ty = zeros( initTuple... );
    tz = zeros( initTuple... );
    sx = zeros( initTuple... );
    sy = zeros( initTuple... );
    sz = zeros( initTuple... );
    
    for eid = 1:nElements

        idxTuple = getIndexTuple( dxn, qnodes_per_element, eid )

        rx[ idxTuple... ] = geoFacs.J[eid].rx[:];
        ry[ idxTuple... ] = geoFacs.J[eid].ry[:];
        rz[ idxTuple... ] = geoFacs.J[eid].rz[:];
        sx[ idxTuple... ] = geoFacs.J[eid].sx[:];
        sy[ idxTuple... ] = geoFacs.J[eid].sy[:];
        sz[ idxTuple... ] = geoFacs.J[eid].sz[:];
        tx[ idxTuple... ] = geoFacs.J[eid].tx[:];
        ty[ idxTuple... ] = geoFacs.J[eid].ty[:];
        tz[ idxTuple... ] = geoFacs.J[eid].tz[:];

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

    allnodes_d = CuArray( gridDataVal.allnodes )
    loc2glb_d = CuArray( gridDataVal.loc2glb )
    fValsRHS_d = CuArray( fValsRHS )
    nodebid_d = CuArray( gridDataVal.nodebid )
    detJ_d = CuArray( geoFacs.detJ )

    solutionValuesMATVEC = CuArray( zeros( configObj.float_type, nnodes ) );

    return (allnodes_d, loc2glb_d, fValsRHS_d, nodebid_d, nnodes, nElements, detJ_d, rx_d, ry_d, rz_d, 
    sx_d, sy_d, sz_d, tx_d, ty_d, tz_d, refelVal.Nqp, refelVal.Np, Qr_d, Qs_d, Qt_d, wg_d, solutionValuesMATVEC)
end

function testGPUMATVEC( fileVal, configObj )

    gridDataVal, refelVal, geoFacs, exactvals, fValsRHS = getParameters( configObj, fileVal );
    
    matVals, globalVec = createGlobalMatrix( gridDataVal, refelVal, geoFacs, configObj ) 

    dxn = 2
    
    gpuVersionParams = getGPUVersionParams( refelVal, geoFacs, gridDataVal, fValsRHS, configObj.float_type, dxn )

    nElements = size( gridDataVal.loc2glb, 2 );

    numthreads = 256
    numblocks = ceil(Int, nElements / numthreads )

    CUDA.@sync begin
        @cuda threads = numthreads blocks = numblocks performGPUMATVEC_version2( gpuVersionParams... );
    end
    
    ### Global Matrix-Vector multiply versus Sparse MATVEC Product Test
    solutionValuesMatMul = matVals * fValsRHS;

    solutionValuesMATVEC = gpuVersionParams[ end ]
    cpuSolutionValues = Array( solutionValuesMATVEC );

    filename = "saveValsMatMulTet.txt";
    fileval1 = open( filename, "w" );
    println( fileval1, solutionValuesMatMul )
    close(fileval1)

    filename = "saveValsMatVecTet.txt";
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

gmshFolderName = "/home/gaurav/CS6958/Project/Code/Mesh3D/TetMesh3D/";

configObj = FinchConfig();

function mul!( C,A::SizedStrangMatrix,B )
    C[:] = performMATVEC_version3( configObj.float_type, gridDataVal.allnodes, gridDataVal.loc2glb, B,
     gridDataVal.nodebid, geoFacs.detJ, geoFacs.J, refelVal.Nqp, refelVal.Np, 
     refelVal.Qr, refelVal.Qs, refelVal.Qt, refelVal.wg );
end

LinearAlgebra.:*(A::SizedStrangMatrix,B::AbstractVector) = mul!(ones( length(B) ), A, B);        

# checkConvergence( gmshFolderName, configObj )
# Adapt.@adapt_structure JacobianDevice

# gmshFileName = gmshFolderName * "tetMesh3D_lvl0.msh";
# fileVal = open( gmshFileName )
# testMATVEC( fileVal, configObj )

# testGPUMATVEC( fileVal, configObj )
# profileMATVEC( fileVal, configObj )
# matvecBench = BenchmarkGroup()
# profileGPUMATVEC( fileVal, configObj, matvecBench )

figname = "Plots/GPUMATVECGridScaling_TetMesh3D.png";
gpuMATVECGridScaling( gmshFolderName, configObj, figname )
# serialMATVECScaling( gmshFolderName, configObj )