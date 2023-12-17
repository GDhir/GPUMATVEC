include("FinchStructs.jl")
include("mesh_read.jl")
include("mesh_data.jl")
include("refel.jl")
include("grid.jl")
include("finch_constants.jl")
include("geometric_factors.jl")

using LinearAlgebra
import LinearAlgebra: mul!
using SparseArrays
using IterativeSolvers

function initFunction(x::Union{Float64,Float64},y::Union{Float64,Float64},z::Union{Float64,Float64})    
    return 1.0; 
end

function exactSol(x::Union{Float64,Float64},y::Union{Float64,Float64},z::Union{Float64,Float64})    
    return (sin(pi*x*1)*sin(pi*y*1)*sin(pi*z*1));
end

function rhsNodalFunction(x::Union{Float64,Float64},y::Union{Float64,Float64},z::Union{Float64,Float64})    
    return (3 * (pi ^ 2) * sin( pi * x * 1 ) * sin( pi * y * 1 ) * sin( pi * z * 1 )); 
end

function apply_boundary_conditions_elemental(var::Vector{FT}, eid::Int, grid::Grid, refel::Refel,
    geo_facs::GeometricFactors, elmat::Matrix{FT}, elvec::Vector{FT} ) where FT<:AbstractFloat

    # Check each node to see if the bid is > 0 (on boundary)
    nnodes = refel.Np;
    for ni=1:nnodes
        node_id = grid.loc2glb[ni,eid];
        node_bid = grid.nodebid[node_id];
        if node_bid > 0
            # This is a boundary node in node_bid
            # Handle the BC for each variable
            row_index = ni;
            col_offset = 0;

            # zero the row
            for nj=1:size(elmat,2)
                elmat[row_index, nj] = 0;
            end
            elvec[row_index] = 0;
            elmat[row_index, row_index] = 1;
        end
    end
end

###########Global Matrix Creation for testing purposes to check MATVEC is correct
function createGlobalMatrix(var::Vector{Float64}, mesh::Grid, refel::Refel, geometric_factors::GeometricFactors, config::FinchConfig ) 
    
    # User specified data types for int and float
    # int type is Int64
    # float type is Float64
    
    function genfunction_2(x::Union{Float64,Float64},y::Union{Float64,Float64},z::Union{Float64,Float64}) 
        return 3*pi*pi*sin(pi*x*1)*sin(pi*y*1)*sin(pi*z*1); 
    end
    
    # Prepare some useful numbers
    dofs_per_node = 1;
    dofs_per_loop = 1;
    dof_offsets = [0];
    nnodes_partition = size(mesh.allnodes,2);
    nnodes_global = nnodes_partition;
    dofs_global = nnodes_global;
    num_elements = size(mesh.loc2glb, 2);
    num_elements_global = num_elements;
    
    nodes_per_element = refel.Np;
    qnodes_per_element = refel.Nqp;
    faces_per_element = refel.Nfaces;
    nodes_per_face = refel.Nfp[1];
    local_system_size = dofs_per_loop * nodes_per_element;
    dofs_per_element = nodes_per_element;
    
    # FEM specific pieces
    Q = refel.Q;
    wg = refel.wg;
    surf_wg = refel.surf_wg[1];
    
    # For partitioned meshes
    #= Allocate global matrix(IJV form) and vector. =#
    allocated_nonzeros = (num_elements * dofs_per_element * dofs_per_element)
    next_nonzero_index = (allocated_nonzeros + 1)
    global_matrix_I::Vector{Int64} = zeros(Int64, allocated_nonzeros)
    global_matrix_J::Vector{Int64} = zeros(Int64, allocated_nonzeros)
    global_matrix_V::Vector{Float64} = zeros(Float64, allocated_nonzeros)
    global_vector::Vector{Float64} = zeros(Float64, dofs_global)
    global_solution::Vector{Float64} = zeros(Float64, dofs_global)
    
    solution = global_solution

    #= I and J vectors should init as ones =#
    global_matrix_I .= 1
    global_matrix_J .= 1
    #= Allocate elemental matrix and vector. =#
    element_matrix::Matrix{Float64} = zeros(Float64, local_system_size, local_system_size)
    element_vector::Vector{Float64} = zeros(Float64, local_system_size)
    #= Boundary done flag for each node. =#
    bdry_done::Vector{Int64} = zeros(Int64, nnodes_global)
    #= No indexed variables =#
    index_values::Vector{Int64} = zeros(Int64, 0)
    #= Allocate coefficient vectors. =#
    detj::Vector{Float64} = zeros(Float64, qnodes_per_element)
    #= Allocate for derivative matrices. =#
    RQ1::Matrix{Float64} = zeros(Float64, qnodes_per_element, nodes_per_element)
    RQ2::Matrix{Float64} = zeros(Float64, qnodes_per_element, nodes_per_element)
    RQ3::Matrix{Float64} = zeros(Float64, qnodes_per_element, nodes_per_element)
    NODALvalue__f_1::Vector{Float64} = zeros(Float64, nodes_per_element)
    value__f_1::Vector{Float64} = zeros(Float64, qnodes_per_element)
    t = 0;

    for ei = 1:num_elements

        eid = mesh.elemental_order[ei]
        index_offset = 0
        build_derivative_matrix(refel, geometric_factors, 1, eid, 0, RQ1)
        build_derivative_matrix(refel, geometric_factors, 2, eid, 0, RQ2)
        build_derivative_matrix(refel, geometric_factors, 3, eid, 0, RQ3)
        #= Prepare derivative matrices. =#
        #= Evaluate coefficients. =#
        for ni = 1:nodes_per_element
            nodeID = mesh.loc2glb[ni, eid]
            x = mesh.allnodes[1, nodeID]
            y = mesh.allnodes[2, nodeID]
            z = mesh.allnodes[3, nodeID]
            NODALvalue__f_1[ni]::Float64 = Float64(genfunction_2( x, y, z ))
        end

        for col = 1:qnodes_per_element
            value__f_1[col] = 0.0
            for row = 1:nodes_per_element
                value__f_1[col] = (value__f_1[col] + (Q[col, row] * NODALvalue__f_1[row]))
            end

        end

        for qnode_i = 1:qnodes_per_element
            detj[qnode_i] = geometric_factors.detJ[qnode_i, eid]
        end

        
        for col=1:nodes_per_element
            for row=1:nodes_per_element
                element_matrix[row, col] = 0;
                
                for i=1:qnodes_per_element
                    element_matrix[row, col] += ((RQ1[i, row] * (wg[i] * detj[i]) * RQ1[i, col]) + (RQ2[i, row] * (wg[i] * detj[i]) * RQ2[i, col]) + (RQ3[i, row] * (wg[i] * detj[i]) * RQ3[i, col]));
                    
                end
            end
        end


        for row=1:nodes_per_element
            element_vector[row] = 0;
            
            for col=1:qnodes_per_element
                element_vector[row] += (Q[col, row] * (wg[col] * detj[col] * value__f_1[col] ));
                
            end
        end


        #= No face loop needed. =#
        #= Apply boundary conditions. =#
        apply_boundary_conditions_elemental(var, eid, mesh, refel, geometric_factors, element_matrix, element_vector )
        #= Place elemental parts in global system. =#
        next_ind = (1 + ((eid - 1) * nodes_per_element * nodes_per_element))
        for ni = 1:nodes_per_element
            glb_i = mesh.loc2glb[ni, eid]
            global_vector[glb_i] = (global_vector[glb_i] + element_vector[ni])
            for nj = 1:nodes_per_element
                glb_j = mesh.loc2glb[nj, eid]
                global_matrix_I[next_ind] = glb_i
                global_matrix_J[next_ind] = glb_j
                global_matrix_V[next_ind] = element_matrix[ni, nj]
                next_ind = (next_ind + 1)
            end
        end            
    end

        
    global_matrix = sparse(@view(global_matrix_I[1:(next_nonzero_index-1)]), @view(global_matrix_J[1:(next_nonzero_index-1)]), @view(global_matrix_V[1:(next_nonzero_index-1)]));
        
    return global_matrix, global_vector;
    
end


function performMATVEC( configObj, geoFacs, refelVal, gridDataVal, uvals )

    nnodes = size( gridDataVal.allnodes, 2 );
    nElements = size( gridDataVal.loc2glb, 2 );
    qnodes_per_element = refelVal.Nqp;
    nodes_per_element = refelVal.Np;

    solutionVals = zeros( configObj.float_type, nnodes, 1 );

    detj::Vector{Float64} = zeros(configObj.float_type, qnodes_per_element);
    #= Allocate for derivative matrices. =#
    RQ1::Matrix{Float64} = zeros(configObj.float_type, qnodes_per_element, nodes_per_element);
    RQ2::Matrix{Float64} = zeros(configObj.float_type, qnodes_per_element, nodes_per_element);
    RQ3::Matrix{Float64} = zeros(configObj.float_type, qnodes_per_element, nodes_per_element);

    ugrad::Matrix{Float64} = zeros(configObj.float_type, qnodes_per_element, 3);

    wg = refelVal.wg;
    uvalsLocal = zeros(configObj.float_type, nodes_per_element, 1);

    element_matrix::Matrix{Float64} = zeros(Float64, nodes_per_element, nodes_per_element);

    for eid = 1:nElements

        build_derivative_matrix(refelVal, geoFacs, 1, eid, 0, RQ1);
        build_derivative_matrix(refelVal, geoFacs, 2, eid, 0, RQ2);
        build_derivative_matrix(refelVal, geoFacs, 3, eid, 0, RQ3);

        for qnode_i = 1:qnodes_per_element
            detj[qnode_i] = geoFacs.detJ[qnode_i, eid];
        end

        uvalsLocal[:, 1] = uvals[ gridDataVal.loc2glb[ :, eid ] ];

        ugrad[:, 1] = RQ1 * uvalsLocal;
        ugrad[:, 2] = RQ2 * uvalsLocal;
        ugrad[:, 3] = RQ3 * uvalsLocal;

        ugrad[:, :] = ugrad .* detj;
        ugrad[:, :] = ugrad .* wg;

        newUvals = zeros(configObj.float_type, nodes_per_element, 1);

        newUvals = sum( RQ1 .* ugrad[ :, 1 ], dims = 1) .+ sum( RQ2 .* ugrad[ :, 2 ], dims = 1) .+
            sum( RQ3 .* ugrad[ :, 3 ], dims = 1 );

        for localNodeId = 1:nodes_per_element

            globalNodeId = gridDataVal.loc2glb[ localNodeId, eid ];

            if( gridDataVal.nodebid[ globalNodeId ] == 0 )
                solutionVals[ globalNodeId ] = solutionVals[ globalNodeId ] .+ newUvals[ localNodeId ];
            end
        end
    end

    return solutionVals;
end

function getRHS( configObj, geoFacs, refelVal, gridDataVal, fvalsNodal, fvalsRHS  )

    nnodes = size( gridDataVal.allnodes, 2 );
    nElements = size( gridDataVal.loc2glb, 2 );
    qnodes_per_element = refelVal.Nqp;
    nodes_per_element = refelVal.Np;

    detj::Vector{Float64} = zeros(configObj.float_type, qnodes_per_element);  
    
    wg = refelVal.wg;
    fvalsLocal = zeros(configObj.float_type, nodes_per_element, 1);
    fvalsQuadPoints = zeros(configObj.float_type, qnodes_per_element, 1);

    for eid = 1:nElements

        for qnode_i = 1:qnodes_per_element
            detj[qnode_i] = geoFacs.detJ[qnode_i, eid];
        end

        fvalsLocal[:, 1] = fvalsNodal[ gridDataVal.loc2glb[ :, eid ] ];

        fvalsQuadPoints[:, 1] = refelVal.Q * fvalsLocal;

        for localNodeId = 1:nodes_per_element

            globalNodeId = gridDataVal.loc2glb[ localNodeId, eid ];

            if( gridDataVal.nodebid[ globalNodeId ] == 0 )

                fvalsRHS[ globalNodeId ] += sum( refelVal.Q[ :, localNodeId ] .* detj .* wg .* fvalsQuadPoints[:, 1], dims = 1 )[1];

            end
        end
    end
end

gmshfilename = "/home/gaurav/CS6958/Project/Code/Mesh3D/regularMesh3D_lvl3.msh";

fileVal = open( gmshfilename, "r" );

meshDataVal = read_mesh( fileVal );

configObj = FinchConfig();

refelVal, gridDataVal = grid_from_mesh( meshDataVal, configObj );

nnodes = size( gridDataVal.allnodes, 2 );

uvals = zeros( configObj.float_type, nnodes, 1 );
rhsNodeVals = zeros( configObj.float_type, nnodes, 1 );
exactvals = zeros( configObj.float_type, nnodes, 1 );

for nodeId = 1:nnodes

    nodeCoords = gridDataVal.allnodes[ :, nodeId ];
    uvals[ nodeId, 1 ] = initFunction( nodeCoords[1], nodeCoords[2], nodeCoords[3] );
    rhsNodeVals[ nodeId, 1 ] = rhsNodalFunction( nodeCoords[1], nodeCoords[2], nodeCoords[3] );
    exactvals[ nodeId, 1 ] = exactSol( nodeCoords[1], nodeCoords[2], nodeCoords[3] );
end

geoFacs = build_geometric_factors( configObj, refelVal, gridDataVal, do_face_detj = false, 
    do_vol_area = false, constant_jacobian = false );

fValsRHS = zeros( configObj.float_type, nnodes, 1 );
getRHS( configObj, geoFacs, refelVal, gridDataVal, rhsNodeVals, fValsRHS  );

struct SizedStrangMatrix
    size::Tuple{Int,Int}
end

Base.eltype(A::SizedStrangMatrix) = Float64;
Base.size(A::SizedStrangMatrix) = A.size;
Base.size(A::SizedStrangMatrix,i::Int) = A.size[i];

A = SizedStrangMatrix((length( fValsRHS ),length( fValsRHS ) ) );

matVals, globalVec = createGlobalMatrix([0.2, 0.3], gridDataVal, refelVal, geoFacs, configObj ) 

function mul!( C,A::SizedStrangMatrix,B )
    C[:] = performMATVEC( configObj, geoFacs, refelVal, gridDataVal, B );
end

LinearAlgebra.:*(A::SizedStrangMatrix,B::AbstractVector) = mul!(ones( length(B) ), A, B);

U = gmres( matVals, fValsRHS; reltol = 1e-14, verbose = true )

using Test

### Residual Error test
@test isapprox( A* U, fValsRHS, atol = 1e-6 )

### Global Matrix-Vector multiply versus Sparse MATVEC Product Test
solutionValuesMATVEC = performMATVEC( configObj, geoFacs, refelVal, gridDataVal, fValsRHS );
solutionValuesMatMul = matVals * fValsRHS;
@test isapprox( solutionValuesMatMul, solutionValuesMATVEC, atol = 1e-6 )

### Solution Error to check correct solution and convergence
solutionErrorVec = U - exactvals;
solutionErrorL2 = norm( solutionErrorVec / nnodes, 2 );
solutionErrorLInf = norm( solutionErrorVec, Inf );
