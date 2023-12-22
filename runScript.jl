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
using Test
using PyPlot
using LaTeXStrings

globalFloatType = Float64;

function initFunction(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat
    return 1.0; 
end

function exactSol(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat   
    return (sin(pi*x*1)*sin(pi*y*1)*sin(pi*z*1));
end

function rhsNodalFunction(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat   
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
function createGlobalMatrix(var::Vector{FT}, mesh::Grid, refel::Refel, geometric_factors::GeometricFactors, config::FinchConfig ) where FT<:AbstractFloat
    
    # User specified data types for int and float
    # int type is Int64
    # float type is FT
    
    function genfunction_2(x::Union{FT,FT},y::Union{FT,FT},z::Union{FT,FT}) where FT<:AbstractFloat
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
    global_matrix_V::Vector{globalFloatType} = zeros(globalFloatType, allocated_nonzeros)
    global_vector::Vector{globalFloatType} = zeros(globalFloatType, dofs_global)
    global_solution::Vector{globalFloatType} = zeros(globalFloatType, dofs_global)
    
    solution = global_solution

    #= I and J vectors should init as ones =#
    global_matrix_I .= 1
    global_matrix_J .= 1
    #= Allocate elemental matrix and vector. =#
    element_matrix::Matrix{globalFloatType} = zeros(globalFloatType, local_system_size, local_system_size)
    element_vector::Vector{globalFloatType} = zeros(globalFloatType, local_system_size)
    #= Boundary done flag for each node. =#
    bdry_done::Vector{Int64} = zeros(Int64, nnodes_global)
    #= No indexed variables =#
    index_values::Vector{Int64} = zeros(Int64, 0)
    #= Allocate coefficient vectors. =#
    detj::Vector{globalFloatType} = zeros(globalFloatType, qnodes_per_element)
    #= Allocate for derivative matrices. =#
    RQ1::Matrix{globalFloatType} = zeros(globalFloatType, qnodes_per_element, nodes_per_element)
    RQ2::Matrix{globalFloatType} = zeros(globalFloatType, qnodes_per_element, nodes_per_element)
    RQ3::Matrix{globalFloatType} = zeros(globalFloatType, qnodes_per_element, nodes_per_element)
    NODALvalue__f_1::Vector{globalFloatType} = zeros(globalFloatType, nodes_per_element)
    value__f_1::Vector{globalFloatType} = zeros(globalFloatType, qnodes_per_element)
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
            NODALvalue__f_1[ni]::globalFloatType = globalFloatType(genfunction_2( x, y, z ))
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

    detj::Vector{globalFloatType} = zeros(configObj.float_type, qnodes_per_element);
    #= Allocate for derivative matrices. =#
    RQ1::Matrix{globalFloatType} = zeros(configObj.float_type, qnodes_per_element, nodes_per_element);
    RQ2::Matrix{globalFloatType} = zeros(configObj.float_type, qnodes_per_element, nodes_per_element);
    RQ3::Matrix{globalFloatType} = zeros(configObj.float_type, qnodes_per_element, nodes_per_element);

    ugrad::Matrix{globalFloatType} = zeros(configObj.float_type, qnodes_per_element, 3);

    wg = refelVal.wg;
    uvalsLocal = zeros(configObj.float_type, nodes_per_element, 1);

    element_matrix::Matrix{globalFloatType} = zeros(globalFloatType, nodes_per_element, nodes_per_element);

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

## Serial MATVEC adapted for GPU implementation (GPU serial version 1)
# Takes more per element memory
function performMATVEC_version1( float_type, allnodes, loc2glb, uvals, nodebid, detJ, Jval, qnodes_per_element, 
    nodes_per_element, Qr, Qs, Qt, wg )

    nnodes = size( allnodes, 2 );
    nElements = size( loc2glb, 2 );

    uvalsLocal::Matrix{float_type} = zeros(float_type, nodes_per_element, 1);
    solutionVals = zeros( float_type, nnodes, 1 );
    
    detj::Vector{float_type} = zeros(float_type, qnodes_per_element);
    ugrad::Matrix{float_type} = zeros(float_type, qnodes_per_element, 1);

    for eid = 1:nElements

        newUvals = zeros(float_type, nodes_per_element, 1);

        for qnode_i = 1:qnodes_per_element
            detj[qnode_i] = detJ[qnode_i, eid];
        end

        J = Jval[eid];

        ## X Direction MATVEC Product
        Jr = J.rx;
        Js = J.sx;
        Jt = J.tx;

        RQ = Qr .* Jr + Qs .* Js + Qt .* Jt;

        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        ugrad = RQ * uvalsLocal;
        
        for pval = 1:nodes_per_element
            for qp = 1:qnodes_per_element
                newUvals[ pval, 1 ] = newUvals[ pval, 1 ] + RQ[ qp, pval ] * ugrad[ qp ] * detj[ qp ] * wg[ qp ];
            end
        end

        ## Y Direction MATVEC Product
        Jr = J.ry;
        Js = J.sy;
        Jt = J.ty;

        RQ = Qr .* Jr + Qs .* Js + Qt .* Jt;

        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        ugrad = RQ * uvalsLocal;

        for pval = 1:nodes_per_element
            for qp = 1:qnodes_per_element
                newUvals[ pval, 1 ] = newUvals[ pval, 1 ] + RQ[ qp, pval ] * ugrad[ qp ] * detj[ qp ] * wg[ qp ];
            end
        end

        ## Z Direction MATVEC Product
        Jr = J.rz;
        Js = J.sz;
        Jt = J.tz;

        RQ = Qr .* Jr + Qs .* Js + Qt .* Jt;

        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        ugrad = RQ * uvalsLocal;

        for pval = 1:nodes_per_element
            for qp = 1:qnodes_per_element
                newUvals[ pval, 1 ] = newUvals[ pval, 1 ] + RQ[ qp, pval ] * ugrad[ qp ] * detj[ qp ] * wg[ qp ];
            end
        end

        for localNodeId = 1:nodes_per_element

            globalNodeId = loc2glb[ localNodeId, eid ];

            if( nodebid[ globalNodeId ] == 0 )
                solutionVals[ globalNodeId ] = solutionVals[ globalNodeId ] .+ newUvals[ localNodeId ];
            end
        end
    end

    return solutionVals;
end


## Serial MATVEC adapted for GPU implementation (GPU serial version 2)
# Takes less per element memory 
function performMATVEC_version2( float_type, allnodes, loc2glb, uvals, nodebid, detJ, J, qnodes_per_element, 
    nodes_per_element, Qr, Qs, Qt, wg )

    nnodes = size( allnodes, 2 );
    nElements = size( loc2glb, 2 );

    uvalsLocal = zeros(float_type, nodes_per_element, 1);
    solutionVals = zeros( float_type, nnodes, 1 );

    for eid = 1:nElements

        newUvals = zeros(float_type, nodes_per_element, 1);

        ## X Direction MATVEC Product
        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        for qp = 1:qnodes_per_element
            for i = 1:nodes_per_element

                phigrad_qi = Qr[ qp, i ] * J[eid].rx[ qp ] + Qs[ qp, i ] * J[eid].sx[ qp ] + Qt[ qp, i ] * J[eid].tx[ qp ]; 
                ux_qp = 0;
                for j = 1:nodes_per_element

                    phigrad_qj = Qr[ qp, j ] * J[eid].rx[ qp ] + Qs[ qp, j ] * J[eid].sx[ qp ] + Qt[ qp, j ] * J[eid].tx[ qp ]; 
                    ux_qp = ux_qp + uvalsLocal[ j ] * phigrad_qj * detJ[ qp, eid ] * wg[ qp ];
                end

                newUvals[i] = newUvals[i] + phigrad_qi * ux_qp;
            end
        end

        ## Y Direction MATVEC Product
        for qp = 1:qnodes_per_element
            for i = 1:nodes_per_element

                phigrad_qi = Qr[ qp, i ] * J[eid].ry[ qp ] + Qs[ qp, i ] * J[eid].sy[ qp ] + Qt[ qp, i ] * J[eid].ty[ qp ]; 
                ux_qp = 0;
                for j = 1:nodes_per_element

                    phigrad_qj = Qr[ qp, j ] * J[eid].ry[ qp ] + Qs[ qp, j ] * J[eid].sy[ qp ] + Qt[ qp, j ] * J[eid].ty[ qp ]; 
                    ux_qp = ux_qp + uvalsLocal[ j ] * phigrad_qj * detJ[ qp, eid ] * wg[ qp ];
                end

                newUvals[i] = newUvals[i] + phigrad_qi * ux_qp;
            end
        end

        ## Z Direction MATVEC Product
        for qp = 1:qnodes_per_element
            for i = 1:nodes_per_element

                phigrad_qi = Qr[ qp, i ] * J[eid].rz[ qp ] + Qs[ qp, i ] * J[eid].sz[ qp ] + Qt[ qp, i ] * J[eid].tz[ qp ]; 
                ux_qp = 0;
                for j = 1:nodes_per_element

                    phigrad_qj = Qr[ qp, j ] * J[eid].rz[ qp ] + Qs[ qp, j ] * J[eid].sz[ qp ] + Qt[ qp, j ] * J[eid].tz[ qp ]; 
                    ux_qp = ux_qp + uvalsLocal[ j ] * phigrad_qj * detJ[ qp, eid ] * wg[ qp ];
                end

                newUvals[i] = newUvals[i] + phigrad_qi * ux_qp;
            end
        end

        for localNodeId = 1:nodes_per_element

            globalNodeId = loc2glb[ localNodeId, eid ];

            if( nodebid[ globalNodeId ] == 0 )
                solutionVals[ globalNodeId ] = solutionVals[ globalNodeId ] .+ newUvals[ localNodeId ];
            end
        end
    end

    return solutionVals;
end


## Serial MATVEC adapted for GPU implementation with fused loops (GPU serial version 3)
# Takes less per element memory 
function performMATVEC_version3( float_type, allnodes, loc2glb, uvals, nodebid, detJ, J, qnodes_per_element, 
    nodes_per_element, Qr, Qs, Qt, wg )

    nnodes = size( allnodes, 2 );
    nElements = size( loc2glb, 2 );

    uvalsLocal = zeros(float_type, nodes_per_element, 1);
    solutionVals = zeros( float_type, nnodes, 1 );

    for eid = 1:nElements

        newUvals = zeros(float_type, nodes_per_element, 1);

        ## All directions MATVEC Product in one loop
        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        for qp = 1:qnodes_per_element
            for i = 1:nodes_per_element

                phigradx_qi = Qr[ qp, i ] * J[eid].rx[ qp ] + Qs[ qp, i ] * J[eid].sx[ qp ] + Qt[ qp, i ] * J[eid].tx[ qp ]; 
                phigrady_qi = Qr[ qp, i ] * J[eid].ry[ qp ] + Qs[ qp, i ] * J[eid].sy[ qp ] + Qt[ qp, i ] * J[eid].ty[ qp ]; 
                phigradz_qi = Qr[ qp, i ] * J[eid].rz[ qp ] + Qs[ qp, i ] * J[eid].sz[ qp ] + Qt[ qp, i ] * J[eid].tz[ qp ]; 

                ux_qp = 0;
                uy_qp = 0;
                uz_qp = 0;

                for j = 1:nodes_per_element

                    phigradx_qj = Qr[ qp, j ] * J[eid].rx[ qp ] + Qs[ qp, j ] * J[eid].sx[ qp ] + Qt[ qp, j ] * J[eid].tx[ qp ]; 
                    phigrady_qj = Qr[ qp, j ] * J[eid].ry[ qp ] + Qs[ qp, j ] * J[eid].sy[ qp ] + Qt[ qp, j ] * J[eid].ty[ qp ]; 
                    phigradz_qj = Qr[ qp, j ] * J[eid].rz[ qp ] + Qs[ qp, j ] * J[eid].sz[ qp ] + Qt[ qp, j ] * J[eid].tz[ qp ]; 

                    ux_qp = ux_qp + uvalsLocal[ j ] * phigradx_qj * detJ[ qp, eid ] * wg[ qp ];
                    uy_qp = uy_qp + uvalsLocal[ j ] * phigrady_qj * detJ[ qp, eid ] * wg[ qp ];
                    uz_qp = uz_qp + uvalsLocal[ j ] * phigradz_qj * detJ[ qp, eid ] * wg[ qp ];

                end

                newUvals[i] = newUvals[i] + phigradx_qi * ux_qp + phigrady_qi * uy_qp + phigradz_qi * uz_qp;
            end
        end

        for localNodeId = 1:nodes_per_element

            globalNodeId = loc2glb[ localNodeId, eid ];

            if( nodebid[ globalNodeId ] == 0 )
                solutionVals[ globalNodeId ] = solutionVals[ globalNodeId ] .+ newUvals[ localNodeId ];
            end
        end
    end

    return solutionVals;
end

## GPU MATVEC implementation with fused loops (GPU version 3)
# Takes less per thread memory but very slightly more without fused loops 
function performGPUMATVEC_version3( float_type, allnodes, loc2glb, uvals, nodebid, nnodes, nElements, detJ, J,
    qnodes_per_element, nodes_per_element, Qr, Qs, Qt, wg )

    uvalsLocal = zeros(float_type, nodes_per_element, 1);
    solutionVals = zeros( float_type, nnodes, 1 );

    for eid = 1:nElements

        newUvals = zeros(float_type, nodes_per_element, 1);

        ## All directions MATVEC Product in one loop
        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        for qp = 1:qnodes_per_element
            for i = 1:nodes_per_element

                phigradx_qi = Qr[ qp, i ] * J[eid].rx[ qp ] + Qs[ qp, i ] * J[eid].sx[ qp ] + Qt[ qp, i ] * J[eid].tx[ qp ]; 
                phigrady_qi = Qr[ qp, i ] * J[eid].ry[ qp ] + Qs[ qp, i ] * J[eid].sy[ qp ] + Qt[ qp, i ] * J[eid].ty[ qp ]; 
                phigradz_qi = Qr[ qp, i ] * J[eid].rz[ qp ] + Qs[ qp, i ] * J[eid].sz[ qp ] + Qt[ qp, i ] * J[eid].tz[ qp ]; 

                ux_qp = 0;
                uy_qp = 0;
                uz_qp = 0;

                for j = 1:nodes_per_element

                    phigradx_qj = Qr[ qp, j ] * J[eid].rx[ qp ] + Qs[ qp, j ] * J[eid].sx[ qp ] + Qt[ qp, j ] * J[eid].tx[ qp ]; 
                    phigrady_qj = Qr[ qp, j ] * J[eid].ry[ qp ] + Qs[ qp, j ] * J[eid].sy[ qp ] + Qt[ qp, j ] * J[eid].ty[ qp ]; 
                    phigradz_qj = Qr[ qp, j ] * J[eid].rz[ qp ] + Qs[ qp, j ] * J[eid].sz[ qp ] + Qt[ qp, j ] * J[eid].tz[ qp ]; 

                    ux_qp = ux_qp + uvalsLocal[ j ] * phigradx_qj * detJ[ qp, eid ] * wg[ qp ];
                    uy_qp = uy_qp + uvalsLocal[ j ] * phigrady_qj * detJ[ qp, eid ] * wg[ qp ];
                    uz_qp = uz_qp + uvalsLocal[ j ] * phigradz_qj * detJ[ qp, eid ] * wg[ qp ];

                end

                newUvals[i] = newUvals[i] + phigradx_qi * ux_qp + phigrady_qi * uy_qp + phigradz_qi * uz_qp;
            end
        end

        for localNodeId = 1:nodes_per_element

            globalNodeId = loc2glb[ localNodeId, eid ];

            if( nodebid[ globalNodeId ] == 0 )
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

    detj::Vector{globalFloatType} = zeros(configObj.float_type, qnodes_per_element);  
    
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

function testFullComputation( A, fValsRHS  )

    U = gmres( A, fValsRHS; reltol = 1e-14, verbose = true )

    return U;

end

function getParameters( configObj, fileVal )

    meshDataVal = read_mesh( fileVal );
    global refelVal, gridDataVal = grid_from_mesh( meshDataVal, configObj );
    nnodes = size( gridDataVal.allnodes, 2 );

    global geoFacs = build_geometric_factors( configObj, refelVal, gridDataVal, do_face_detj = false, 
        do_vol_area = false, constant_jacobian = false );

    uvals = zeros( configObj.float_type, nnodes, 1 );
    rhsNodeVals = zeros( configObj.float_type, nnodes, 1 );
    exactvals = zeros( configObj.float_type, nnodes, 1 );

    for nodeId = 1:nnodes

        nodeCoords = gridDataVal.allnodes[ :, nodeId ];
        uvals[ nodeId, 1 ] = initFunction( nodeCoords[1], nodeCoords[2], nodeCoords[3] );
        rhsNodeVals[ nodeId, 1 ] = rhsNodalFunction( nodeCoords[1], nodeCoords[2], nodeCoords[3] );
        exactvals[ nodeId, 1 ] = exactSol( nodeCoords[1], nodeCoords[2], nodeCoords[3] );
    end

    global fValsRHS = zeros( configObj.float_type, nnodes, 1 );
    getRHS( configObj, geoFacs, refelVal, gridDataVal, rhsNodeVals, fValsRHS  );

    return [ gridDataVal, refelVal, geoFacs, exactvals, fValsRHS ]
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

gmshFileName = gmshFolderName * "regularMesh3D_lvl0.msh";
fileVal = open( gmshFileName )
testMATVEC( fileVal, configObj )