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