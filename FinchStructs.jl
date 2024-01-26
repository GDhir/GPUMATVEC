export Refel, GeometricFactors, MeshData, Grid, Jacobian
include("mesh_data.jl")
using CUDA 

mutable struct Refel{T<:AbstractFloat}
    dim::Int                # Dimension
    N::Int                  # Order of polynomials
    Np::Int                 # Number of nodes
    Nqp::Int                # Number of quadrature points
    Nfaces::Int             # Number of faces
    Nfp::Vector{Int}       # Number of nodes for each face
    
    ######################################
    # Volume nodes and quadrature matrices
    ######################################
    r1d::Vector{T}     # Node coordinates in 1D
    r::Matrix{T}       # dim-dim Node coordinates
    
    wr1d::Vector{T}    # r1d gll Quadrature weights
    wr::Vector{T}      # r gll Quadrature weights
    
    g1d::Vector{T}     # 1D Gauss points
    wg1d::Vector{T}    # 1D Gauss weights
    
    g::Matrix{T}       # dim-dim Gauss points
    wg::Vector{T}      # dim-dim Gauss weights
    
    V::Matrix{T}       # basis at r
    gradV::Matrix{T}   # grad of basis at r
    invV::Matrix{T}    # Inverse V
    
    Vg::Matrix{T}      # basis at Gauss
    gradVg::Matrix{T}  # grad of basis at g
    invVg::Matrix{T}   # Inverse Vg
    
    Dr::Matrix{T}      # Differentiation matrix for r
    Ds::Matrix{T}      # Differentiation matrix for s
    Dt::Matrix{T}      # Differentiation matrix for t
    Dg::Matrix{T}      # Differentiation matrix for g
    
    # Useful quadrature matrices for the volume integrals
    Q1d::Matrix{T}     # 1D quadrature matrix: like Vg*invV
    Q::Matrix{T}       # dim-dim quadrature matrix
    Qr::Matrix{T}      # quad of derivative matrix: like gradVg*invV
    Qs::Matrix{T}      # 
    Qt::Matrix{T}      # 
    
    Ddr::Matrix{T}      # Derivatives at the elemental nodes, not quadrature nodes
    Dds::Matrix{T}      # 
    Ddt::Matrix{T}      #
    
    #######################################
    # Surface nodes and quadrature matrices
    #######################################
    face2local::Vector{Vector{Int}}       # maps face nodes to local indices
    
    surf_r::Vector{Matrix{T}}       # surface node coordinates
    surf_wr::Vector{Vector{T}}      # surface gll weights
    
    surf_g::Vector{Matrix{T}}       # surface Gauss points
    surf_wg::Vector{Vector{T}}      # surface Gauss weights
    
    surf_V::Vector{Matrix{T}}       # basis at surf_r
    surf_gradV::Vector{Matrix{T}}   # grad of basis at surf_r
    surf_invV::Vector{Matrix{T}}    # Inverse surf_V
    
    surf_Vg::Vector{Matrix{T}}      # basis at surf_g
    surf_gradVg::Vector{Matrix{T}}  # grad of basis at surf_g
    surf_invVg::Vector{Matrix{T}}   # Inverse surf_Vg
    
    surf_Dr::Vector{Matrix{T}}      # Differentiation matrix for surf_r
    surf_Ds::Vector{Matrix{T}}      # Differentiation matrix for surf_s
    surf_Dt::Vector{Matrix{T}}      # Differentiation matrix for surf_t
    surf_Dg::Vector{Matrix{T}}      # Differentiation matrix for surf_g
    
    surf_Q::Vector{Matrix{T}}       # quadrature matrix
    surf_Qr::Vector{Matrix{T}}      # derivative quadrature matrix
    surf_Qs::Vector{Matrix{T}}      # 
    surf_Qt::Vector{Matrix{T}}      # 
    
    surf_Ddr::Vector{Matrix{T}}     # Derivatives at the elemental nodes, not quadrature nodes
    surf_Dds::Vector{Matrix{T}}     # 
    surf_Ddt::Vector{Matrix{T}}     #
    
    # Constructor needs at least this information
    Refel(T::DataType, dim, order, nnodes, nfaces, nfp) = new{T}(
        dim,
        order,
        nnodes,
        -1,
        nfaces,
        nfp,
        zeros(T,0),zeros(T,0,0),
        zeros(T,0),zeros(T,0),
        zeros(T,0),zeros(T,0),
        zeros(T,0,0),zeros(T,0),
        zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),
        zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),
        zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),
        zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),
        zeros(T,0,0),zeros(T,0,0),zeros(T,0,0),
        [zeros(Int,0)],
        [zeros(T,0,0)],[zeros(T,0)],
        [zeros(T,0,0)],[zeros(T,0)],
        [zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)],
        [zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)],
        [zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)],
        [zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)],
        [zeros(T,0,0)],[zeros(T,0,0)],[zeros(T,0,0)]
    )
end

struct Jacobian{T<:AbstractFloat}
    rx::Vector{T}
    ry::Vector{T}
    rz::Vector{T}
    sx::Vector{T}
    sy::Vector{T}
    sz::Vector{T}
    tx::Vector{T}
    ty::Vector{T}
    tz::Vector{T}
end

# struct JacobianDevice
#     rx_d::CuArray
#     ry_d::CuArray
#     rz_d::CuArray
#     sx_d::CuArray
#     sy_d::CuArray
#     sz_d::CuArray
#     tx_d::CuArray
#     ty_d::CuArray
#     tz_d::CuArray

#     JacobianDevice( rx, ry, rz, sx, sy, sz, tx, ty, tz ) = 
#     ( new( CuArray( rx ), CuArray( ry ), CuArray( rz ), CuArray( sx ), CuArray( sy ), CuArray( sz ), 
#         CuArray( tx ), CuArray( ty ), CuArray( tz ) );
#     )
# end

struct GeometricFactors{T<:AbstractFloat}
    J::Vector{Jacobian{T}}        # Jacobian for each element
    detJ::Matrix{T}      # Determinant of Jacobian for each element
    
    # These below are only computed if needed, otherwise empty arrays
    volume::Vector{T}    # Volume of each element (used by FV)
    face_detJ::Vector{T}  # Determinant of Jacobian for each face (used by DG and FV)
    area::Vector{T}      # Area of each face (used by FV)
end

struct MeshData
    #### Minimal required information ####
    # Nodes
    nx::Int;                    # Number of vertices
    nodes::Array{Float64,2};    # vertex locations (array has size (dim,nx))
    indices::Array{Int,1};      # vertex indices may not be in order
    # Elements
    nel::Int;                   # Number of elements
    elements::Array{Int,2};     # Element vertex mapping (array has size (Np, nel))*assumes only one element type
    etypes::Array{Int,1};       # Element types as defined by GMSH
    nv::Array{Int,1};           # Number of vertices for each element. Only different if they are different types,
    
    #### Optional information that will be built if not provided ####
    invind::Array{Int,1}        # Inverse of indices, maps vertex index to position in nodes array (invind[indices[i]] = i)
    face2vertex::Array{Int,2}   # Vertices defining each face (array has size (Nfp, Nfaces))
    face2element::Array{Int,2}  # Indices of elements on each side of the face. If 0, it is a boundary face. (size is (2,Nfaces))
    element2face::Array{Int,2}  # Indices of faces on each side of the element. (size is (NfacesPerElement, nel))
    normals::Array{Float64,2}   # Normal vectors for each face pointing from first to second in face2element order (size is (dim, Nfaces))
    bdryID::Array{Int,1}        # Boundary ID for each face (0=interior face)
    mixed_elements::Bool        # Are there mixed element types
    
    # The minimal constructor needs to build the optional information.
    # Note: Must uncomment to build.
    MeshData(n, x, ind, ne, el, et, v) = (
        # inv = invert_index(ind);
        # face2v = Array{Int,2}(undef,0,0);
        # face2e = Array{Int,2}(undef,0,0);
        # e2face = Array{Int,2}(undef,0,0);
        # norms = Array{Float64,2}(undef,0,0);
        # bdry = Array{Int,1}(undef,0);
        
        # uncomment these to compute. WARNING: can be slow
        inv = invert_index(ind);
        ismixed = maximum(et) > minimum(et);
        (face2v, face2e, e2face) = build_faces(ne, el, et, ismixed);
        norms = find_normals(face2v, x);
        bdry = find_boundaries(face2e);
        new(n, x, ind, ne, el, et, v, inv, face2v, face2e, e2face, norms, bdry, ismixed);
    )
    # The complete constructor
    MeshData(n, x, ind, ne, el, et, v, inv, face2v, face2e, e2face, norms, bdry) = (
        ismixed = maximum(et) > minimum(et);
        new(n, x, ind, ne, el, et, v, inv, face2v, face2e, e2face, norms, bdry, ismixed);
    )
    # An empty mesh
    MeshData() = new(
        0, zeros(0,0), zeros(Int,0), 0, zeros(0,0), zeros(Int,0), zeros(Int,0), zeros(Int,0),
        zeros(Int,0,0), zeros(Int,0,0), zeros(Int,0,0), zeros(0,0), zeros(Int,0), false
    )
end

struct Grid{T<:AbstractFloat}
    # nodes
    allnodes::Matrix{T}         # All node coordinates size = (dim, nnodes)
    
    # boundaries
    bdry::Vector{Vector{Int}}        # Indices of boundary nodes for each BID (bdry[bid][nodes])*note:array of arrays
    bdryface::Vector{Vector{Int}}    # Indices of faces touching each BID (bdryface[bid][faces])*note:array of arrays
    bdrynorm::Vector{Matrix{T}}    # Normal vector for boundary nodes for each BID (bdrynorm[bid][dim, nodes])*note:array of arrays
    bids::Vector{Int}                # BID corresponding to rows of bdrynodes
    nodebid::Vector{Int}             # BID for every node in allnodes order(interior=0)
    
    # elements
    loc2glb::Matrix{Int}             # local to global map for each element's nodes (size is (Np, nel))
    glbvertex::Matrix{Int}           # global indices of each elements' vertices (size is (Nvertex, nel))
    
    # faces (For CG, G=1. For DG, G=2)
    face2glb::Array{Int}             # local to global map for faces (size is (Nfp, G, Nfaces))
    element2face::Matrix{Int}        # face indices for each element (size is (Nfaces, nel))
    face2element::Matrix{Int}        # elements on both sides of a face, 0=boundary (size is (2, Nfaces))
    facenormals::Matrix{T}         # normal vector for each face
    faceRefelInd::Matrix{Int}        # Index for face within the refel for each side
    facebid::Vector{Int}             # BID of each face (0=interior face)
    
    mixed_elements::Bool            # are there mixed element types
    el_type::Vector{Int8}            # element type for each element or empty
    refel_ind::Vector{Int8}          # index of each element's refel in the refels array
    
    # When partitioning the grid, this stores the ghost info.
    # Items specifying (for solver type) will be empty/0 for other types.
    is_subgrid::Bool            # Is this a partition of a greater grid?
    elemental_order::Vector{Int}     # Order used in elemental loops
    nel_global::Int             # Number of global elements
    nel_owned::Int              # Number of elements owned by this partition
    nel_ghost::Int              # Number of ghost elements (for FV)
    nface_owned::Int            # Number of faces owned by this partition
    nface_ghost::Int            # Number of ghost faces that are not owned (for FV)
    nnodes_global::Int          # Number of global nodes
    nnodes_borrowed::Int        # Number of nodes borrowed from another partition (for CG)
    element_owner::Vector{Int}       # The rank of each element's owner or -1 if locally owned (for FV)
    node_owner::Vector{Int}          # The rank of each node's owner (for FE)
    partition2global_element::Vector{Int}           # Map from partition elements to global mesh element index
    partition2global::Vector{Int}    # Global index of nodes (for CG,DG)
    global_bdry_index::Vector{Int8}   # Index in bids for every global node, or 0 for interior (Only proc 0 holds, only for FE)
    
    num_neighbor_partitions::Int   # number of partitions that share ghosts with this.
    neighboring_partitions::Vector{Int} # IDs of neighboring partitions
    ghost_counts::Vector{Int}           # How many ghost elements for each neighbor (for FV)
    ghost_index::Vector{Matrix{Int}}     # Lists of ghost elements to send/recv for each neighbor (for FV)
    
    # constructors
    # up to facebid only: not partitioned
    Grid(T::DataType, allnodes, bdry, bdryfc, bdrynorm, bids, nodebid, loc2glb, glbvertex, f2glb, element2face, 
         face2element, facenormals, faceRefelInd, facebid) = 
     new{T}(allnodes, bdry, bdryfc, bdrynorm, bids, nodebid, loc2glb, glbvertex, f2glb, element2face, 
         face2element, facenormals, faceRefelInd, facebid, false, zeros(Int8,0),  zeros(Int8,0),
         false, Array(1:size(loc2glb,2)), size(loc2glb,2), size(loc2glb,2), 0,size(face2element,2), 0, 0, 0, zeros(Int,0), zeros(Int,0), 
         zeros(Int,0), zeros(Int,0), zeros(Int8,0), 0, zeros(Int,0), zeros(Int,0), [zeros(Int,2,0)]);
         
    Grid(T::DataType, allnodes, bdry, bdryfc, bdrynorm, bids, nodebid, loc2glb, glbvertex, f2glb, element2face, 
         face2element, facenormals, faceRefelInd, facebid, ismixed, eltypes, refelind) = 
     new{T}(allnodes, bdry, bdryfc, bdrynorm, bids, nodebid, loc2glb, glbvertex, f2glb, element2face, 
         face2element, facenormals, faceRefelInd, facebid, ismixed, eltypes, refelind,
         false, Array(1:size(loc2glb,2)), size(loc2glb,2), size(loc2glb,2), 0,size(face2element,2), 0, 0, 0, zeros(Int,0), zeros(Int,0), 
         zeros(Int,0), zeros(Int,0), zeros(Int8,0), 0, zeros(Int,0), zeros(Int,0), [zeros(Int,2,0)]);
    
    # full: partitioned
    Grid(T::DataType, allnodes, bdry, bdryfc, bdrynorm, bids, nodebid, loc2glb, glbvertex, f2glb, element2face, 
         face2element, facenormals, faceRefelInd, facebid, ismixed, eltypes, refelind,
         ispartitioned, el_order, nel_global, nel_owned, nel_ghost, nface_owned, nface_ghost, nnodes_global, nnodes_borrowed, element_owners, 
         node_owner, partition2global_element, partition2global, glb_bid, num_neighbors, neighbor_ids, ghost_counts, ghost_ind) = 
     new{T}(allnodes, bdry, bdryfc, bdrynorm, bids, nodebid, loc2glb, glbvertex, f2glb, element2face, 
         face2element, facenormals, faceRefelInd, facebid, ismixed, eltypes, refelind,
         ispartitioned, el_order, nel_global, nel_owned, nel_ghost, nface_owned, nface_ghost, nnodes_global, nnodes_borrowed, element_owners, 
         node_owner, partition2global_element, partition2global, glb_bid, num_neighbors, neighbor_ids, ghost_counts, ghost_ind); # subgrid parts included
         
    # An empty Grid
    Grid(T::DataType) = new{T}(
        zeros(T,0,0),[zeros(Int,0)],[zeros(Int,0)],[zeros(T,0,0)],zeros(Int,0),zeros(Int,0),
        zeros(Int,0,0),zeros(Int,0,0),
        zeros(Int,0,0,0),
        zeros(Int,0,0), zeros(Int,0,0), zeros(T,0,0), zeros(Int,0,0),
        zeros(Int,0),
        false, zeros(Int8,0), zeros(Int8,0),
        false,zeros(Int,0),0,0,0,0,0,0,0,zeros(Int,0),zeros(Int,0),zeros(Int,0),zeros(Int,0),zeros(Int8,0),
        0,zeros(Int,0),zeros(Int,0),[zeros(Int,0,0)]
    )
end

mutable struct FinchConfig
    # Domain
    dimension::Int          # 1,2,3
    geometry::String        # square, irregular
    mesh_type::String       # unstructured, tree, uniform grid
    
    # Discretization details
    solver_type::String     # cg, dg, fv
    
    # FEM
    trial_function::String  # Legendre
    test_function::String   # same as above
    elemental_nodes::String # uniform, gauss, lobatto (higher order node distribution within elements)
    quadrature::String      # uniform, gauss, lobatto (similar to above)
    p_adaptive::Bool        # Do adaptive p-refinement?
    basis_order_min::Int    # minimum order to use in p-refinement, or if p_adaptive is false
    basis_order_max::Int    # maximum order
    
    # FVM
    fv_order::Int           # Order of reconstruction at faces
    
    # Time stepping
    t_adaptive::Bool        # Do adaptive t_refinement?
    stepper::String         # Euler-explicit/implicit, RK4, LSRK4, etc. Type of time stepper to use
    
    # Other solver details
    linear::Bool            # Is the equation linear?
    linalg_matrixfree::Bool             # Use matrix free methods?
    linalg_iterative::Bool              # Use an iterative solver?
    linalg_iterative_method::String     # GMRES or CG
    linalg_iterative_pc::String         # AMG, ILU, NONE
    linalg_iterative_maxiter::Int       # max iters for iterative solver
    linalg_iterative_abstol::Float64    # absolute tolerance
    linalg_iterative_reltol::Float64    # relative tolerance
    linalg_iterative_gmresRestart::Int  # GMRES restart iterations
    linalg_iterative_verbose::Bool      # print convergence info?
    
    linalg_usePetsc::Bool   # use PETSc?
    
    # Output
    output_format::String   # VTK, raw, custom (format for storing solutions)
    
    # Parallel details
    use_mpi::Bool           # Is MPI available?
    num_procs::Int          # number of processes
    proc_rank::Int          # this proccess rank
    num_threads::Int        # number of available threads
    num_partitions::Int     # number of mesh partitions
    partition_index::Int    # this process's partition
    use_gpu::Bool           # Use GPU?
    partitioner::String     # The partitioning library/method to use
    
    # Data types
    index_type::Type
    float_type::Type
    
    # Cachesim
    use_cachesim::Bool
    
    # Constructor builds a default config.
    FinchConfig() = new(
        3,
        SQUARE,
        UNIFORM_GRID,
        
        CG,
        LEGENDRE,
        LEGENDRE,
        LOBATTO,
        GAUSS,
        false,
        1,
        1,
        
        1,
        
        false,
        EULER_IMPLICIT,
        
        true,
        false,
        true,
        "GMRES",
        "ILU",
        0,
        0,
        1e-8,
        0,
        false,
        false,
        
        VTK,
        
        false,
        1,
        0,
        1,
        1,
        0,
        false,
        METIS,
        
        Int64,
        Float32,
        
        false
    );
end