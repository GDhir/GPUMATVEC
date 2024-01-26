

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

## Serial MATVEC adapted for GPU implementation (GPU serial version 1)
# Takes more per element memory
function performMATVEC_version1_Modified( float_type, allnodes, loc2glb, uvals, nodebid, detJ, Jval, qnodes_per_element, 
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

        RQ1 = Qr .* J.rx + Qs .* J.sx + Qt .* J.tx;
        RQ2 = Qr .* J.ry + Qs .* J.sy + Qt .* J.ty;
        RQ3 = Qr .* J.rz + Qs .* J.sz + Qt .* J.tz;

        uvalsLocal[:, 1] = uvals[ loc2glb[ :, eid ] ];

        ugradx = RQ1 * uvalsLocal;
        ugrady = RQ2 * uvalsLocal;
        ugradz = RQ3 * uvalsLocal;
        
        for pval = 1:nodes_per_element
            for qp = 1:qnodes_per_element
                newUvals[ pval, 1 ] = newUvals[ pval, 1 ] + ( 
                    RQ1[ qp, pval ] * ugradx[ qp ] +
                    RQ2[ qp, pval ] * ugrady[ qp ] + 
                    RQ3[ qp, pval ] * ugradz[ qp ] ) * detj[ qp ] * wg[ qp ];
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

        for i = 1:nodes_per_element
            for qp = 1:qnodes_per_element

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