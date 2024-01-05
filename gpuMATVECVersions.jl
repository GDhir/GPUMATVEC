## GPU MATVEC implementation with fused loops (GPU version 3)
# Takes less per thread memory but very slightly more without fused loops 
function performGPUMATVEC_version3( allnodes, loc2glb, uvals, nodebid, nnodes, nElements, detJ, 
    rx, ry, rz, sx, sy, sz, tx, ty, tz, qnodes_per_element, nodes_per_element, Qr, Qs, 
    Qt, wg, solutionVals )

    eid = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    ## All directions MATVEC Product in one loop

    if (eid <= nElements)
        for i = 1:nodes_per_element

            solutionVal = 0;
            globalNodeId_i = loc2glb[ i, eid ];
            
            if( nodebid[ globalNodeId_i ] == 0 )
                for qp = 1:qnodes_per_element                    

                    phigradx_qi = Qr[ qp, i ] * rx[ qp, eid ] + Qs[ qp, i ] * sx[ qp, eid ] + Qt[ qp, i ] * tx[ qp, eid ]; 
                    phigrady_qi = Qr[ qp, i ] * ry[ qp, eid ] + Qs[ qp, i ] * sy[ qp, eid ] + Qt[ qp, i ] * ty[ qp, eid ]; 
                    phigradz_qi = Qr[ qp, i ] * rz[ qp, eid ] + Qs[ qp, i ] * sz[ qp, eid ] + Qt[ qp, i ] * tz[ qp, eid ]; 

                    ux_qp = 0;
                    uy_qp = 0;
                    uz_qp = 0;

                    for j = 1:nodes_per_element

                        globalNodeId_j = loc2glb[ j, eid ];

                        phigradx_qj = Qr[ qp, j ] * rx[ qp, eid ] + Qs[ qp, j ] * sx[ qp, eid ] + Qt[ qp, j ] * tx[ qp, eid ]; 
                        phigrady_qj = Qr[ qp, j ] * ry[ qp, eid ] + Qs[ qp, j ] * sy[ qp, eid ] + Qt[ qp, j ] * ty[ qp, eid ]; 
                        phigradz_qj = Qr[ qp, j ] * rz[ qp, eid ] + Qs[ qp, j ] * sz[ qp, eid ] + Qt[ qp, j ] * tz[ qp, eid ]; 

                        ux_qp = ux_qp + uvals[ globalNodeId_j ] * phigradx_qj * detJ[ qp, eid ] * wg[ qp ];
                        uy_qp = uy_qp + uvals[ globalNodeId_j ] * phigrady_qj * detJ[ qp, eid ] * wg[ qp ];
                        uz_qp = uz_qp + uvals[ globalNodeId_j ] * phigradz_qj * detJ[ qp, eid ] * wg[ qp ];

                    end

                    solutionVal = solutionVal + phigradx_qi * ux_qp + phigrady_qi * uy_qp + phigradz_qi * uz_qp;
                end

                CUDA.@atomic solutionVals[ globalNodeId_i ] = solutionVals[ globalNodeId_i ] + solutionVal
            end
        end
    end

    return nothing;
end