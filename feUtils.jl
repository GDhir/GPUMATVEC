function getGridParameters( configObj, fileVal, orderVal = 1 )

    meshDataVal = read_mesh( fileVal );
    global refelVal, gridDataVal = grid_from_mesh( meshDataVal, configObj, order = orderVal );
    nnodes = size( gridDataVal.allnodes, 2 );

    return [refelVal, gridDataVal, nnodes ]
end

function getParameters( configObj, fileVal, orderVal = 1 )

    refelVal, gridDataVal, nnodes = getGridParameters( configObj, fileVal, orderVal )

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
function getParameters( configObj, fileVal, orderVal = 1 )

    meshDataVal = read_mesh( fileVal );
    global refelVal, gridDataVal = grid_from_mesh( meshDataVal, configObj, order = orderVal );
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