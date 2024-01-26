function getAllNodes( gmshfilename )

    file = open( gmshfilename, "r" )

    allLines = readlines( file )

    idxStart = 0
    idxEnd = 0

    for (index, lineval) in enumerate( allLines )

        if lineval == "\$Nodes"
            idxStart = index + 2
        end

        if lineval == "\$EndNodes"
            idxEnd = index
            break
        end

    end

    numNodes = parse( Int, allLines[ idxStart - 1 ] )

    # println( idxStart, idxEnd )

    allNodes = allLines[ idxStart:idxEnd - 1 ]

    nodeMat = Vector{ Vector{Float32} }( undef, numNodes )

    for ( indexval, nodeval ) in enumerate( allNodes )
        nodeMat[ indexval ] = [ parse(Float32, d) for d in split( nodeval, " ") ]
    end

    return numNodes, nodeMat

end

function getNumNodes( gmshfilename )

    fileVal = open( gmshfilename, "r" )

    idxStart = 0
    idxEnd = 0

    numNodes = 0
    while !eof( fileVal )

        lineval = readline( fileVal )

        if lineval == "\$Nodes"
            nodeLineVal = readline( fileVal )
            numNodes = parse( Int64, nodeLineVal )
            break
        end
    end

    return numNodes

end

function getNumNodesFolder( gmshFolderName )

    numNodes = []

    for meshVal in meshVals
        append!( numNodes, getNumNodes( meshVal ) )
    end

    return numNodes
end

function getElementNodeIndices( gmshfilename )

    file = open( gmshfilename, "r" )

    allLines = readlines( file )

    idxStart = 0
    idxEnd = 0
    
    for (index, lineval) in enumerate( allLines )

        if lineval == "\$Elements"
            idxStart = index + 2
        end

        if lineval == "\$EndElements"
            idxEnd = index
            break
        end
    end

    numElements = parse( Int, allLines[ idxStart - 1 ] )

    eleTypes = [ "5" ]
    pointType = "15"

    elementLines = allLines[ idxStart:idxEnd - 1 ]
    bcNodes = []

    elementMat = Vector{ Vector{Float32} }( undef, numElements )

    for (index, lineval) in enumerate( elementLines )

        lineval = lineval.split()

        if lineval[2] in eleTypes

            ntags = lineval[3]
            indices = lineval[ 4 + int(ntags): end ]
            indices = [ round(Int, index) for index in indices ]

            elementMat.append( indices )
        elseif lineval[2] == pointType 

            bcNodes.append( round( Int, lineval[end] ) )
        end
        
    end

    return elementMat
end