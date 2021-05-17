import numpy as np
import itertools as it
'''
    09/09/2020 Made multiple input dimensions for genome map
'''

def create_map( feature_resolution , domain):
    '''createmap - defines the map-elites niche map for N-feature dimensions 
    
    Inputs: 
        feature_resolution  -   [ 1 X N ]   - no. of map partitions (P) per dim
        feature_min         -   [ 1 X N ]   - minimum feature values
        feature_max         -   [ 1 X N ]   - maximum feature values
        sampleInd           -   [ 1X1 ]     - example individual

    Outputs
        map             - Dict - Map/archive
           .edges       - [1  X N ]            - niches edges for behaviour space
           .fitness     - [M(1) X M(2) X M(N)] - fitness value in each bin
           .genomes     - [M(1) X M(2) X M(N)] - individual in each bin
           .otherVals   - [M(1) X M(2) X M(N)] - other value in each bin
  

    Author: Paul Kent
    Warwick University
    email: paul.kent@warwick.ac.uk}
    Aug 2020; Last revision: 27-Aug-2020 
    '''    
    class Map:
        def __init__( self ):
            self.edges = []
            self.fitness = None
            self.genomes = None
            self.dims = None
    
    ## Domain specific information
    sample_ind = domain.example_genome
    fdims = len(domain.feature_resolution)
    mins = domain.feat_mins
    maxs = domain.feat_maxs
    xdims = len(sample_ind)
    
    map = Map()

    #Add niche boundaries/edges to map
    for i in range( fdims ):
        edge_boundaries = np.linspace( mins[ i ] , 
                            maxs[ i ],
                            feature_resolution[ i ] + 1 )

        map.edges.insert( i, edge_boundaries ) 
    
    map.edges = np.array( map.edges )
    map.dims  = fdims
    
    blank_map       = np.empty( feature_resolution )#, dtype = object)
    blank_map[ : ]  = np.nan
    map.fitness     = blank_map #Single fitness value
    map.features    = np.copy(blank_map)

    blank_genome_map = np.empty( ( *feature_resolution , xdims ) )
    map.genomes = blank_genome_map #Multiple input dimensions.
    # creates ranges of feature dimensions (to cycle through the array index)
    genome_dims = [ range( i ) for i in np.shape( map.genomes ) ] 
    # now make it iterable
    iterdims = list( it.product( *genome_dims ) ) 

    for i in iterdims:
        map.genomes[ i[ 0 ] ][ i[ 1 ] ] =  np.array(sample_ind).reshape(xdims)
    #TODO - Add areas of special interest
    return( map, map.edges )


if __name__ == '__main__':
    class behaviour_space:
        def __init__(self, mins, maxs):
            self.dims = len(mins)
            self.mins = mins
            self.maxs = maxs
    
    features = behaviour_space([0,0],[1,2])
    
    #map , edges = create_map( [20,20], [-1]*10, fdims = 2, mins = [0,0], maxs = [1,1] , xdims=10)
    print(np.shape(map.genomes))
    print('fitness '  , map.fitness)
    print('initial genome ' , map.genomes)
