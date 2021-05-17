import numpy as np
import random
import itertools as it
#import matplotlib.pyplot as  plt

#######################################################################
################## Main function ######################################
#######################################################################

def create_children(batchsize, domain, initialise = False, map = None, plot = False , mprob =0):
    '''create_children - produce new children through mutation of map elite

    Syntax:  children = createChildren(d, batchsize)
    
    Inputs:
        batchSize   - [int]     - number of children to create
        initialise  - [Boolean] - Flag to indicate first run of the algorithm

    Global Inputs (from config):
        mp.map      - [MAP]     - map of individuals and performance
            .genomes- [Array]   - genotypes of all individuals
        mp.domaind  - [DOMAIN]  - recombination/variation parameters

    Outputs:
            children- [Array]   - newly created points to sample
    '''
    ### Create local copies 

    genomes = map.genomes ; fitness = map.fitness

    ## create reshaped array of genomes in map
    flatdims = genomes.shape[ 0 ] * genomes.shape[ 1 ]  #number of n-dimensional genomes.
    genomelen = map.genomes.shape[-1] # length of a single genome
    parent_pool = genomes.reshape( [ flatdims, genomelen ] ) # flat array of genomes
    fitness_pool = fitness.reshape( [ flatdims, 1] ) # reshape fitness
    
    # Remove any nans
    if not initialise:
        parent_pool = parent_pool[ [ not np.isnan(x).any() for x in fitness_pool ] ]

    ## uniformly at random select parents from the parent_pool
    index = np.random.randint( 0 , len( parent_pool ), batchsize )
    parents = parent_pool[ index ]
    
    ## perform mutation 
    children = []

    for count, parent in enumerate( parents ):
        if np.random.uniform( 0, 1 ) < mprob:
            ind = np.random.randint( len( parents ) )
            parent = merge_mutate( parents[ count ], parents[ ind ], domain )
        mutant = simple_mutate( domain, parent )
        ## Now ensure that all ranges are legal in the input dimensions
        cliprange( domain, mutant )
        ## finally collate the children
        children.append( list( mutant ) )
    
    ########### For Plotting #################
    ###########################################
    if plot:
        plotchildren( children, genomes, domain )
    #############################################
    #############################################
    return( children )

######################################################################
##################  TOOL FUNCTIONS ###################################
######################################################################

def cliprange(domain, genome):
    '''cliprange - Forces child points to conform to acceptable range

    Syntax:  newchild = cliprange(domain, child_genome)

    Inputs:
        domain      - [DOMAIN]      - Domain specific range information
        genome      - [1xD ARRAY]   - The genotypic data of a child

    Outputs:
        genome      - [1xD Array]   - Conformed genotypic data
    '''
    vrange = domain.valid_ranges

    for index in range( len( vrange ) ):

        if genome[ index ] < vrange[ index ][ 0 ]:
            genome[ index ] = vrange[ index ][ 0 ]

        if genome[ index ] > vrange[ index ][ 1 ]:
            genome[ index ] = vrange[ index ][ 1 ]

    return(genome)

def simple_mutate(domain, genome):
    '''simple_mutate - Performs a simple mutation on every input dimension

    Syntax:  mutant = simple_mutate(genome, domain)

    Inputs:
        domain      - [DOMAIN]      - Domain specific range information
        genome      - [1xD ARRAY]   - The genotypic data of a child

    Outputs:
        mutant      - [1xD Array]   - Mutated Genome
    '''
    ## add gaussian noise scaled by sigma

    mutation = np.random.normal( size = len( genome ) ) * domain.mutation_sigma
    
    return( genome + mutation )

def merge_mutate( genome1, genome2, domain ):
    '''merge_mutate - Performs a split and merge evolution

    Syntax:  mutant = merge_mutate(genome1, genome3, domain)

    Inputs:
        genome1     - [1xD ARRAY]   - The genotypic data of a child
        genome2     - [1xD ARRAY]   - The genotypic data of another child
        domain      - [DOMAIN]      - Domain specific range information


    Outputs:
        child      - [1xD Array]   - Conformed genotypic data
    '''  
    split = np.random.randint( 1 , len( genome1 ) ) # Index to cut at
    child = np.array( [ *genome1[:split] , *genome2[split:] ] ) # Merge

    return( child )

def plotchildren(children, genomes, domain):
    ''' Plotting function for 2d, plots the children (blue) and the 
    elites (orange) '''
    from IPython import display
    import time
    def close_event():
        plt.close() #timer calls this function after 3 seconds and closes the window 

    fig = plt.figure()
    
    #creating a timer object and setting an interval of 1000 milliseconds
    timer = fig.canvas.new_timer( interval = 600 )         
    timer.add_callback( close_event )

    timer.start()

    beh = np.array( eval( 'domain.feat_fun')( children) ) 
    plt.scatter( beh[ : , 0 ] , beh[ : , 1 ] )  
    plt.scatter( [ elite[ 0 ] for elite in genomes.reshape( -1, 2 ) ],
            [ elite[ 1 ] for elite in genomes.reshape( -1 , 2 ) ] )
    plt.xlim( domain.feat_mins[ 0 ], domain.feat_maxs[ 0 ] )
    plt.ylim( domain.feat_mins[ 1 ], domain.feat_maxs[ 1 ] )
    plt.show()

######################################################################
########################## TESTING ###################################
######################################################################


if __name__ == '__main__':
    children = create_children(10 , map = mp.map, plot = True)
    children = mp.map.genomes
    kids = np.array(children).T
    plt.scatter(kids[0], kids[1])