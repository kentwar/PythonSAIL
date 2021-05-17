import numpy as np
import multiprocessing
from functools import partial

################################################################################
################## Main Function ###############################################
################################################################################

def niche_compete( points , map , domain):
    '''nicheCompete - results of competition with map's existing elites

Syntax:  [replaced, replacement] = nicheCompete(map,sampled_points)

Inputs:
    map             - [Array]      - Current map of individuals and performance
    points          - [Array]      - List of lists of observations 
            [i][0]  - [Array] NxG  - Genomes
            [i][1]  - [Array] Nx1  - Observed fitness Value 
            [i][2]  - [Array] NxB  - Observed behavioural values 


Outputs:
    niche_index     - [NX1] - Index of map cells (niches) to recieve replacements
    point_index     - [NX1] - index of points to replace current points in map

    '''

    # update_dict is an empty dictionary (will hold best performing points)
    update_dict = {} 

    #indexes = tuple( nichefinder ( points[:, 2 ] , map , domain) ) 
    for count, point in enumerate( points ):
        #find niche/map reference from behaviour
        index = tuple( nichefinder ( point[ 2 ] , map , domain) ) 
        #convert index to string for dictionary key
        strindex = str( index )     
        try:
            fitness = point[ 1 ]
            current_best = update_dict[ strindex ][ 1 ]
            # Compare against current observed points (NOT ELITES) 
            if fitness > current_best:
                update_dict[ strindex ] = point                
            # Try loop will fail if no point exists in the same niche meaning
            # the except loop will trigger, creating a dictionary item
        except:
            update_dict[ strindex ] = point

    # Compare to Elites
    update_count = 0 #Number of children added to map.

    for key , value in update_dict.items():
        try:
            indextup = keyToTuple( key )
        #If there is an improvement over elites, replace them in the map
            if ( value[ 1 ] > map.fitness[ indextup ] or 
                    np.isnan( map.fitness[ indextup ] ) ):
                map.fitness[ indextup ] = value[ 1 ]
                map.genomes[ indextup ] = value[ 0 ]      
                update_count += 1
        except:
            continue
    improvement_percentage = update_count/len( points )

    return( map , improvement_percentage )


################################################################################
################## Tools #######################################################
################################################################################

def keyToTuple(string):  
    '''Tool to transform a string in to a tuple (as index for multidimensional array) 
    '''
    
    string = string.split(',') 
    import re
    string =[re.sub('\ |\(|\)|\,|\/|\;|\:', '', stri) for stri in string]
    vals = [ int( i ) for i in string ]
    return( tuple( vals ) )

def edgefinder( edges , x , n=0 ):
    '''returns the niche index in a single dimension from the list of edges.
    This is a recursive search algorithm which is *probably* not the most 
    efficient method to find the index, but in tests performed up to 10* faster 
    than np.digitize method.
    '''
    #TODO - Explore better implementation (In trials this beat built in methods)

    if n == 0:
        if x < edges[ 0 ] or x > edges[ -1 ]:
            #print('Error, feature out of bounds')
            return( np.nan )    
    if x <= edges[ n+1 ]:
        return( n )
    else:
        return( edgefinder( edges , x , n = n + 1 ) )

def nichefinder( behaviour , map ,domain ):
    ''' identifies the niche that a new observation belongs to
    as an array reference.
    '''
    index = []
    dims = domain.feat_dims
    #loop through all dimensions, finding which niche it belongs to

    for dim in range( dims ):
        feature_index = edgefinder(map.edges[ dim ] , behaviour[ dim ] )
        index.append(feature_index)

    ## Returns an array of nans if any behavioural observation is outside range
    if np.isnan( index ).any():
        nanmat = np.empty( dims )
        nanmat[ : ] = np.nan
        return( nanmat )
    return( index )

################################################################################
################## Code Testing ################################################
################################################################################

if __name__ == '__main__':
    #pass
    #point = [[fitness],[behaviour]]
    from createmap import create_map
    #from addtomap import add_to_map
   