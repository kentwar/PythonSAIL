import numpy as np 
import matplotlib.pyplot as plt
import random
import copy

import setseed as ss
ss.initseed(ss.myseed)

## This import location is defined by the parent script.
## domain_config exists in a folder selected at runtime.
import domain_config as mp

from sampling_lib import sobol_sample, duplicatecheck, keep_valid, keep_valid_singular
from mapelites.nichecompete import nichefinder , keyToTuple
from colorama import Fore,Style

def initial_sampling(n_initial_samples):
    '''initial_sampling -

    Initial samples are produced using a Sobol sequence to evenly sample the
    parameter space. If initial samples are invalid (invalid geometry, or did
    not converge in simulator), the next sample in the Sobol sequence is
    chosen. Lather, rinse, repeat until all initial samples are clean.
    
    Example: initpoints = initialsampling(100)
    
    Inputs (arguments):
       n_initial_samples - [ Integer ] - number of initial samples to produce
      
    Inputs (from config file):
        mp.domain - domain Class object
          .valid_ranges     - [ List ]  - To check validity of samples

    
    Outputs:
       valid_points - [ n_initial_samples * [ [ x ] , [ y ] , [ f ] ] ] ]
    
    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 16-Oct-2020 
    '''
    
    # produce initial samples ##################################################
    
    valid_ranges = mp.constraints #mp.domain.valid_ranges # Search domain ranges
    initial_points = sobol_sample( n_initial_samples , valid_ranges )
    # keep valid samples #######################################################
    valid_points = []
    valid_points , n_missing = keep_valid( valid_points, initial_points , 
                                            n_initial_samples, validate)
    # If not enough initial points, recurse ####################################

    while n_missing > 0 :
        print(n_missing )
        filler_points = sobol_sample( n_missing , valid_ranges )
        valid_points , n_missing = keep_valid( valid_points, filler_points , 
                                                n_missing, validate)
    print('')
    print(f'{Fore.GREEN}Success: {Style.RESET_ALL}initial point selection ')
    print(f'{Fore.YELLOW}Checking{Style.RESET_ALL} initial sample set')
    # Report on issues with initial set ########################################

    assert (duplicatecheck(valid_points) == False , 
    f'{Fore.RED}Found duplicates in the initial set{Style.RESET_ALL}')

    assert (n_initial_samples == len(valid_points) ,
    f'{Fore.RED}Warning, Observation set smaller than specified{Style.RESET_ALL}') 

    print(f'{Fore.GREEN}Completed:{Style.RESET_ALL} initial sampling.')

    return( valid_points )


#TODO Replace the validate function 


def validate(point , funcs = [mp.domain.fit_fun,mp.domain.feat_fun]):
    '''Validate attempts to return valid fitness and feature valuations 
    '''
    ## fitness will return a single value
    try:
        fitness = funcs[ 0 ]( point )
    except:
        fitness = np.nan

    ## feature will return an array of feature evaluations
    try:
        feature = funcs[ 1 ]( point )
    except:
        feature = [np.nan,np.nan]

    #TODO make sure values are within an acceptable range

    return( fitness , feature )

def additional_sampling(n_add_samples  , sobol_set, sobol_point , map ):
    '''additional_sampling -

    Samples are produced using a Sobol sequence that evenly selects elites from 
    the current acquisition map. Those elites are then evaluated and added to the
    observation list (which will improve the surrogate model and also improve the
    prediction map) 
    
    ##TODO If samples are invalid (invalid geometry, or did
    not converge in simulator), the next sample in the Sobol sequence is
    chosen. Lather, rinse, repeat until all initial samples are clean.
    
    Example: new_points = additional_sampling(100 , my_sobol, current_point , map)
    
    Inputs (arguments):
       n_add_samples    - [ Integer ]   - number of samples to produce
       sobol_set        - [ List ]      - A set of random points in feature space
       sobol_point      - [ Integer ]   - An index for the sobol_set 
       map              - [ Map Class ] - The current acquisition map.

    Inputs (from config file):
        mp.domain - domain Class object
          .valid_ranges     - [ List ]  - To check validity of samples

    
    Outputs:
       valid_points - [ n_initial_samples * [ [ x ] , [ y ] , [ f ] ] ] ] 
       sample_end   - [ Integer ]      - An update to sobol_point (an index)  

    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 16-Oct-2020 
    '''
    valid_ranges = mp.domain.valid_ranges  # Valid Search domain ranges
    new_value = []; new_sample = [] ; n_missing = n_add_samples ; new_points = []

    # randomly sample from map #########################################    
    # feat_res = mp.feature_resolution
    
    # n_niches = np.product( feat_res )
    # nichelist = list(range( n_niches ))

    # niche_index1 = np.floor(np.divide(nichelist , feat_res[0]))
    # niche_index2 = nichelist - niche_index1 * feat_res[0]
    # niche_index = [(int(niche_index1[i]), int(niche_index2[i])) for i in range(len(niche_index1))]
    # random.shuffle(niche_index)

    sample_end = sobol_point + 10*n_add_samples
    random_genomes = [ sobol_set[i][sobol_point : sample_end ] for i in range(len(sobol_set))]
    random_genomes = np.array(random_genomes).T
    random_genomes = np.reshape(random_genomes , (-1 , len( valid_ranges ) ) )

    # identify which niche each point belongs to (in feature space)
    niche_index = [nichefinder( random_genomes[ i ] , map , mp.domain) for i in range( len( random_genomes ) ) ]
    # Remove duplicates, keep track of the random genomes index (n) and the feature values (i)
    niche_index = [ i for n, i in enumerate(niche_index) if i not in niche_index[:n]] 
    
    valid_points = []

    while n_missing >0 and len(niche_index) > 0:
        
        niche_id = niche_index[-1]
        #print(niche_id)
        #print(map.fitness[ tuple( niche_id ) ])
        fit = map.fitness[ tuple( niche_id ) ]
        gen = map.genomes[ tuple( niche_id ) ]
        if not np.isnan( fit ).any( ):
            true_fit = mp.fitness_fun( gen )
            true_feat = mp.feature_fun( gen )
            
            if ~np.isnan(true_fit) and ~np.isnan(true_feat).any():
                valid_points.append( [gen , true_fit, true_feat] )
                n_missing -= 1
            else:
                true_fit = 0
                valid_points.append( [gen , true_fit, true_feat] )
                n_missing -= 1
            
            ### Assume NAN is zero
            ### WARNING - TEST Assumption


    
           
        niche_index.pop( )
        # valid = []

        # valid , n_missing = keep_valid( valid, 
        #                                 np.array(new_points).T , 
        #                                 n_missing, 
        #                                 validate )

        # valid_points.append(valid)

        if len( niche_index ) <= 0:
            print( 'Not enough unique samples to make all new evaluations')
            print( 'This can happen on the first few runs')
            print( 'or could indicate a problem with your functions')
            n_missing = 0

      
    if n_missing <= 0:
        print(f'{Fore.GREEN}Success: {Style.RESET_ALL} New points to evaluate chosen from acquisition map')
    #Reshaping
    # print('value check', validate(new_points[0].T))
    
    #new_points = [new_points[i].T for i in range(len(new_points))]
    # print(keep_valid(   new_points.T, 
    #                     new_points , 
    #                     n_add_samples, 
    #                     validate ))
    # new_points = np.array(new_points).T
    #validating and sample from fitness/feature functions
    # valid_points = []

    # valid_points , n_missing = keep_valid(  valid_points, 
    #                                         np.array(new_points).T , 
    #                                         n_add_samples, 
    #                                         validate )
    print(len(valid_points))
    if len(valid_points) == n_add_samples:
        print(f'{Fore.GREEN}Success: {Style.RESET_ALL} New points evaluated')
    elif len(valid_points) ==0:
        print(f'{Fore.RED}FAILURE: {Style.RESET_ALL} No new Points selected ')
    else:
        print(f'{Fore.YELLOW}Warning: {Style.RESET_ALL} Not all new points evaluated correctly ')
    return( valid_points , sample_end, len(valid_points) )



if __name__ == '__main__':
    sample = initial_sampling( 50 )
    print(sample)

