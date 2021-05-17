## Allow modules to run using their own module references.
import sys, os
sys.path.insert(0, os.path.dirname(__file__))


import numpy as np
from createmap import create_map
from createchildren import create_children
from nichecompete import niche_compete
import multiprocessing
import math


def map_elites( domain, 
                fit_fun = None ,
                feat_fun = None ,  
                init_map = None ,
                init_n = 2**8 , 
                plot = False ,
                verbose = False ,
                me_params = None ,
                return_points = False ):
      
    def sample( n , 
                domain ,
                fitness_fun  ,
                feature_fun ,
                sample_map = None , 
                init = False , 
                plot = True ,
                mutation_prob = 0.4,):
        '''Sample performs the calls to the objective and feature functions and
        returns them in the correct format for Map-Elites 
        '''
        population = np.array(create_children( batchsize = n ,
                                        domain = domain ,
                                        map = sample_map , 
                                        initialise = init, 
                                        plot = plot ))
        fitness = fitness_fun(population)
        behaviour = feature_fun(population )       
        sampled_points = [ [population[i] , fitness[i] , behaviour[i]] for i in range(n) ]
        return( sampled_points )
    
#######################################################################
################# Initialising ########################################
#######################################################################  
    # num_cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(num_cores)
    fdims = domain.feature_resolution
    mutation_prob, pop_size, mut_sigma, n_gens  = me_params #Probability of performing 1 slice cross mutation
    max_evals = pop_size * n_gens
    mins = domain.feat_mins
    maxs = domain.feat_maxs 
    example = domain.example_genome
    xdims = len(example)

    points_list = []

    num_evals = 0

    sampled_points = sample(n = init_n,
                            domain = domain ,
                            sample_map = init_map, 
                            init = True , 
                            fitness_fun = fit_fun , 
                            feature_fun = feat_fun ,
                            plot = plot ,
                            mutation_prob= mutation_prob)


    mymap, improvement_percentage = niche_compete( sampled_points , init_map , domain )
    
    num_evals += init_n
#######################################################################
################### Map-Elites #######################################
#######################################################################
    popsize = pop_size ; generation = 1

    while num_evals < max_evals:

        new_samples = sample(   n = popsize,
                                domain = domain ,  
                                sample_map = mymap , 
                                fitness_fun = fit_fun , 
                                feature_fun = feat_fun ,
                                plot = plot,
                                mutation_prob= mutation_prob)

        num_evals += popsize

        sampled_points.extend(new_samples)
        mymap, improvement_percentage = niche_compete( new_samples , mymap , domain)
        if verbose:
            print('generation ',generation, ' accepted ',improvement_percentage,' % of points')
            generation +=1


    #sampled = [s[0] for s in sampled_points]
    # x1 = [x[0] for x in sampled]
    # x2 = [x[1] for x in sampled]
    # plt.scatter(x1,x2)
    # plt.show()
    if return_points:
        return( mymap , sampled_points )
    else:
        return( mymap )


# def parallel_eval(evaluate_function, to_evaluate, pool):
#     s_list = pool.map(evaluate_function, to_evaluate)
#     return s_list

if __name__ == '__main__':
    #import domain.simple2d as mp
    # prediction_map , _ = create_map( mp.feature_resolution , mp.example)
    # mymap = map_elites(mp.domain , init_map=prediction_map)
    

    def calculate_final_score(mymap , domain , fitness_fun):
        '''This function takes the predictive map and calculates the predicted score
        by assessing their value on the real functions.
        '''
        xdims = len(domain.valid_ranges)
        genomes = mymap.genomes[ ~np.isnan( mymap.fitness ) ].flatten( )
        truevals = [ fitness_fun( x ) for x in np.reshape( genomes, [ -1,xdims ] ) ]
        return( np.nansum( truevals ) )

    def rastrigin(xx):
        if xx.shape == (10,):
            x = np.dot(xx , 10) - 5
            f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * np.pi * x)).sum()
        else: 
            x = np.dot(xx , 10) - 5 # scaling to [-5, 5]
            f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * np.pi * x)).sum(axis=1)
        return(-f)

    def feature_fun(genomes ):
        assert type(genomes)==list or type(genomes) == np.ndarray,'Input to feature_function must be array or list' 
        try:
            x_up = [genome[1] for genome in genomes]
            z_up = [genome[2] for genome in genomes]
            features = zip(x_up,z_up)
            return( list(features) )
        except:
            features = [genomes[i] for i in range(1,3)]
        return( features )

    class Domain:

        def __init__(self, featmins, featmaxs, example_genome, ns,fitness_function=None, feature_function=None):
            self.mutation_sigma = 0.1
            self.valid_ranges =   [[ 0 , 1 ] for i in range(10)]    #input x ranges [min,max]
            self.fit_fun = fitness_function
            self.feat_fun = feature_function
            self.feat_dims = len( featmins )
            self.feat_mins = featmins
            self.feat_maxs = featmaxs
            self.example_genome = example_genome
            self.feature_resolution = [ns,ns]
            self.valid_feat_ranges = [ [featmins[ i ], featmaxs[ i ] ] for i in range(len( featmins ))] 

    domain = Domain( [ 0 , 0 ] , [ 1 , 1 ] , [0,5]*10,10,rastrigin, ( feature_fun ) )
    mprob = 0.0 #Probability of performing 1 slice cross mutation
    n_children         = 2**6; 
    mut_sigma          = 0.1; 
    n_gens             = 2**7
    ME_params = ( mprob  , n_children  , mut_sigma , n_gens )
    mymap , _ = create_map([10,10],domain)
    mymap = map_elites(domain,rastrigin,feature_fun,mymap,me_params=ME_params)
    print(calculate_final_score(mymap,domain,rastrigin))


# %%
