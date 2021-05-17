import numpy as np
from mapelites.createmap import create_map
from mapelites.createchildren import create_children
from mapelites.nichecompete import niche_compete
import domain.set_domain as sd
if sd.domain_flag == 'ffd':
    import domain.problem_config_ffd as mp
if sd.domain_flag == '2d':
    import domain.problem_config_2d as mp
if sd.domain_flag == 'simple2d':
    import domain.simple2d as mp

import matplotlib.pyplot as  plt
import multiprocessing
import math

def map_elites( domain, 
                init_map = None , 
                fit_fun = mp.domain.fit_fun , 
                feat_fun = mp.domain.feat_fun ,
                experiment = False, 
                number_evals = mp.n_gens, 
                plot = False ,
                verbose = False ):

    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    def sample( n , 
                sample_map = None , 
                init = False , 
                fitness_fun = mp.domain.fit_fun , 
                feature_fun = mp.domain.feat_fun ,
                plot = True , 
                sample_pool = pool):
        '''Sample performs the calls to the objective and feature functions and
        returns them in the correct format for Map-Elites 
        '''
        if sample_map == None:
            sample_map = mp.map
        population = np.array( create_children( batchsize = 10**4 , 
                                                map = sample_map , 
                                                initialise = init, 
                                                plot = plot ) )
        fitness = np.array(parallel_eval( fitness_fun, population, sample_pool ))
        #print(fitness)
        behaviour = []
        # for pop in population:
        #     behaviour.append(eval( 'feature_fun' )( pop ))
        behaviour = np.array(parallel_eval( feature_fun, population, sample_pool ))
        # behaviour = np.array( behaviour )
        sampled_points = [ [population[i], fitness[i], behaviour[i]] for i in range( n ) ]
        return( sampled_points )

#######################################################################
################# Initialising ########################################
#######################################################################  


    
    if init_map == None:
        experiment = True
        if mp.map is None:
            mp.map, edges = create_map(mp.feature_resolution, mp.example )

    mymap = init_map
  
    init_n = mp.n_children # init_n : initial population size 

    sampled_points = sample(n = init_n, 
                            sample_map = mymap, 
                            init = True , 
                            fitness_fun = fit_fun , 
                            plot = plot ,
                            sample_pool = pool)

    mymap, improvement_percentage = niche_compete( sampled_points , mymap )

#######################################################################
################### Map-Elites #######################################
#######################################################################
    generation = 1; terminate = False; popsize = mp.n_children

    while terminate != True:
        if experiment:
            mymap = None
        new_samples = sample(   n = popsize, 
                                sample_map = mymap , 
                                fitness_fun = fit_fun , 
                                plot = plot )

        sampled_points.extend(new_samples)
        mymap, improvement_percentage = niche_compete( points = new_samples , map = mymap )
        if verbose:
            print('generation ',generation, ' accepted ',improvement_percentage,' % of points')
        generation += 1
        if generation > number_evals :
            terminate = True

    sampled = [s[0] for s in sampled_points]
    # x1 = [x[0] for x in sampled]
    # x2 = [x[1] for x in sampled]
    # plt.scatter(x1,x2)
    # plt.show()

    
    if experiment:
        return()

    return( mymap )

def parallel_eval(evaluate_function, to_evaluate, pool, parallel = False):
    if parallel == True:
        s_list = pool.map(evaluate_function, to_evaluate)
    else:
        s_list = map(evaluate_function, to_evaluate)
    return list(s_list)

if __name__ == '__main__':
    # prediction_map , _ = create_map( mp.feature_resolution , mp.example)
    # mymap = map_elites(mp.domain , init_map=prediction_map)

    #children = create_children(100 , True , map = prediction_map, plot =True)
    import time
    tic = time.time()    
    for i in range(3):
        mp.map = map_elites(mp.domain, init_map = mp.map, verbose =True)
    print(calculate_final_score(mp.map))
    toc = time.time()
    print(toc-tic)

# %%
