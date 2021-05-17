import numpy as np
import random
import sys, os
import setseed as ss
import argparse

## When Taking arguments from the command line.
## --seed sets the random seed
## --UCB_Param sets the UCB exploration parameter.
try:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default = 8 , type=int)
    parser.add_argument('--UCB_param', default = 3.7 , type=float)
    parser.add_argument('--max', default = 1000 , type=int)
    parser.add_argument('--init', default = 1000 , type=int)
    args = parser.parse_args()
    seed = args.seed
    UCB_param = args.UCB_param
    max = args.max 
    init_n = args.init
except: 
    seed = 8
    UCB_param = 3.7
    max = 110
    init_n = 100

ss.myseed = seed
ss.initseed(ss.myseed)
cur_dir = os.path.dirname(os.path.realpath(__file__))

#################################################    
###### Set the problem config file to load from 
#################################################

domain_flag = 'parsec' ##Change this to change your domain
#######

if domain_flag == 'rastrigin':
    sys.path.insert(0, cur_dir+'/domain/Rastrigin_config')
if domain_flag == 'parsec':
    sys.path.insert(0, cur_dir+'/domain/Parsec_config')

################################################

import domain_config as mp

## This sets the UCB exploration parameter, tests show that SAIL is sensitive
## to this value. In experiments on Parsec/0 problem (a relatively smooth surface)
## the optimal parameter setting value decreased with niche granularity.

try:
    mp.beta = args.UCB_param
except:
    mp.beta = UCB_param
import GPy , copy , time , os , pickle

from mapelites.mapelites import map_elites

from createmap import create_map
from mapelites.nichecompete import niche_compete , nichefinder
from initial_sampling import initial_sampling , additional_sampling , sobol_sample
from colorama import Fore,Style
from create_prediction_map import create_prediction_map
from sail_lib import *

def sail( max_evals = 110, preload = None , n_initial_samples = 100 ):
    '''sail.py performs the Surrogate Assisted Illumination algorithm on a domain 
    defined in the mapelites.map_elites_config file.

    Example: output_predictions = sail(100)  

    Inputs (arguments): 
        max_evals           - [ Integer ] - Evaluation budget 
        preload             - [ Array   ] - Preloaded evaluations
        n_initial_samples   - [ Integer ] - Number of samples to initialise

    Inputs (from config file): 
        mp.domain           - [ Domain Class ]   - Domain specific information
        mp.map              - [ Map Class ]      - Global map (for map-elites)

    Outputs
        Predict_map_list    - [ List of Maps ]   - All illuminated maps throughout
                                                 optimisation process.
    

    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 14-Oct-2020 
    '''    
    ## Pretty Stuff
    introprint(max_evals, n_initial_samples)

    ## Data Collection stuff
    DC_predict_map_list = [] 
    DC_hyper_parameters = [] 
    DC_map_list = [] 
    DC_percent_improvement = []
    DC_training_times = [] 
    DC_illumination_times = []
    DC_evaluation_times = []
    DC_fitness_value = []
    DC_niches_filled = []
    DC_pred_map_val = []
    param_counter = 0; param_flag = False
    pred_map_counter = 0; pred_map_flag = False
    

    ## Data File.
    datestamp = time.strftime('%d-%m %H:%M:%S')
    mydir = './Data_collection/SAIL'+str(seed)+domain_flag +' '+datestamp+ 'UCBP'+ str(mp.beta)
    os.mkdir(mydir)
    readme(mydir, domain_flag, n_initial_samples, max_evals, seed)

    n_add_samples = mp.n_add_samples
    hypers = mp.GP_params  
    
    #1. Produce initial samples ###############################################

    if not preload:
        print( f'{Fore.YELLOW}Performing{Style.RESET_ALL} initial sampling procedure' )
        print( 'Samples required: ',n_initial_samples )
        initial_samples = initial_sampling( n_initial_samples )


    else:
        print( f'{Fore.YELLOW}Importing{Style.RESET_ALL} initial samples' )
        initial_samples = preload

    n_samples = n_initial_samples ; observations = initial_samples   
    # Pretty Line to visually seperate output
    printline()

    ## Initialise main map  
    mp.map , edges = create_map( mp.feature_resolution , mp.domain)
    initialisemap( initial_samples )
    DC_map_list.append([mp.map.genomes,mp.map.fitness])
    DC_niches_filled.append( count_niches_filled( ) )

    save_map([mp.map.genomes], mydir , 'initial_map' )

    print('initial point value:' , np.nansum(mp.map.fitness.flatten()))
    #2. Main Acquisition Loop ##################################################
    while n_samples < max_evals:

        # 2.1 Create Surrogate and Acquisition Function ########################
        # Surrogate models are created from all evaluated samples, and these
        # models are used to produce an acquisition function.
        print(f'PE ' , n_samples ,' | Training Surrogate Models')

        ## Only relearn hyperparameters every training_mod iterations
        if (n_samples == n_initial_samples or param_flag == True):
            fit_gp , timer, hypers  = buildpymodel(  observations , n_initial_samples, *hypers)
            param_flag = False
        else:
            # train GP but inherit hyperparameters
            fit_gp ,timer, hypers = buildpymodel(  observations , n_initial_samples, *hypers , retrain = False)

        DC_training_times.append( timer)
        DC_hyper_parameters.append(( hypers ))
        
        ### Build The UCB acquisition Function
        acq_fun = build_pytorch_acq_fun( fit_gp , UCBflag = True, meansflag = False )

        
        # 2.2 Create Prediction map for data capture ###########################
        
        if pred_map_flag and mp.do_intermediate_pred_maps:
            print(f'PE ' , n_samples ,' | Illuminating Prediction Map')
            pred_map = create_prediction_map( fit_gp , observations )
            DC_predict_map_list.append( (pred_map.genomes,pred_map.fitness) )
            pred_map_flag = False
            predval = calculate_final_score(pred_map)
            DC_pred_map_val.append( predval )
            print( f'{Fore.YELLOW}Current prediction map value:{Style.RESET_ALL}', predval )
            save_map(pred_map, mydir , n_samples + 'Pred' )
  
        # 2.3 Create Acquisition map ###########################################

        tic = time.perf_counter()
        
        if n_samples < max_evals:
            print(f'PE ' , n_samples ,' | Illuminating Acquisition Map ')
            init_acq_map = copy.deepcopy(mp.map)
            acq_map = map_elites(   mp.domain , 
                                    init_map = init_acq_map , 
                                    fit_fun = acq_fun , 
                                    feat_fun = mp.feature_fun,
                                    plot = False ,
                                    me_params = mp.ME_params)                
        print( f'{Fore.YELLOW}Current solution value:{Style.RESET_ALL}', np.nansum( mp.map.fitness ) )
        toc = time.perf_counter()
        DC_illumination_times.append( toc - tic )

        # 2.4 Infill samples ###################################################       
        # Use sobol sampling to randomly choose acquisition elites to evaluate          
        tic = time.perf_counter()
        
        if n_samples == n_initial_samples:
            # Create an initial Sobol set.
            feat_sobol = sobol_sample( max_evals*50 , mp.domain.valid_feat_ranges )      
            sobol_point = 0

        print(f'PE ' , n_samples ,' | Sampling New Points ')        
        new_samples , sobol_point , n_new_samples = additional_sampling( 
            n_add_samples , 
            feat_sobol , 
            sobol_point ,
            acq_map )
        
        # 
        observations += new_samples

        savepoints(observations , mydir ) #Save current observations
        
        # Update master map and collect improvement data. 

        percent_improvement = updatemapSAIL( new_samples )
        print( len(new_samples) , ' unique points evaluated')
        print( percent_improvement*100 , '% were added to the map')

        toc = time.perf_counter()
        DC_evaluation_times.append( toc - tic )
        DC_percent_improvement.append( percent_improvement )
        knownval = np.nansum( mp.map.fitness )
        DC_fitness_value.append( knownval )
        DC_niches_filled.append( count_niches_filled( ) )
        
        print( f'{Fore.YELLOW}Current solution value:{Style.RESET_ALL}', knownval )
        
        n_samples += n_new_samples
        param_counter += n_new_samples
        if param_counter >= mp.GP_mod*n_add_samples:
            param_counter -= mp.GP_mod*n_add_samples
            param_flag = True

        pred_map_counter += n_new_samples
        if pred_map_counter >= mp.pred_map_mod*n_add_samples:
            pred_map_counter -= mp.pred_map_mod*n_add_samples
            pred_map_flag = True

        DC_map_list.append([mp.map.genomes,mp.map.fitness])

 
        printline()

    # 3 - Return final prediction map
    print(f'\n Illuminating Final Prediction Map')
    pred_map = create_prediction_map( fit_gp , observations )
    save_pred_map(pred_map , mydir , n_samples)

    DC_predict_map_list.append((pred_map.genomes,pred_map.fitness))
    pred_map_value = calculate_final_score(pred_map)
    DC_pred_map_val.append(pred_map_value)
    save_data(  DC_map_list,
                observations,
                DC_hyper_parameters ,
                DC_evaluation_times ,
                DC_illumination_times ,
                DC_training_times ,
                DC_percent_improvement ,
                DC_fitness_value ,
                DC_predict_map_list ,
                DC_pred_map_val ,
                DC_niches_filled ,
                seed,
                mydir)
    print('Final score : ' + str( np.nansum( mp.map.fitness ) ) )
    print('Final pred_map score : ' + str( pred_map_value ) )
    return( DC_predict_map_list, pred_map_value )          



#TODO Parallelize 


def printline():
    print(f'{Fore.GREEN}-{Style.RESET_ALL}'*40)
    return()

def calculate_final_score(mymap):
    '''This function takes the predictive map and calculates the predicted score
    by assessing their value on the real functions.
    '''
    xdims = len(mp.domain.valid_ranges)
    genomes = mymap.genomes[ ~np.isnan( mymap.fitness ) ].flatten( )
    truevals = [ mp.fitness_fun( x ) for x in np.reshape( genomes, [ -1,xdims ] ) ]
    return( np.nansum( truevals ) )

if __name__ == '__main__':
    
    mypred, fitness = sail( max, n_initial_samples = init_n )
# %%



