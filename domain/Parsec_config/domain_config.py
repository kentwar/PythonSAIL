import numpy as np

import setseed as ss
ss.initseed(ss.myseed)



from domain.pyAFT.ffdFoil import *
from domain.pyAFT.foil_eval import *

import domain.pyAFT.parsec as prs



class Domain:

    def __init__(self, 
                featmins, 
                featmaxs, 
                fitness_function=None, 
                feature_function=None,
                GP_params = None,
                example_genome = None,
                feature_resolution = None,
                ME_params = None):
        self.mutation_sigma = 0.1
        self.valid_ranges =   [[ 0 , 1 ] for i in range(len(example))]    #input x ranges [min,max]
        self.fit_fun = fitness_function
        self.feat_fun = feature_function
        self.feat_dims = len( featmins )
        self.feat_mins = featmins
        self.feat_maxs = featmaxs
        self.valid_feat_ranges = [ [featmins[ i ], featmaxs[ i ] ] for i in range(len( featmins ))] 
        self.GP_Params = GP_params
        self.example_genome = example_genome
        self.feature_resolution = feature_resolution
        self.ME_params = ME_params

# n_points = 10
# naca0012_ffd = ffdFoil( n_points , base='domain/pyAFT/naca0012.csv', defRange = 1 )

def ffd_fit( ):
    foil = prs.Airfoil()

    #naca0012_ffd = ffdFoil( n_points , base='domain/pyAFT/naca0012.csv', defRange = 1 )
    # Evaluate base foil
    base_coord = foil.express()

    basefit , basefeats = evalFoil(base_coord )
    baselift , basearea = basefeats

    def ffd_true_fit( params , verbose = False):
        new_coords = foil.express( params = params )
        
        # Evaluate deformed foil
        fit , feats = evalFoil( new_coords )
        lift , area = feats   
       
        if lift < baselift:
            lift_penalty = (lift/baselift)**2
        else: lift_penalty = 1

        if area > basearea:
            area_penalty = ( 1 - ( area - basearea )/basearea )**7
        else: area_penalty = 1
        
        true_fitness = fit * lift_penalty * area_penalty 

        if verbose:
            print('base lift ', baselift )
            print('base area ', basearea ,'\n')
            print( 'new lift  ' , lift)
            print( 'new area  ' , area , '\n')
            print( 'lift pen =' , lift_penalty)
            print( 'areapen =' , area_penalty)
            print( '-log(cd) = ' , fit , '\n')
            print( 'final fitness = ' , true_fitness )

        return( true_fitness )
    
    def true_fit_runner( genomes , verbose = False ):
        try:
            return( ffd_true_fit(genomes , verbose) )
        except:
            return_val = [ ffd_true_fit(g) for g in genomes ]
            return( return_val )

    return( true_fit_runner )



fitness_fun = ffd_fit()


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


#####

map = None
sampled_points = None


# example is an initialising point to populate an empty map (if this is 
# non-observable the whole algorithm will break)
example = [0.115455731631420,0.769456784342656,0.139077011276969,0.278798623701840,0.124359615964012,0.151578562448986,0.751881051352209,0.307595979711475,0.262943052636945,0.762983847665074]

# feature_resolution is how many discrete niches in each dimension. Total number
# of niches will therefore be feature_resolution[0]*features_resolution[1]
feature_resolution = [10,10]

# surrogate GP settings: 
variance = 0.01
lengthscale = 0.5 
noise_var = 1e-5
GP_params = ( noise_var , lengthscale  , variance )
GP_mod = 2
smean = 0
std = 1

# Experimental parameters
n_add_samples = 10
pred_map_mod = 10 
do_intermediate_pred_maps = False #Calculate prediction maps throughout the process
truemean = 0.3132138832732077

# Map Elites config
mprob = 0.0 #Probability of performing 1 slice cross mutation
n_children         = 2**6; 
mut_sigma          = 0.1; 
n_gens             = 2**7
ME_params = ( mprob  , n_children  , mut_sigma , n_gens )
PM_params = ( mprob  , n_children  , mut_sigma , 2**8 )

# BOP_elites parameters
penalty = 0

# input constraints
constraints = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1] ]

domain_flag = 'parsec'
domain = Domain(featmins = [ 0 ] *2 , 
                featmaxs = [ 1 ] *2 , 
                fitness_function = fitness_fun, 
                feature_function = ( feature_fun ) ,
                GP_params = GP_params ,
                example_genome = example ,
                feature_resolution = feature_resolution,
                ME_params = ME_params)