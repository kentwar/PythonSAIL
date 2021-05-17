import numpy as np
import setseed as ss
ss.initseed(ss.myseed)

class Domain:

    def __init__(self, 
                featmins, 
                featmaxs, 
                fitness_function=None, 
                feature_function=None,
                valid_ranges = None,
                GP_params = None,
                example_genome = None,
                feature_resolution = None,
                ME_params = None):
        self.mutation_sigma = 0.1
        self.valid_ranges = valid_ranges    #input x ranges [min,max]
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

 
fitness_fun = MY FITNESS FUNCTION
feature_fun = MY FEATURE FUNCTIONS




#####
map = None
sampled_points = None

# example is an initialising point to populate an empty map (it must be 
# valid point in both the fitness and feature functions)
# It is a list [d1,d2 ... , dn]

example = [EXAMPLE POINT IN LIST FORMAT]

# feature_resolution is how many discrete niches in each dimension. Total number
# of niches will therefore be feature_resolution[0]*features_resolution[1]
# example : feature_resolution = [10,10]
feature_resolution = [ , ]

# surrogate GP settings: 

variance = 0.01
lengthscale = 0.5 
noise_var = 1e-5
GP_params = ( noise_var , lengthscale  , variance )
GP_mod = 2 #How often to retrain the parameters

# Experimental parameters
n_add_samples = 10 #How many samples to make each step
pred_map_mod = 10 #How often to make a predictive map
do_intermediate_pred_maps = False #Calculate prediction maps throughout the process
# trumean is used in specific domains where you know the fitness mean
# truemean = 0.3132138832732077 

# Map Elites config
mprob = 0.0 #Probability of performing 1 slice cross mutation
n_children         = 2**6; 
mut_sigma          = 0.1; 
n_gens             = 2**7
ME_params = ( mprob  , n_children  , mut_sigma , n_gens )

# Predictive Map parameters. 
PM_params = ( mprob  , n_children  , mut_sigma , 2**9 )

# input constraints - The valid ranges for each input dimension
constraints = [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1] ]
# Range for the feature functions
featmins = [0, 0]
featmaxs = [1, 1]

domain_flag = 'MY_DOMAIN' 

domain = Domain(featmins = featmins , 
                featmaxs = featmaxs , 
                fitness_function = fitness_fun, 
                feature_function = ( feature_fun ) ,
                valid_ranges = constraints,
                GP_params = GP_params ,
                example_genome = example ,
                feature_resolution = feature_resolution,
                ME_params = ME_params)