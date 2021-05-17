import numpy as np
from mapelites.createmap import create_map
from mapelites.nichecompete import niche_compete

## This import location is defined by the parent script.
## domain_config exists in a folder selected at runtime.
import domain_config as mp

from mapelites.mapelites import map_elites
from sail_lib import *

import setseed as ss
ss.initseed(ss.myseed)

def create_prediction_map( model , observed , UCB = False , means = True, *args , ):
    '''create_prediction_map creates a prediction map by using the map made 
    from all the current best points and then using map-elites to illuminate it
    using the current acquisition function.

    Example: prediction_map = create_prediction_map( gp , points)

    Input:
        model    - [ GPy model ]                - the current posterior GP
        observed - [ n*[ [ x ],[ y ],[ f ] ] ]  - all evaluated points
        *args    - extra args

    Output:
    prediction_map  -   [ Map Class ]   - The best map with current data.

    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 14-Oct-2020 
    '''
    fdims = len(mp.feature_resolution)
    mins = mp.domain.feat_mins
    maxs = mp.domain.feat_maxs 
    xdims = len(mp.example)
    #Make acquisition
    acq_fun  = build_pytorch_acq_fun( model , UCBflag = UCB, meansflag = means )
    #seed map with precise evaluations
    prediction_map , _ = create_map( mp.feature_resolution , mp.domain)
    prediction_map , _ = niche_compete( points = observed , map = prediction_map , domain = mp.domain)
    prediction_map = map_elites( mp.domain, init_map = prediction_map , feat_fun = mp.feature_fun , fit_fun = acq_fun , plot=False ,me_params=mp.PM_params)
    return(prediction_map)



