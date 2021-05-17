import numpy as np
import torch, gpytorch
from colorama import Fore,Style
import copy , time , os , pickle
import mapelites.createchildren
#import domain.set_domain as sd
from mapelites.nichecompete import nichefinder , edgefinder

import setseed as ss
ss.initseed(ss.myseed)

## This import location is defined by the parent script.
## domain_config exists in a folder selected at runtime.
import domain_config as mp

def build_acquisition_fun( gp , UCB = True , means = False):
    '''build_acquisition_fun() wraps the output from a GPy predictive model into 
    a function that returns an array of acquisition evaluations based on the 
    posterior predictive distribution of the GPs. 

    ##TODO - Implement UCB !!!!!!!!! -Only returns posterior mean at the moment?
    ##

    Example: acquisition_function = build_acquisition_fun( my_gp_model )  

    Inputs (arguments): 
        gp       - [ GPy model ]  - A GPy gp model

    Outputs
        acq_fun  - [ function ]   - Acquisition function

        #TODO - Timing output (other output?)
    

    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 16-Oct-2020 
    '''

    fun = gp.predict
    if means:
        def acquisition_fun( points , give_var = False ):
            #Check if it's a single observation
            xdims = len(mp.domain.valid_ranges)
            if np.shape( points ) == (xdims,):
                point = np.array( points[0] ).reshape( -1 , len( points[0] ) ) 
                evaluation = fun( point )

                fitness, posterior_variance = evaluation[ 0 ] , evaluation[ 1 ]

                if give_var:
                    return( float( fitness ) , float( posterior_variance ) )
                else:
                    return( float( fitness ) )

            else:
            #Process multiple points
                n = len(points)
                points = [np.array(point).reshape( -1 , len(point)) for point in points]
                evaluation = [fun( points[i] ) for i in range(n)]

                fitness = [ float(e[ 0 ] )  for e in evaluation]
                posterior_variance = [ float(e[ 1 ] )  for e in evaluation]
                fitlist = [ float( f ) for f in fitness ]
                varlist = [ float( v ) for v in posterior_variance ]
                if give_var:
                    return( fitlist ,  varlist )
                else:
                    return( np.array(fitlist) )

        return( acquisition_fun )

    if UCB:
        def UCB_acquisition_fun( points , give_var = False ):
            #Check if it's a single observation
            alpha = 1
            beta = 20
            xdims = len(mp.domain.valid_ranges)
            if np.shape( points ) == (xdims,):
                point = np.array( points[0] ).reshape( -1 , len( points[0] ) ) 
                evaluation = fun( point )

                posterior_mean, posterior_variance = evaluation[ 0 ] , evaluation[ 1 ]

                fitness = alpha * posterior_mean + beta * posterior_variance
                
                if give_var:
                    return( float( fitness ) , float( posterior_variance ) )
                else:
                    return( float( fitness ) )

            else:
            #Process multiple points
                n = len(points)
                points = [np.array(point).reshape( -1 , len(point)) for point in points]
                evaluation = [fun( points[i] ) for i in range(n)]

                fitness = [ float( alpha * e[ 0 ] + beta * e[ 1 ] )  for e in evaluation]
                posterior_variance = [ float(e[ 1 ] )  for e in evaluation]
                fitlist = [ float( f ) for f in fitness ]
                varlist = [ float( v ) for v in posterior_variance ]
                if give_var:
                    return( fitlist ,  varlist )
                else:
                    return( np.array(fitlist) )

        return( UCB_acquisition_fun )

def build_pytorch_acq_fun( gp , UCBflag = True , meansflag = False):
    '''build_acquisition_fun() wraps the output from a GPy predictive model into 
    a function that returns an array of acquisition evaluations based on the 
    posterior predictive distribution of the GPs. 

    ##TODO - Implement UCB !!!!!!!!! -Only returns posterior mean at the moment?
    ##

    Example: acquisition_function = build_acquisition_fun( my_gp_model )  

    Inputs (arguments): 
        gp       - [ GPy model ]  - A GPy gp model

    Outputs
        acq_fun  - [ function ]   - Acquisition function

        #TODO - Timing output (other output?)
    

    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 16-Oct-2020 
    '''

    fun = gp
    if meansflag:
        def acquisition_fun( points , give_var = False ):
            #Check if it's a single observation
            xdims = len(mp.domain.valid_ranges)
            if np.shape( points ) == (xdims,):

                point = torch.tensor( points[0] , dtype=torch.float).reshape( -1 , xdims ) 
                evaluation = fun( point )

                fitness, posterior_variance = evaluation.mean 

                return( float( fitness ) ) #*mp.std + mp.smean ) )


            else:
            #Process multiple points
                n = len(points)
                points = np.array([point for point in points])
                points = torch.tensor( points , dtype = torch.float).reshape( -1 , xdims)
                evaluation = [fun( points[i].reshape(-1,xdims) ) for i in range(n)]
                #fitness = [ float( e.mean *mp.std + mp.smean  )  for e in evaluation]
                fitness = [ float( e.mean  )  for e in evaluation]
                posterior_variance = [ float(e.variance )  for e in evaluation]
                fitlist = [ float( f ) for f in fitness ]
                varlist = [ float( v ) for v in posterior_variance ]
                if give_var:
                    return( fitlist ,  varlist )
                else:
                    return( np.array(fitlist) )

        return( acquisition_fun )

    if UCBflag:
        def UCB_acquisition_fun( points , give_var = False ):
            #Check if it's a single observation
            alpha = 1
            beta = 20
            xdims = len(mp.domain.valid_ranges)
            if np.shape( points ) == (xdims,):

                point = torch.tensor( points[0] ,dtype = torch.float).reshape( -1 , len( points[0] ) ) 
                evaluation = fun( point )

                posterior_mean, posterior_variance = evaluation.mean , evaluation.variance

                #fitness = alpha * (posterior_mean*mp.std + mp.smean) + beta * posterior_variance
                fitness = alpha * (posterior_mean) + beta * posterior_variance
                if give_var:
                    return( float( fitness ) , float( posterior_variance ) )
                else:
                    return( float( fitness ) )

            else:
            #Process multiple points
                n = len(points)
                points = np.array([point for point in points])
                points = torch.tensor( points , dtype = torch.float).reshape( -1 , xdims)
                evaluation = [fun( points[i].reshape(-1,xdims) ) for i in range(n)]

                #fitness = [ float( alpha * (e.mean *mp.std + mp.smean) + beta * e.variance )  for e in evaluation]
                fitness = [ float( alpha * (e.mean ) + beta * e.variance )  for e in evaluation]
                posterior_variance = [ float(e.variance )  for e in evaluation]
                fitlist = [ float( f ) for f in fitness ]
                varlist = [ float( v ) for v in posterior_variance ]
                if give_var:
                    return( fitlist ,  varlist )
                else:
                    return( np.array(fitlist) )

        return( UCB_acquisition_fun )

def build_model(    samples , variance = 0.01 , lengthscale = 0.5 , 
                    noise_var = 1e-5 , update = True ):
    '''build_model() wraps GPy model building and does the data parsing.  

    Example: my_gp_model = build_model( my_observation )  

    Inputs (arguments): 
        samples             - N*[[x],[y],[f]]   - observations of QD data.
        variance            - [ Float ]         - Prior for the GP variance
        lengthscale         - [ Float ]         - Prior for the GP lengthscale
        noise_var           - [ Floar ]         - Prior for voise variance
        update              - [ Boolean ]       - Flag to retrain hyper-parameters

    Outputs
        model               - [ GPy model ]     - A Posterior Gaussian Process
        kernel              - [ GPy kernel ]    - A Posterior Kernel

        #TODO - Timing output (other output?)
    

    Code Author: Paul Kent 
    Warwick University
    email: paul.kent@warwick.ac.uk
    Oct 2020; Last revision: 16-Oct-2020 
    '''
    train_x = [ s[ 0 ] for s in samples ]
    train_y = [ s[ 1 ] for s in samples ]

    try:
        input_dims = len( train_x[ 0 ] )
    except:
        input_dims = 1
    try:
        output_dims = len( train_y[ 0 ] )
    except:
        output_dims = 1

    kernel = GPy.kern.RBF( input_dims , variance , lengthscale )

    #Data reshaping for GPy
    x = np.array( train_x ).reshape( -1 , input_dims )
    y = np.array( train_y ).reshape( -1 , output_dims )
    model = GPy.models.GPRegression( x , y , noise_var = noise_var , kernel = kernel)
    ##TODO Check if these still work as they are domain dependent
    #kernel.lengthscale.constrain_bounded( 0.001 , 2 )
    #model.Gaussian_noise.variance.constrain_bounded( 0 , 1e-5 )
    
    # if False allows for hyperparameters to be retained from previous run
    if update:
        model.optimize_restarts( num_restarts = 100, verbose = False )
    
    return( model , kernel )

# def build_pytorch_model(    samples , variance = None , lengthscale = None , 
#                     noise_var = None , update = True ):
#     '''build_model() wraps GPy model building and does the data parsing.  

#     Example: my_gp_model = build_model( my_observation )  

#     Inputs (arguments): 
#         samples             - N*[[x],[y],[f]]   - observations of QD data.
#         variance            - [ Float ]         - Prior for the GP variance
#         lengthscale         - [ Float ]         - Prior for the GP lengthscale
#         noise_var           - [ Floar ]         - Prior for voise variance
#         update              - [ Boolean ]       - Flag to retrain hyper-parameters

#     Outputs
#         model               - [ GPy model ]     - A Posterior Gaussian Process
#         kernel              - [ GPy kernel ]    - A Posterior Kernel

#         #TODO - Timing output (other output?)
    

#     Code Author: Paul Kent 
#     Warwick University
#     email: paul.kent@warwick.ac.uk
#     Oct 2020; Last revision: 16-Oct-2020 
#     '''
#     x = [ s[ 0 ] for s in samples ]
#     y = [ float(s[ 1 ]) for s in samples ]
#     print(train_x.shape[-1])
#     train_x = torch.tensor( x , dtype=torch.double).reshape(-1,len(samples[0][0]))
#     train_y = torch.tensor( y , dtype=torch.double)

#     # initialize likelihood and model
#     likelihood = gpytorch.likelihoods.GaussianLikelihood()
#     model = ExactGPModel(train_x, train_y, likelihood)
#     model.covar_module = gpytorch.kernels.ScaleKernel(
#         gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[-1]))
    

#     # Find optimal model hyperparameters
#     model.train()
#     likelihood.train()

#     # Use the adam optimizer
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

#     # "Loss" for GPs - the marginal log likelihood
#     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

#     training_iter = 100

#     for i in range(training_iter):
#         # Zero gradients from previous iteration
#         optimizer.zero_grad()
#         # Output from model
#         output = model(train_x)
#         # Calc loss and backprop gradients
#         loss = -mll(output, train_y)
#         loss.backward()
#         print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#             i + 1, training_iter, loss.item(),
#             model.covar_module.base_kernel.lengthscale.item(),
#             model.likelihood.noise.item()
#         ))
#         optimizer.step()
#     model.eval()
#     #likelihood.train()
#     return( model )

def buildpymodel(observations, n_initial_samples, noise_var = None, lengthscale = None, scale = None, retrain = True):
    import math
    import torch
    import gpytorch
    

    def trainmodel(observations, n_initial_samples, iter, min_noise, noise_var = None, lengthscale = None, scale = None, retrain = True  ):
        tic = time.perf_counter()
        x_train = np.array( [ obs[ 0 ] for obs in observations ] , dtype=np.float32)
        y_train = np.array( [ obs[ 1 ] for obs in observations ] , dtype=np.float32)
        x_pytrain = torch.from_numpy(x_train)
        y_pytrain = torch.from_numpy(y_train)
        x_pytrain = x_pytrain.float()
        y_pytrain = y_pytrain.float()

        ## Save mean and std
        # sample_mean = y_pytrain[:n_initial_samples].mean()
        # if mp.truemean:
        #     mp.smean = mp.truemean
        # else:
        #     mp.smean = sample_mean
        # sample_sd = y_pytrain[:n_initial_samples].std()
        # mp.std = sample_sd
        # Normalization 
        # y_pytrain = (y_pytrain - sample_mean)/(sample_sd)
        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(min_noise))
        model = ExactGPModel(x_pytrain, y_pytrain, likelihood)
        model.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=len(x_pytrain[-1])))
        # Initialize with previous hypers
        if type(lengthscale) == torch.Tensor:
            hypers = {
                'likelihood.noise_covar.noise': noise_var,
                'covar_module.base_kernel.lengthscale': lengthscale +  torch.tensor(np.random.standard_normal(len(xtrain[0]))*iter),
                'covar_module.outputscale': scale +  torch.tensor(np.random.standard_normal(1)*iter),
            }
            model.initialize(**hypers)

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()
        # Use the adam optimizer
        # Includes GaussianLikelihood parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model) 
        training_iter = 100
        for i in range(training_iter):

            # Zero gradients from previous iteration
            optimizer.zero_grad()

            # Output from model
            output = model(x_pytrain) 
            # Calc loss and backprop gradients
            loss = -mll(output, y_pytrain)
            loss.backward()
            
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            #     i + 1, training_iter, loss.item(),
            #     model.covar_module.base_kernel.lengthscale.item(),
            #     model.likelihood.noise.item()
            # ))
            optimizer.step()

        # Save hypers
        noise = model.likelihood.noise_covar.noise.item()
        lengthscale = model.covar_module.base_kernel.lengthscale
        scale = model.covar_module.outputscale.item()
        hypers = (noise, lengthscale, scale)
        model.eval()
        likelihood.eval()
        toc = time.perf_counter()
        done = True
        return( model, toc-tic , hypers, done )

    done = False
    iter = 0
    min_noise = 1e-6
    while done == False:
        try:
            model, timer , hypers , done = trainmodel(observations, n_initial_samples, iter, min_noise, noise_var = None, lengthscale = None, scale = None, retrain = True  )

        except:
            iter += 0.1
            min_noise += 1e-6
            print('The GP hyper-parameters failed to converge, retrying.')
            pass

    return( model, timer , hypers )


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def count_niches_filled():
    '''looks at the solution map and reports how many cells have an elite
    '''
    count = np.sum( ~np.isnan(mp.map.fitness.flatten()) )
    return(count)

def updatemapSAIL( observations ):
    count = 0 
    for point in observations:
        index = tuple( nichefinder( point[ 2 ], mp.map , mp.domain ) )
        if np.isnan( mp.map.fitness[ index ] ):
            mp.map.genomes[ index ] = point[ 0 ]
            mp.map.fitness[ index ] = point[ 1 ]
            count += 1
        else:
            if observations[ -1 ][ 1 ] > mp.map.fitness[ index ]:
                mp.map.genomes[ index ] = point[ 0 ]
                mp.map.fitness[ index ] = point[ 1 ]  
                count += 1
    if len(observations)==0:
        return(0)
    else:
        return( count / len(observations) )

def initialisemap( observations ):
    for point in observations:
        index = tuple( nichefinder( point[ 2 ], mp.map , mp.domain ) )
        if np.isnan( mp.map.fitness[ index ] ):
            mp.map.genomes[ index ] = point[ 0 ]
            mp.map.fitness[ index ] = point[ 1 ]
        else:
            if observations[ -1 ][ 1 ] > mp.map.fitness[ index ]:
                mp.map.genomes[ index ] = point[ 0 ]
                mp.map.fitness[ index ] = point[ 1 ]  

def introprint(max_evals, n_initial_samples):
        print(f'{Fore.GREEN}-{Style.RESET_ALL}'*40)
        print(f'{Fore.GREEN}pySAIL - Created by Gaier et. al, implemented by Paul Kent{Style.RESET_ALL}')
        print(f'{Fore.GREEN}max budget = {Style.RESET_ALL}' , max_evals)
        print(f'{Fore.GREEN}Initial Sample Size = {Style.RESET_ALL}' , n_initial_samples)
        print(f'{Fore.GREEN}-{Style.RESET_ALL}'*40)

def save_data(  map_list,
                observations,
                hps , 
                eval_time , 
                il_time , 
                trn_time, 
                perc_imp, 
                fit_val, 
                pred_maps,
                pred_map_val , 
                niches_filled ,
                seed,
                mydir_ ):

    data_collect = [['hyper parameters' , hps ] , 
                    [ 'eval times' , eval_time ] ,
                    ['illumination times' , il_time ], 
                    ['GP training times ' , trn_time ] , 
                    ['Percent Improvement ' , perc_imp ] , 
                    ['Fitness Value' , fit_val ] ,
                    ['pred map Value' , pred_map_val ] ,
                    ['Niches Filled' ,niches_filled ],
                    ['map fitness' , mp.map.fitness ] ,
                    ['Final Map' , mp.map.genomes] ]

 
    binary_data = { 'points':observations,
                    'map_list':map_list,
                    'hyper parameters':hps, 
                    'eval times':eval_time,
                    'illumination times':il_time, 
                    'gp training times ':trn_time, 
                    'percent Improvement ':perc_imp, 
                    'fitness Value':fit_val,
                    'pred map Value':pred_map_val,
                    'niches Filled':niches_filled,
                    'map fitness': mp.map.fitness,
                    'final Map':mp.map.genomes ,
                    'pred_maps':pred_maps}

    filehandler = open(mydir_+'/BDC.pickle', 'wb') 
    pickle.dump(binary_data, filehandler)

    np.savetxt(mydir_+'/DC.csv',  
            data_collect, 
            delimiter =", ",  
            fmt ='% s') 

def save_map(map, mydir_, filename):
    binary_data = map
    filehandler = open(mydir_+'/' + filename + '.pickle', 'wb')      
    pickle.dump(binary_data, filehandler)

def save_pred_map(pred_map, mydir_ , n):
    binary_data = pred_map.genomes
    filehandler = open(mydir_+'/' + 'n='+str(n)+'_pred_genomes'  + '.pickle', 'wb')     
    pickle.dump(binary_data, filehandler)

    binary_data = pred_map.fitness
    filehandler = open(mydir_+'/' + 'n='+str(n)+'_pred_fitness'  + '.pickle', 'wb')     
    pickle.dump(binary_data, filehandler)

def readme(dir_ , problem, n_iter, max_iter, seed):
    with open(dir_ + '/readme.txt', 'w') as f:
        f.write('SAIL experiment readme\n' +
        'Problem Function : '+ problem +'\n' +
        'Starting n : ' + str(n_iter) +'\n' +
        'Number of iterations :' + str(max_iter) + '\n' +
        'Start time : ' + str(time.strftime('%d-%m_%H:%M:%S')) +'\n' +
        'Random Seed : ' + str(seed) + '\n' +
        'niches: ' +str(mp.feature_resolution) + '\n'       )

def savepoints(map, dir_ ):
    ''' Saves the points that have been visited in binary format
    '''
    filehandler = open(dir_+'/'+ 'points.pickle', 'wb') 
    pickle.dump(map, filehandler)