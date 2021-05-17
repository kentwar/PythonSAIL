# pySAIL
a working repository for a port of the SAIL algorithm by Adam Gaier
 
original MATLAB : https://github.com/agaier/sail_ecj2018


#### Description
Surrogate Assisted Illumination 'SAIL' is a QD search algorithm utilising the power of Gaussian Process predictive modelling and the efficacy of the map-elites algorithm to search for high performing (in the fitness evaluation) points from a diverse (in the feature space) set of points. 


# How to use:

## Install dependencies
from the terminal type :

```
pip install -r requirements.txt
```

## Running for the first time. 
you can run a test on a 2d rastrigin problem with  
```
python sail.py --init 100 --max 110
```
sail.py takes the following from the terminal. 

**init** = initial sampling budget, 
**seed** = random seed
**max** = maximum sampling budget
**init** = initial random sampling (sobol) budget
**UCB_Param** = the UCB parameter used during SAIL.

## Adding a Domain
You can change domain specific information for this algorithm by editing config files in the domain folder, or add your own. Config files should include
* Feature functions
* Fitness Functions
* Number of boundaries for each feature dimension
* constraints for fit/feat functions
When making a new domain you will need to create a new config file and add a flag to all scripts that call on the domain information. (See the first few lines of sail script)

## Return values
The main function sail, in sail.py returns a prediction_map and a fitness value. It may be useful to ignore the prediction maps, and just harvest the fitness
```
_ , fitness = sail( my arguments )
```

## Data capture
The sail script keeps a running pickle of all observed points in a time-stamped folder. This can be used to analyze the quality of the results from the sail algorithm. In addition to this, a human readable file called DC, a pickle file of all data and an iteratively updated map are saved.

The main pickle file BDC can be harvested with the following script contains the following data:
```
directory = 'Data_collection/MY DATA FOLDER'
import pickle 
filehandler = open(directory, 'rb') 
data = pickle.load(filehandler)
hypers  = data['hyper parameters']
eval_t  = data['eval times']
ill_t   = data['illumination times']
gp_t    = data['gp training times ']
per_imp = data['percent Improvement ']
fit_val = data['fitness Value']
pred_val= data['pred map Value']
niches_filled = data['niches Filled']
fit_map = data['map fitness']
final_genomes = data['final Map']
pred_maps = data['pred_maps']
maps = data['map_list']
points = data['points']
```
## Requirements
This project was created with Python 3.7 on a linux machine (Ubuntu 18.04.4 LTS)
The following modules are used:
* Numpy 1.19.1
* GPytorch 1.0.1
* PYtorch 1.4
* colorama 0.4.3
* matplotlib 3.3.1
* matplotlib-base 3.3.1
* itertools 

# Known issues
The parsec domain requires some setting up and may not work out of the box
Please report any issues you find.
