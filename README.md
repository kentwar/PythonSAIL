# pySAIL vs BOP-Elites
a working repository for a port of the SAIL algorithm by Adam Gaier
 
original MATLAB : https://github.com/agaier/sail_ecj2018


#### Description
Surrogate Assisted Illumination 'SAIL' is a QD search algorithm utilising the power of Gaussian Process predictive modelling and the efficacy of the map-elites algorithm to search for high performing (in the fitness evaluation) points from a diverse (in the feature space) set of points. 

BOP-Elites is a fully Bayesian Optimisation approach to the quality diversity problem, using an adapted version of the popular Expected Improvement algorithm

#### How to use:
More to come

## Tweaking pySAIL
You can change domain specific information for this algorithm by editing config files in the domain folder, or add your own. Config files should include
* Feature functions
* Fitness Functions
* Number of boundaries for each feature dimension
* constraints for fit/feat functions
When making a new domain you will need to create a new config file and add a flag to all scripts that call on the domain information. 

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
Please report any issues you find.
