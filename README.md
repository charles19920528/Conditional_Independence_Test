# Conditional_Independence_Test
The repository contains code and data for a neural conditional independence method which is still under development.

generate_train_fucntions.py contains code for data generating and model fitting. 

data_generation.py generates the simulation data and save them under the data directory.

simulation_functions.py contains helper functions for simulations.

simulation.py runs simulation on data under the data directory and save the raw results under the results directory.

result_analysis_functions.py contains helper functions for analysis of the raw results produced by the simulation.py.

result_analysis.py analyzes the raw results produced by the simulation.py and save the analyzed results under the the results directory.

ising_tuning_functions.py contains helper functions for ising_tuning.py.

ising_tuning.py performs hyperparameter tuning for the nerual networks.

Other scripts are still under development. 
