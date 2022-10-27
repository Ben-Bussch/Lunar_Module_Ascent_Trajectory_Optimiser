# Lunar_Module_Ascent_Optimiser

This program was produced to optimise the trajectory of a rocket taking off from the moon, as to minimise fuel costs. It makes use of GEKKO's ipopt NLP solver to numerically optimise the trajectory the rocket must take to reach orbit around the moon. 

The numbers and parameters used are from the Apollo 11 mission, as to be able to compare the program results to real data, to check the validity of the model. Results from the code indicate the launch time of the LM from the lunar surface up to orbit should be around 435.298 seconds, which mirrors the real launch duration of the mission, given at 435 seconds according to NASA's offical website (https://www.nasa.gov/mission_pages/apollo/missions/apollo11.html). Further information about sources for all the data, along with derivations of the governing equations and system dynamics, is given in the Lunar_Module_Trajectory_Optimisation.pdf file attached on the main page. The document was written as a school assessment, so apologies for the unconventional formating.

I intended to do further work on the code to clean up some of the redundant variables that currently exist, as well as rewrite the ploting code to make the program more concise and user friendly.
