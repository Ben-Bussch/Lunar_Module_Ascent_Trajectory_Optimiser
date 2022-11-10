# Lunar_Module_Ascent_Optimiser

This program was developed to optimise the trajectory of a rocket taking off from the moon, as to minimise fuel costs. It makes use of GEKKO's ipopt NLP solver to numerically optimise the trajectory the rocket must take to reach orbit around the moon. 

The numbers and parameters used are from the Apollo 11 mission, as to be able to compare the program results to real data, to check the validity of the model. Results from the code indicate the launch time of the LM from the lunar surface up to orbit should be around 434 seconds, which practically mirrors the real launch duration of the mission, given at 435 seconds according to NASA's offical website (https://www.nasa.gov/mission_pages/apollo/missions/apollo11.html). Further information about sources for all the data, along with derivations of the governing equations and system dynamics, is given in the Lunar_Module_Trajectory_Optimisation.pdf file attached on the main page. 

The main branch of the code has been majorly updated since the creation of this document, so there are some  discontinuities between the actual code and what the document provides, but the original code is appendixed within the document itself if this could provide useful insight. The document was written as a school assessment, so apologies for the unconventional formating. The major updates in the code include a limit in angular acceleration of the spacecraft (which is currently a guess as I could not find data about it), as well as modeling the launch into an eliptical orbit (which the real apollo 11 ascent module did) as opposed to the simplified circular orbit the code use to model.


