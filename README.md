# NonLinearOscillationsInSlices
Julia scripts to reproduce non-linear oscillation observed in cortical slices stimulated with sinusoidal currents.

In the paper we use two different kinds of models. 
The detailed spiking simulation is implemented with PERSEO avaiable [HERE](https://github.com/mauriziomattia/Perseus.git).
To run the spiking simulation simply download all the *.ini* files to the location of the PERSEO executable and run it form the terminal.

The second is based on the mean-field description of the population activity and rquieres the integration of a Fokker-Planck PDE.
The same network initialization files can be used in Julia using the functions implemented in *FP.jl*. In *FokkerPlanckIntegration.jl* an example is presented that generates the bifurcation diagram described in the paper.

Feel free to contact me at *gianni.vinci@iss.it* for help in reproducing our results.
