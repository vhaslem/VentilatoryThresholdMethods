This is the simulation code used to test the error from the generated breakpoint that each method -- strucchange (rewritten in Python, but initially an R package), Computational VSlope, and our new method -- predicts given that the simulated data is generated using one of the models and cross tested against the others.

NOTE 1: Strucchange is an R Package used in Aida et al. (2022) to find inflection points of ventilation parameters, and tests against "laboratory" VSlope, meaning estimates from the researchers and not the computed values.
I rewrote Strucchange in Python (but only allowed for a maximum of one breakpoint.) In order to publish, we must rewrite the code in R to use the proper Strucchange package and allow for more breakpoints.

NOTE 2: Data was generated pseudo-normally, with data generation restricted to positive values to prevent negative logarithm issues in simulation.

NOTE 3: The following arguments may be modified:
$N$ - number of rounds of simulation
$n$ - number of points generated
$\sigma$ - standard deviation from generated model
$\alpha$ - statistical significance threshold.

Initial findings show for low $\sigma$, Hodgin-Haslem produces a wider band of predictions that do encompass the true value, while VSlope underpredicts and strucchange stays accurate. However, as $\sigma$ is increased, both VSlope and strucchange begin to diverge from the breakpoint while Hodgin-Haslem stays fairly accurate.
