Code used in "Finding the Breakpoint: A Novel Maximum Derivative Difference Method to Detect Ventilatory Threshold and a Comparison to Current R-Based Algorithms"



##### FOR RESEARCH USE

###### R/segmented_method.R
Gold standard computational method found in study using segmented package. Contains guided steps to use on lab data

###### R/mdd_method.R
Comparable performance to segmented method but slightly less robust. Contains guided steps to use on lab data

###### R/strucchange_method.R
Findings indicate this method is robust in simulations but subpar in lab data, which disagrees with the findings in Aida et al. (2022), which used multiple breakpoints and a smoothing function on the data. We found even smoothed data didn't perform at the same level as MDD or Segmented Method, but could still be useful when comparing VT1 estimates and for further study. Contains guided steps to use on lab data

###### R/simulation_and_method_analysis.R
Simulation, graphical, and tabular analysis of methods and error performance against true breakpoint in generated data. Simulation assumptions include:
Lognormal distribution of regression parameters
Normal distribution of regression noise
Fixed number of data points (50)
Fixed range of X/VO2 values, with points uniformly distributed

Further simulation studies should consider more dynamic instances of lab data to better match reality and increase simulation accuracy.
