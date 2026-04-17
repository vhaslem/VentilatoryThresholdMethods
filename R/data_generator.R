###################
# Valon Haslem
# 1 -- 27 -- 2026
# This establishes code to generate
# 1: rand distributed data fitting an exponential distribution 
# using randomly distributed parameters 
# 2: rand distributed data fitting 1 or 2 breakpoints
# using randomly distributed (but dependent betas??)
# 3. test HH accuracy on both generations
# 4. test vslope accuracy on both generations
# 5. test bps accuracy on both generations
###################

# 1

################# DEFUNCT APPROACH ####################
# NOTE:::: for whatever reason, the general curvature function always calculates
# the very first x -value as the kneepoint always. this method doesn't work well
# for growth generated data
# generate_data_and_knee_point_from_exponential_regression <- function(sigma) {
#   # define exp regression line
#   # these are approximations from lab data, shouldn't matter in actual instances
#   alpha = rnorm(1, 25, 1)
#   beta = rnorm(1, 0.3, 0.01)
#   # ASSUMPTION: x_vals are evenly spaced
#   x_vals = seq(from = 1, to = 6, length.out = 100) # automatically uses 100 pts
#   # random noise around y values
#   y_vals = sapply(x_vals, function(x) {alpha * exp(x *beta) + rnorm(1,0,sigma)})
#   
#   # calculate knee point -- find maximum curvature of continuous function -- uses KNEEPOINT FORMULA (insert citation)
#   x_func = seq(from = 10, to = 100, length.out = 1000) # approximate -- should we increase length.out to increase precision of curvature?
#   curvature = sapply(x_func, function(x) {abs(alpha * beta * beta * x ^beta)/((1+(alpha *beta* x ^beta)^2)^15)})
#   kneepoint = x_func[which.max(curvature)] # can I make this handle inconclusive result or multiple maxes?
#   # gather into single list
#   data = list(x = x_vals, 
#               y = y_vals, 
#               bp = kneepoint)
#   return(data)
# }

############### NEW APPROACH USE THIS #############################

set.seed(67)
library(soilphysics)# use for max curve func
generate_data_and_knee_point_from_exponential_regression <- function(sigma) {
  # define exp regression line
  # these are approximations from lab data, shouldn't matter in actual instances
  alpha = rnorm(1, 25, 1)
  beta = rnorm(1, 0.3, 0.01)
  
  x_vals = sort(runif(100, min = 1, max =6)) # automatically uses 100 pts
  # random noise around y values
  y_vals = sapply(x_vals, function(x) {alpha * exp(x *beta) + rnorm(1,0,sigma)})
  
  # calculate kneepoint using maxcurv function
  kneepoint = maxcurv(x.range = c(1, 6), function(x) alpha*exp(x*beta), 
          method = 'LRP', # method assumes NLS -- nonlinear is better
          x0ini = 3,
          graph = FALSE)$x0
  # gather into single list

  return(data.frame(x = x_vals, y = y_vals, bp = unname(kneepoint)))
}

# 2
generate_data_and_breakpoint_from_bp_regression <- function(sigma){
  # def two regression
  r1_base = rnorm(1, 1.5, 1)
  r1_slope = rnorm(1, 30, 0.1)
  r2_base = rnorm(1, -30, 1)
  r2_slope = rnorm(1, 50,0.1)
  # breakpoint is computed intersection
  breakpoint = (r2_base - r1_base)/(r1_slope - r2_slope)
  
  # ASSUME: x_vals are evenly spaced (same amt as exponential data)
  x_vals_reg_1 = sort(runif(50, min = 1, max =3.5))
  y_vals_reg_1 = sapply(x_vals_reg_1, function(x) {r1_base + r1_slope * x + rnorm(1,0, sigma)})
  x_vals_reg_2 = sort(runif(50, min = 3.5, max =6))
  y_vals_reg_2 = sapply(x_vals_reg_2, function(x) {r2_base + r2_slope * x + rnorm(1,0, sigma)})
  
  # make single list
  data = list(x = c(x_vals_reg_1, y_vals_reg_1), 
              y = c(x_vals_reg_2, y_vals_reg_2), 
              bp = breakpoint)
  return(data)
}

# next step: functions to calculate breakpoint using all three methods
# calculate using vslope method
library(segmented)
vslope_method <- function(data) {
  # def regression
  y_vals = data$y
  x_vals = data$x
  model = lm(y_vals ~ x_vals)
  # use segmented package
  seg_model = segmented(model, seg.Z = ~ x_vals, psi = (max(x_vals) + min(x_vals))/2) # initial guess is about midway, could be more precise
  breakpoint = seg_model$psi[2] # the x_value estimate produced
  return(breakpoint)
}

# calculate using strucchange
library(strucchange)
strucchange_method <- function(data) {
  # def regression
  y_vals = data$y
  x_vals = data$x
  # use strucchange package
  bp_obj = breakpoints(y_vals ~ x_vals, breaks = 1) # what to do if no breakpoints found?
  # should i handle NAs by taking mean or by dropping? 
  # just take mean
  breakpoint = ifelse(is.na(bp_obj$breakpoints[1]), mean(x_vals), x_vals[bp_obj$breakpoints[1]])
  return(breakpoint)
}

# calculate using novel method 
hodgin_haslem_method <- function(data) {
  # def regression log-linear
  y_vals = data$y
  x_vals = data$x
  # use fitted values from the log-linear regression to compute nls model
  # wont use log linear because it will amplify the noise
  model = lm(log(y_vals) ~ x_vals)
  coefs = coef(model)
  fitted_alpha = exp(coefs['(Intercept)'])
  fitted_beta = coefs['x_vals']
  # fit a nonlinear model, assume nonlinear distribution, should avoid noise magnification
  model = nls(y_vals ~ alpha * exp(beta * x_vals), start = list(alpha = fitted_alpha, beta = fitted_beta))
  coefs = coef(model)
  # form is y = alpha e^beta * x
  alpha = coefs["alpha"]
  beta = coefs["beta"]
  # now define derivative
  dx_vals = sapply(x_vals, function(x) {alpha * beta * exp(beta * x)})
  # want max, but x value of max DIFFERENCE
  dx_diff_vals = diff(dx_vals) # this new list is 1 length shorter
  breakpoint = x_vals[which.max(dx_diff_vals)] 
  return(breakpoint)
}

###########################
library(dplyr)
library(purrr)
library(tidyverse)

# first, define the number of sigmas to use
len_sigmas = 5 # can be tweaked alter
sigmas = seq(from = 1, to = 5, length.out = len_sigmas)

# we need to generate a bunch of data
exp_data_generated_df <- expand.grid(
  Sigma = sigmas, 
  Rep = 1:1000 # because we want 1000 instances
) %>%
  # Use pmap_dfr to pass multiple columns (Location, Treatment) to your function
  # and automatically bind the resulting 1-row data frames together
  mutate(results = pmap(list(Sigma), generate_data_and_knee_point_from_exponential_regression)) %>%
  unnest(results)

# reformat
exp_data = exp_data_generated_df %>% group_by(Rep, Sigma) %>%
  summarize(
    x_vals = list(x),
    y_vals = list(y),
    bp = mean(bp),
    sigma = mean(Sigma)
  )

# setup our final df
# now we use case logic to compute
# formula is sqrt((bp - calc value)^2)
computed_df <- exp_data %>%
  group_by(Rep, Sigma) %>%
  reframe(
    VT.Methods = c("V-Slope", "Strucchange", "Novel.Method"),
    Noise = Sigma[1], # will need this to group by in graph
    current_data = list(data.frame(x = x_vals[[1]], y = y_vals[[1]])), 
    RSE = c(
      sqrt((bp - vslope_method(current_data[[1]]))^2),
      sqrt((bp - strucchange_method(current_data[[1]]))^2),
      sqrt((bp - hodgin_haslem_method(current_data[[1]]))^2)
    )
  )

values_df <- computed_df %>%
  filter(Noise == 5) %>%
  select(Rep, VT.Methods, RSE) %>%
  pivot_wider(
    names_from = VT.Methods,
    values_from = RSE
  )
###################
library(writexl)
folder_path <- "C:/Users/valon/OneDrive/Desktop/at_estimation/rse_values_recreational.xlsx"
write_xlsx(x = values_df, path = folder_path)

#############
ggplot(computed_df, aes(x = factor(Noise), y = RSE, fill = VT.Methods)) +
  geom_boxplot(outlier.size = 1, alpha = 0.7) +
  # Customizing colors and theme for a professional look
  scale_fill_brewer(palette = "Set2") + 
  theme_minimal() + 
  labs(
    title = "Error Analysis of VT Methods",
    x = "Noise in Simulated Data (Standard Error)",
    y = "Root Square Error",
    fill = "VT Methods"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

df <- data.frame(
  Noise = sigmas,
  VT.Method = rep(c("V-Slope", "Strucchange", "Novel.Method"), times = 1000),
  Values = numeric(15000)
)
rmse_df = data.frame('VSlope' = numeric(len_sigmas),
                    'Strucchange' = numeric(len_sigmas),
                    'Hodgin.Haslem' = numeric(len_sigmas))

for (i in 1:len_sigmas) {
  # define a sigma
  sigma = sigmas[i]
  squared_error_df = data.frame('VSlope' = numeric(1000),
             'Strucchange' = numeric(1000),
             'Hodgin.Haslem' = numeric(1000))
  for (j in 1:1000){
    # gen data and prep for usage
    key_vals = generate_data_and_knee_point_from_exponential_regression(sigma)
    bp = key_vals$bp
    df = data.frame(x = unlist(key_vals$x), y = unlist(key_vals$y))
    # compute squared error for each breakpoint estimate
    squared_error_df[j, 'VSlope'] = (vslope_method(df) - bp)**2
    squared_error_df[j, 'Strucchange'] = (strucchange_method(df) - bp)**2
    squared_error_df[j, 'Hodgin.Haslem'] = (hodgin_haslem_method(df) - bp)**2
  }
  # now compute mean squared errors for each
  rmse_df[i, 'VSlope'] = sqrt(mean(squared_error_df$VSlope))
  rmse_df[i, 'Strucchange'] = sqrt(mean(squared_error_df$Strucchange))
  rmse_df[i, 'Hodgin.Haslem'] = sqrt(mean(squared_error_df$Hodgin.Haslem))
}

# new objective, graph mse using ggplot2
library(ggplot2)
library(tidyr)
# plot
ggplot(rmse_df, aes(x = sigmas)) +
  # Layer 1: VSlope
  geom_point(aes(y = VSlope, color = "VSlope"), shape = 16) +
  geom_line(aes(y = VSlope, color = "VSlope"), linewidth = 1.1) +
  
  # Layer 2: Strucchange
  geom_point(aes(y = Strucchange, color = "Strucchange"), shape = 17) +
  geom_line(aes(y = Strucchange, color = "Strucchange"), linewidth = 1.1) +
  
  # Layer 3: Hodgin-Haslem (Using the dot notation for the column name)
  geom_point(aes(y = Hodgin.Haslem, color = "Hodgin-Haslem"), shape = 18) +
  geom_line(aes(y = Hodgin.Haslem, color = "Hodgin-Haslem"), linewidth = 1.1) +
  
  # Customizing the legend and labels
  labs(
    title = "Error From True Breakpoint Using Simulated Segmented Data", 
    x = "Noise of Data (Standard Error)", 
    y = "RMSE",
    color = "Methods"
  ) +
  theme_minimal()


