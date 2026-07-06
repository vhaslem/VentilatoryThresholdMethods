###################
# Valon Haslem
# 7 -- 6 -- 2026
# Generates 5000 GXT simulations and computes vt estimate on all, putting values into dataframe
# NOTE: entire script takes ~20 minutes, longer if more sims
###################

#################### DATA GENERATION ####################
set.seed(3)
library(soilphysics) # use for max curve function
# data gen parameters
num_points = 50 # default 50 pts
# mean and sd used from lab data
mean_alpha = 14.70414
mean_beta = 0.7010852
sd_alpha = 5.943829
sd_beta = 0.3192802

generate_data_and_knee_point_from_exponential_regression <- function(sigma) {
  # define exp regression line
  alpha = rlnorm(1, mean_alpha, sd_alpha) 
  beta = rlnorm(1, mean_beta, sd_beta) 

  x_vals = sort(runif(num_points, min = 1, max =2)) 
  # noise around y values
  y_vals = sapply(x_vals, function(x) {alpha * exp(x *beta) + rnorm(1, 0, sigma)})
  
  # kneepoint calculation -- uses maxcurv function
  kneepoint = maxcurv(x.range = c(min(x_vals), max(x_vals)), function(x) alpha*exp(x*beta), 
          method = 'spline', # method assumes NLS -- nonlinear is better
          x0ini = 1.5,
          graph = FALSE)$x0
  
  return(data.frame(x = x_vals, y = y_vals, bp = unname(kneepoint)))
}


########## VT/BREAKPOINT CALCS #################
### SEGMENTED METHOD ###
library(segmented)
segmented_method <- function(data) {
  # def regression
  y_vals = data$y
  x_vals = data$x
  model = lm(y_vals ~ x_vals)
  # use segmented package
  seg_model = segmented(model, seg.Z = ~ x_vals, psi = mean(x_vals)) 
  breakpoint = seg_model$psi[2] 
  return(breakpoint)
}

### STRUCCHANGE METHOD ###
library(strucchange)
strucchange_method <- function(data) {
  # def regression
  y_vals = data$y
  x_vals = data$x
  # use strucchange package
  bp_obj = breakpoints(y_vals ~ x_vals, breaks = 1) 
  ### NOTE: in rare instances, strucchange method doesn't converge. mean is taken to eliminate nulls
  # in sim, didn't find any instances of nonconvergence, conditional added to ensure functionality
  breakpoint = ifelse(is.na(bp_obj$breakpoints[1]), mean(x_vals), x_vals[bp_obj$breakpoints[1]])
  return(breakpoint)
}

### MDD METHOD ###
mdd_method <- function(data) {
  # def regression log-linear
  y_vals = data$y
  x_vals = data$x
  # use fitted values from the log-linear regression to compute nls model
  model = lm(log(y_vals) ~ x_vals)
  coefs = coef(model)
  fitted_alpha = exp(coefs['(Intercept)'])
  fitted_beta = coefs['x_vals']
  # fit a nonlinear model, assume nonlinear distribution
  model = nls(y_vals ~ alpha * exp(beta * x_vals), start = list(alpha = fitted_alpha, beta = fitted_beta),
              control = nls.control(warnOnly = TRUE))
  coefs = coef(model)
  # take estimates for alpha and beta
  alpha = coefs["alpha"]
  beta = coefs["beta"]
  # now define derivative
  dx_vals = sapply(x_vals, function(x) {alpha * beta * exp(beta * x)})
  # take x value of max difference in derivatives
  dx_diff_vals = diff(dx_vals) 
  breakpoint = x_vals[which.max(dx_diff_vals)] 
  return(breakpoint)
}

######## SIMULATIONS #############
library(dplyr)
library(purrr)
library(tidyverse)

# NUM SIGMAS and RANGE
len_sigmas = 5 # 5 different values
sigmas = seq(from = 1, to = 5, length.out = len_sigmas) # set from 1 to 5 and

# generated data df
exp_data_generated_df <- expand.grid(
  Sigma = sigmas, # we use range set by sigmas earlier
  Rep = 1:1000 # because we want 1000 instances
) %>% mutate(results = pmap(list(Sigma), generate_data_and_knee_point_from_exponential_regression)) %>%
  unnest(results)

# reformatted generated data df
exp_data = exp_data_generated_df %>% group_by(Rep, Sigma) %>%
  summarize(
    x_vals = list(x),
    y_vals = list(y),
    bp = mean(bp),
    vo2_max = max(y),
    sigma = mean(Sigma)
  )

# computations df
computed_df <- exp_data %>%
  group_by(Rep, Sigma) %>%
  reframe(
    VT.Methods = c("Segmented", "Strucchange", "MDD"),
    Noise = Sigma[1], # will need this to group by in graph
    current_data = list(data.frame(x = x_vals[[1]], y = y_vals[[1]])),
    percent_error = c(
      abs(bp - segmented_method(current_data[[1]]))/bp,
      abs(bp - strucchange_method(current_data[[1]]))/bp,
      abs(bp - mdd_method(current_data[[1]]))/bp
    )
  )


########## GRAPHICAL #######################
ggplot(computed_df, aes(x = factor(Noise), y = percent_error, fill = VT.Methods)) +
  geom_boxplot(outlier.size = 1, alpha = 0.7) +
  # Customizing colors and theme for a professional look
  scale_fill_brewer(palette = "Set2") + 
  theme_minimal() + 
  labs(
    x = "Standard Deviation of Regression Noise",
    y = "Percent Error",
    fill = "VT1 Methods"
  ) +
  theme_minimal() +
  theme(
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black"),
        axis.text = element_text(color = "black"))

############ TABULAR ANALYSIS ##################

## tabular performance of error
error_performance_df <- computed_df %>%
  group_by(Sigma, VT.Methods) %>%
  reframe( error_2 = sum(percent_error < 0.2)/1000,
           error_1 = sum(percent_error < 0.1)/1000,
           error_05 = sum(percent_error < 0.05)/1000)

## mean performance of error
mean_performance_df <- computed_df %>% 
  group_by(VT.Methods) %>%
  reframe(mean_percent_error = mean(percent_error))

## EXAMPLE values performance ##
values_df <- computed_df %>%
  filter(Noise == 3) %>%
  select(Rep, VT.Methods, RSE) %>%
  pivot_wider(
    names_from = VT.Methods,
    values_from = RSE
  )
################### optional write to excel ################
library(writexl)
folder_path = "input folder path"
write_xlsx(x = error_performance_df, path = folder_path)
