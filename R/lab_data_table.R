########################
# Valon Haslem
# 2 -- 21 -- 26
# Compute breakpoints for each computed lab value
########################

# methods to calculate breakpoint
# NOTE: takes data input as data.frame(x,y), may not be simplest design
# calculate using vslope method
set.seed(67)
library(writexl)

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

# calculate using novel method -- dubbed HODGIN HASLEM
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

adj_hodgin_haslem_method <- function(data) {
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
  # scaled to diff between x values
  dydx_diff_vals = dx_diff_vals/diff(x_vals)
  breakpoint = x_vals[which.max(dydx_diff_vals)] 
  return(breakpoint)
}



# Get all file names
folder_path <- "C:/Users/valon/OneDrive/Desktop/at_estimation/at_data/VO2 Max Data"
all_files <- list.files(path = folder_path, pattern = "*.xlsx", full.names = TRUE)

# result dataframes
result_df = data.frame()

# loop, compute methods
library(readxl)
for (l in 1:length(all_files)) {
  
  # use readexcel to read first sheet from random sample
  temp_data = read_excel(all_files[l], sheet = 1)
  # convert to df for handling
  data = data.frame(x = temp_data$VO2, y = temp_data$VE)
  
  # use vo2max later to express breakpoint as fraction of vo2max
  vo2_max = max(data$x)
  # store the numeric breakpoint
  vslope_bp = vslope_method(data = data)
  strucc_bp = strucchange_method(data = data)
  hh_bp = hodgin_haslem_method(data = data)
  adj_hh_bp = adj_hodgin_haslem_method(data=data)
  
  # assign to basic results
  result_df[l,'VSlope'] = vslope_bp
  result_df[l,'VSlope As Percent of VO2Max'] = vslope_bp/vo2_max
  result_df[l, 'Strucchange'] = strucc_bp
  result_df[l, 'Strucchange As Percent of VO2Max'] = strucc_bp/vo2_max
  result_df[l, 'Hodgin.Haslem'] = hh_bp
  result_df[l, 'Hodgin.Haslem As Percent of VO2Max'] = hh_bp/vo2_max
  # Store the breakpoint, but express as a fraction of vo2max

}

write_xlsx(x = result_df, path = "C:/Users/valon/OneDrive/Desktop/at_estimation/VTMethods/results_no_vo2_restriction.xlsx")

