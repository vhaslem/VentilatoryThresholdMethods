############################
# Valon Haslem
# 7 -- 6 -- 2026
# Code for MDD method to estimate VT1 from XLSX file
#
############################


##################### PACKAGES ###############
install.packages("readxl")
library(readxl)


########### IMPLEMENTATION #################
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

mdd_calculation <- function(file_path) {
  # readexcel for file handling
  temp_data = read_excel(file_path, sheet = 1)
  # convert to df for handling
  data = data.frame(x = temp_data$VO2, y = temp_data$VE)
  # use vo2max later to express breakpoint as fraction of vo2max
  vo2_max = max(data$x)
  
  # calc
  mdd_est = mdd_method(data = data)
  return(data.frame(vt1 = mdd_est, vt1_pct_v02_max = mdd_est/vo2_max, vo2_max = vo2_max))
}

########### INPUTS ###################
# on sheet 1 of xlsx file, data should be structured as follows - columnar format
## VO2    VE
## 1.2342 40.23442
## 1.34349 42.23949

## provide complete file path
file.path = ""

## returns as frame
## format: vt1 estimate in measurement format, vt1 estimate in pct v02 max, calculated vo2 max
mdd_df = mdd_calculation(file_path = file.path)

## optional: write to excel
write_xlsx(x = mdd_df, path = "write path and new filename")
