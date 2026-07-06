############################
# Valon Haslem
# 7 -- 6 -- 2026
# Code for strucchange method to estimate VT1 from XLSX file
#
############################

### CITATION ###
# Zeileis A, Leisch F, Hornik K, Kleiber C (2002). 
#“strucchange: An R Package for Testing for Structural Change in Linear Regression Models.” 
# Journal of Statistical Software, 7(2), 1–38. doi:10.18637/jss.v007.i02

##################### PACKAGES ###############
install.packages("strucchange")
library(strucchange)
install.packages("readxl")
library(readxl)


########### IMPLEMENTATION #################
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


strucchange_calculation <- function(file_path) {
  # readexcel for file handling
  temp_data = read_excel(file_path, sheet = 1)
  # convert to df for handling
  data = data.frame(x = temp_data$VO2, y = temp_data$VE)
  # use vo2max later to express breakpoint as fraction of vo2max
  vo2_max = max(data$x)
  
  # calc
  strucchange_est = strucchange_method(data = data)
  return(data.frame(vt1 = strucchange_est, vt1_pct_v02_max = strucchange_est/vo2_max, vo2_max = vo2_max))
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
strucchange_df = strucchange_calculation(file_path = file.path)

## optional: write to excel
write_xlsx(x = strucchange_df, path = "write path and filename")


