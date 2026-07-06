############################
# Valon Haslem
# 7 -- 6 -- 2026
# Code for segmented method to estimate VT1 from XLSX file
#
############################

### CITATION ###
# Muggeo VMR. segmented: An R Package to Fit Regression Models with Broken-Line Relationships. R News. 2008;8(1):20-25 

##################### PACKAGES ###############
install.packages("segmented")
library(segmented)
install.packages("readxl")
library(readxl)


########### IMPLEMENTATION #################
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

segmented_calculation <- function(file_path) {
  # readexcel for file handling
  temp_data = read_excel(file_path, sheet = 1)
  # convert to df for handling
  data = data.frame(x = temp_data$VO2, y = temp_data$VE)
  # use vo2max later to express breakpoint as fraction of vo2max
  vo2_max = max(data$x)
  
  # calc
  segmented_est = segmented_method(data = data)
  return(data.frame(vt1 = segmented_est, vt1_pct_v02_max = segmented_est/vo2_max, vo2_max = vo2_max))
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
segmented_df = segmented_calculation(file_path = file.path)

## optional: write to excel
write_xlsx(x = segmented_df, path = "write path and filename")


