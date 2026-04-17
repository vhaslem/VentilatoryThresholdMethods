###################################
# Valon Haslem
# 2--17--2026
# This script takes a bootstrap sample of VO2 Max test data
# and removes random x% of data
# and evaluates accuracy of 3 methods
# by distribution of VT1 predictions
# distributions will be in % of vo2 max
# evaluate accuracy by how tight the distribution of guesses is
######################################
# methods to calculate breakpoint
# NOTE: takes data input as data.frame(x,y), may not be simplest design
# calculate using vslope method
set.seed(67)
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


library(readxl)
library(ggplot2)

# Get all file names
folder_path <- "C:/Users/valon/OneDrive/Desktop/at_estimation/at_data/VO2 Max Data"
all_files <- list.files(path = folder_path, pattern = "*.xlsx", full.names = TRUE)

# Define num of bootstrap iterations
n_bootstrap <- 1000
resampled_path <- sample(all_files, size = n_bootstrap, replace = TRUE)

# df of results for histograms
result_df = data.frame('VSlope' = numeric(n_bootstrap),
                       'Strucchange' = numeric(n_bootstrap),
                       'Hodgin.Haslem' = numeric(n_bootstrap))

for (b in 1:n_bootstrap) {
  # Resample the FILE LIST with replacement
  # bootstrapping by taking a sample (1 at a time)
  # use readexcel to read first sheet from random sample
  
  # and we just want 95 percent!!!! STABILITY ANALYSIS
  temp_data = read_excel(resampled_path[b], sheet = 1)
  # convert to df for handling
  data = data.frame(x = temp_data$VO2, y = temp_data$VE)
  data = data[sample(1:nrow(data), size = 0.95 * nrow(data)), ]
  
  # use vo2max later to express breakpoint as fraction of vo2max
  vo2_max = max(data$x)
  # Store the breakpoint, but express as a fraction of vo2max
  result_df[b,'VSlope'] = vslope_method(data = data)/vo2_max
  result_df[b, 'Strucchange'] = strucchange_method(data = data)/vo2_max
  result_df[b, 'Hodgin.Haslem'] = hodgin_haslem_method(data = data)/vo2_max
}

## use this to graph results, do one for each method
ggplot(result_df, aes(x = VSlope)) +
  # histogram
  # use after_stat(density) so the curve and bars share the same scale
  geom_histogram(aes(y = after_stat(density)), 
                 bins = 15, fill = "gray80", color = "white") +
  
  # density line
  # Mapping 'color' to a string creates the legend entry
  geom_density(aes(color = "VT Estimate Distribution"), linewidth = 1) +
  
  # 3. A Vertical Mean Line
  geom_vline(aes(xintercept = mean(VSlope), color = "Mean VT Estimate"), 
             linetype = "dashed", linewidth = 1) +
  
  # define colors and legend
  scale_color_manual(name = "Legend", 
                     values = c("VT Estimate Distribution" = "blue", 
                                "Mean VT Estimate" = "red")) +
  labs(title = "Stability Analysis of V-Slope",
       x = "VT Estimate From Bootstrapped Uncertain Dataset (%VO2 Max) ",
       y = "Density") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

## Hodgin Haslem
ggplot(result_df, aes(x = Hodgin.Haslem)) +
  # histogram
  # use after_stat(density) so the curve and bars share the same scale
  geom_histogram(aes(y = after_stat(density)), 
                 bins = 15, fill = "gray80", color = "white") +
  
  # density line
  # Mapping 'color' to a string creates the legend entry
  geom_density(aes(color = "VT Estimate Distribution"), linewidth = 1) +
  
  # 3. A Vertical Mean Line
  geom_vline(aes(xintercept = mean(Hodgin.Haslem), color = "Mean VT Estimate"), 
             linetype = "dashed", linewidth = 1) +
  
  # define colors and legend
  scale_color_manual(name = "Legend", 
                     values = c("VT Estimate Distribution" = "blue", 
                                "Mean VT Estimate" = "red")) +
  labs(title = "Stability Analysis of Novel Method",
       x = "VT Estimate From Bootstrapped Uncertain Dataset (%VO2 Max) ",
       y = "Density") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))

## Strucchange
ggplot(result_df, aes(x = Strucchange)) +
  # histogram
  # use after_stat(density) so the curve and bars share the same scale
  geom_histogram(aes(y = after_stat(density)), 
                 bins = 15, fill = "gray80", color = "white") +
  
  # density line
  # Mapping 'color' to a string creates the legend entry
  geom_density(aes(color = "VT Estimate Distribution"), linewidth = 1) +
  
  # 3. A Vertical Mean Line
  geom_vline(aes(xintercept = mean(Strucchange), color = "Mean VT Estimate"), 
             linetype = "dashed", linewidth = 1) +
  
  # define colors and legend
  scale_color_manual(name = "Legend", 
                     values = c("VT Estimate Distribution" = "blue", 
                                "Mean VT Estimate" = "red")) +
  labs(title = "Stability Analysis of Strucchange",
       x = "VT Estimate From Bootstrapped Uncertain Dataset (%VO2 Max) ",
       y = "Density") +
  theme_minimal()+
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size = 14))





library(tidyverse)

# 1. Transform data to long format
plot_data <- result_df %>%
  pivot_longer(
    cols = c(VSlope, Hodgin.Haslem, Strucchange),
    names_to = "Method",
    values_to = "VT_Estimate"
  ) %>%
  # Clean up the names so they look nice in the legend
  mutate(Method = case_match(Method,
                             "VSlope" ~ "V-Slope",
                             "Hodgin.Haslem" ~ "Novel Method",
                             "Strucchange" ~ "Strucchange",
                             .default = Method
  ))
# 2. Calculate means for the vertical lines
means_df <- plot_data %>%
  group_by(Method) %>%
  summarize(mean_val = mean(VT_Estimate, na.rm = TRUE))

# 3. Create the single overlaid plot
ggplot(plot_data, aes(x = VT_Estimate, fill = Method, color = Method)) +
  # Semi-transparent Histograms
  # position = "identity" allows them to overlap instead of stacking
  geom_histogram(aes(y = after_stat(density)), 
                 bins = 20, position = "identity", alpha = 0.3, color = "white") +
  
  # Color-coded Density Lines
  geom_density(linewidth = 1, alpha = 0) + 
  
  # Color-coded Vertical Mean Lines
  geom_vline(data = means_df, aes(xintercept = mean_val, color = Method),
             linetype = "dashed", linewidth = 0.8) +
  
  # Formatting
  scale_fill_brewer(palette = "Set1") +
  scale_color_brewer(palette = "Set1") +
  labs(
    title = "Stability Analysis Comparison",
    subtitle = "Overlaid distributions of VT estimates across methods",
    x = "VT Estimate From Bootstrapped Uncertain Dataset (%VO2 Max)",
    y = "Density",
    fill = "Method",
    color = "Method"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
    plot.subtitle = element_text(hjust = 0.5),
    legend.position = "bottom"
  )
