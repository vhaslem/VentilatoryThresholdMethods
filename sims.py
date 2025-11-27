from estimators import breakpointFitter, exponentialFitter, vslopeModel
import numpy as np
import pandas as pd
from scipy.stats import truncnorm



def test_strucModel(N, sigma, alpha, n, scale):
    acc_values = {}
    acc_values['pred y'] = []
    acc_values['exp ci'] =[]
    acc_values['vslope ci'] =[]
    stat_values = {}
    stat_values['struc'] = []
    stat_values['exp'] =[]
    stat_values['v'] = []
    x_values = {}
    x_values['break'] = []
    x_values['Strucchange'] = []
    x_values['VSlope Method'] = []
    x_values['Hodgin-Haslem'] = []
    breaks = np.random.uniform(.25 * scale, .75 * scale, size=N)
    
    for i in range(N):

        struc_model = breakpointFitter()
        # fit data from struc model breaks[i]
        break_est = breaks[i] 
        slope_rand = np.random.uniform(2, 4)
        struc_model.generate_data(n = n, break_x = [break_est], slopes=(1, slope_rand), intercepts= (0, break_est - slope_rand * break_est), noise = sigma, scale = scale)
        struc_model.fit(x = struc_model.x, y = struc_model.y)
        
        struc_model_ssr = struc_model.ssr_total_
        bp_guess = struc_model.breakpoints_[0]
        struc_model_pi = struc_model.predict_interval(x_new = struc_model.x[bp_guess], alpha = alpha)[0:3]
        struc_stats = [break_est, struc_model_ssr] +  struc_model_pi
        pred = struc_model.predict(x_new = struc_model.x[bp_guess])
        stat_values['struc'].append(struc_stats)
        # fit model for other two
        exp_model = exponentialFitter(x= struc_model.x, y = struc_model.y)
        exp_model.fit()
        exp_model_pred = exp_model.compute_t_max()
        
        exp_model_ssr = exp_model.summary()['SSR']
        exp_model_pi = exp_model.predict_interval(exp_model_pred, alpha = alpha)[0:3]
         # use to test accuracy,exp_model.approx_interval_prediction(exp_model_pred, alpha = alpha) #
        exp_stats = [exp_model_pred, exp_model_ssr] + exp_model_pi # remove later
        # exp_stats = [exp_model_pred, exp_model_ssr] + exp_model_pi
        stat_values['exp'].append(exp_stats)
        
        
        v_model = vslopeModel(x= struc_model.x, y = struc_model.y)
        v_model.fit()
        v_model_pred = v_model.intersection()[0]
        v_model_ssr = v_model.summary()['SSR']
        v_model_pi = v_model.predict_interval(v_model_pred, alpha = alpha)[0:3]
        v_model_stats = [v_model_pred, v_model_ssr] + v_model_pi
        stat_values['v'].append(v_model_stats)

        # print(exp_stats, v_model_stats, break_est, pred) # code currently measured against what breakpoint predicts
        # calculate accuracy
        acc_values['pred y'].append(pred)
        acc_values['exp ci'].append((exp_model_pi[1], exp_model_pi[2]))
        acc_values['vslope ci'].append((v_model_pi[1], v_model_pi[2]))

        x_values['break'].append(break_est)
        x_values['Strucchange'].append(bp_guess)
        x_values['VSlope Method'].append(v_model_pred)
        x_values['Hodgin-Haslem'].append(exp_model_pred)
    
    return stat_values, acc_values, x_values

def test_vslopeModel(N, sigma, alpha, n, scale):
    acc_values = {}
    acc_values['pred y'] = []
    acc_values['exp ci'] =[]
    acc_values['struc ci'] =[]
    stat_values = {}
    stat_values['struc'] = []
    stat_values['exp'] =[]
    stat_values['v'] = []
    x_values = {}
    x_values['break'] = []
    x_values['Strucchange'] = []
    x_values['VSlope Method'] = []
    x_values['Hodgin-Haslem'] = []
    breaks = np.random.uniform(.25 * scale, .75 * scale, size=N)
    for i in range(N):

        # fit data from vslope model
        break_est = breaks[i]
        slope_rand = np.random.uniform(0,4)
        v_model = vslopeModel(x = None, y = None)
        v_model.generate_sample_data(n = n, x_break = break_est, m1 = 1, b1 = 0, m2 = slope_rand + 1, b2 = break_est - break_est*slope_rand, noise_std = sigma, x_range=(0, scale))

        struc_model = breakpointFitter()
        struc_model.fit(x = v_model.x, y = v_model.y)
        bp_guess = v_model.x[struc_model.breakpoints_[0]]
        
        struc_model_ssr = struc_model.ssr_total_
        struc_model_pi = struc_model.predict_interval(x_new= bp_guess, alpha = alpha)[0:3]
        struc_stats = [break_est, struc_model_ssr] +  struc_model_pi
        
        
        stat_values['struc'].append(struc_stats)
        # fit model for other two
        exp_model = exponentialFitter(x= v_model.x, y = v_model.y)
        exp_model.fit()
        exp_model_pred = exp_model.compute_t_max()
        
        exp_model_ssr = exp_model.summary()['SSR']
        exp_model_pi =  exp_model.predict_interval(exp_model_pred, alpha = alpha)[0:3]
         # use to test accuracy,exp_model.approx_interval_prediction(exp_model_pred, alpha = alpha) #
        exp_stats = [exp_model_pred, exp_model_ssr] + exp_model_pi # remove later
        exp_stats = [exp_model_pred, exp_model_ssr] + exp_model_pi
        stat_values['exp'].append(exp_stats)

        v_model.fit()
        v_model_pred = v_model.intersection()[0]
        

        v_model_ssr = v_model.summary()['SSR']
        v_model_pi = v_model.predict_interval(v_model_pred, alpha = alpha)[0:3]
        v_model_stats = [v_model_pred, v_model_ssr] + v_model_pi
        stat_values['v'].append(v_model_stats)
        pred = v_model.predict(v_model_pred)
        
        # calculate accuracy
        acc_values['pred y'].append(pred)
        acc_values['exp ci'].append((exp_model_pi[1], exp_model_pi[2]))
        acc_values['struc ci'].append((struc_model_pi[1], struc_model_pi[2]))

        x_values['break'].append(break_est)
        x_values['Strucchange'].append(bp_guess)
        x_values['VSlope Method'].append(v_model_pred)
        x_values['Hodgin-Haslem'].append(exp_model_pred)

    return stat_values, acc_values, x_values

def test_expModel(N, sigma, alpha, n, scale):
    acc_values = {}
    acc_values['pred y'] = []
    acc_values['vslope ci'] =[]
    acc_values['struc ci'] =[]
    stat_values = {}
    stat_values['struc'] = []
    stat_values['exp'] =[]
    stat_values['v'] = []
    # x values for graphic
    x_values = {}
    x_values['break'] = []
    x_values['Strucchange'] = []
    x_values['VSlope Method'] = []
    x_values['Hodgin-Haslem'] = []
    breaks = np.random.uniform(.25 * scale, .75 * scale, size=N)
    for i in range(N):

        # fit data from vslope model
        break_est = breaks[i]
        
        exp_model = exponentialFitter(x = None, y = None)
        exp_model.generate_sample_data(n = n, x_range = (0,scale), a= np.random.uniform(1,3), b = np.random.uniform(0.1, .9), noise_std = sigma)

        struc_model = breakpointFitter()
        struc_model.fit(x = exp_model.x, y = exp_model.y)
        
        struc_model_ssr = struc_model.ssr_total_
        bp_guess = exp_model.x[struc_model.breakpoints_[0]]
        struc_model_pi = struc_model.predict_interval(x_new= bp_guess, alpha = alpha)[0:3]
        struc_stats = [break_est, struc_model_ssr] +  struc_model_pi
        stat_values['struc'].append(struc_stats)
        # fit model for other two
        
        exp_model.fit()
        exp_model_pred = exp_model.compute_t_max()
        exp_model_ssr = exp_model.summary()['SSR']
        exp_model_pi = exp_model.predict_interval(exp_model_pred, alpha = alpha)[0:3]
        # use to test accuracy,exp_model.approx_interval_prediction(exp_model_pred, alpha = alpha) #
        exp_stats = [exp_model_pred, exp_model_ssr] + exp_model_pi # remove later
        stat_values['exp'].append(exp_stats)
        pred = exp_model.predict(x_new = exp_model_pred)
        
        v_model = vslopeModel(x= exp_model.x, y = exp_model.y)
        v_model.fit()
        v_model_pred = v_model.intersection()[0]
        v_model_ssr = v_model.summary()['SSR']
        v_model_pi = v_model.predict_interval(v_model_pred, alpha = alpha)[0:3]
        v_model_stats = [v_model_pred, v_model_ssr] + v_model_pi
        stat_values['v'].append(v_model_stats)
        # print(struc_stats, exp_stats, v_model_stats)

        # calculate accuracy
        acc_values['pred y'].append(pred)
        acc_values['vslope ci'].append((exp_model_pi[1], exp_model_pi[2]))
        acc_values['struc ci'].append((struc_model_pi[1], struc_model_pi[2]))
        # calculate difference of x values:
        x_values['break'].append(break_est)
        x_values['Strucchange'].append(bp_guess)
        x_values['VSlope Method'].append(v_model_pred)
        x_values['Hodgin-Haslem'].append(exp_model_pred)


    return stat_values, acc_values, x_values

def compute_accuracy(acc_dict, model_names, N):
    for (pred_model, ci_model) in model_names:
        hits = [
            1 if acc_dict[f"{ci_model} ci"][i][0] <= acc_dict['pred y'][i] <= acc_dict[f"{ci_model} ci"][i][1]
            else 0
            for i in range(N)
        ]
        acc = sum(hits)/len(hits)
        print(f"Accuracy of {ci_model} predicting {pred_model} AT: {acc:.3f}")
import matplotlib.pyplot as plt
def make_fig(x_vals_dict, test_data, sigma):
        # Extract keys and values
# Compute mean for each category (handles both scalar and list)
    means = {k: np.mean(v) if isinstance(v, (list, np.ndarray)) else v for k, v in x_vals_dict.items()}

    # Extract keys and mean values
    keys = list(means.keys())
    values = list(means.values())

    # Compute differences from the first value
    base = values[0]
    diffs = [v - base for v in values]

    # Assign distinct colors
    colors = plt.cm.tab10(np.arange(len(keys)))

    # Create the figure
    plt.figure(figsize=(8, 5))
    for i, key in enumerate(keys):
        plt.scatter(i, diffs[i], color=colors[i], s=100, label=key)

    # Formatting
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xticks(range(len(keys)), keys)
    plt.ylabel("Mean Difference from Break Points  ")
    plt.title(f"Category Mean Differences With {test_data} and $\sigma = $ {sigma}")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"category_mean_differences_{test_data}.png", dpi=300, bbox_inches='tight')
    plt.close()


def make_fig_scatter(x_vals_dict, test_data, sigma):
        
    # Ensure all lists are numpy arrays for vectorized math
    x_vals_dict = {k: np.array(v, dtype=float) for k, v in x_vals_dict.items()}

    # Reference category (first key)
    ref_key = list(x_vals_dict.keys())[0]
    ref_values = x_vals_dict[ref_key]

    # Prepare color map
    colors = plt.cm.tab10(np.arange(len(x_vals_dict)))

    # Create scatter plot
    plt.figure(figsize=(8, 5))
    for i, (key, values) in enumerate(x_vals_dict.items()):
        # Align lengths by truncating or padding if necessary
        min_len = min(len(values), len(ref_values))
        diff = values[:min_len] - ref_values[:min_len]
        
        # x-axis: index positions (0, 1, 2, ...)
        x = np.arange(min_len)
        
        plt.scatter(x, diff, color=colors[i], s=60, label=key, alpha=0.8)

    # Formatting
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel("Index")
    plt.ylabel(f"Difference from {ref_key}")
    plt.title(f"Break Difference From Model By {test_data}, $\sigma = $ {sigma}")
    plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save and close
    plt.savefig(f"color_coded_differences_{test_data}.png", dpi=300, bbox_inches='tight')
    plt.close()

    
# NOTE: n is currently set to 100 for values
def main(N = 1000, sigma = .5, alpha = 0.05, n = 150, scale = 6):
    
    # test strucModel
    struc_stats, struc_accuracy, StrucModelGeneration = test_strucModel(N = N, sigma = sigma, alpha = alpha, n= n, scale = scale)
    vslope_stats, vslope_accuracy, VSlopeGeneration = test_vslopeModel(N = N, sigma = sigma, alpha = alpha, n= n, scale = scale)
    exp_stats, exp_accuracy, ExponentialGeneration = test_expModel(N = N, sigma = sigma, alpha = alpha, n= n, scale = scale)
    for dicts in [(StrucModelGeneration, 'Strucchange Data'), (VSlopeGeneration, 'VSlope Data'), (ExponentialGeneration, 'Exponential Data')]:
        make_fig(dicts[0], dicts[1], sigma)
    for dicts in [(StrucModelGeneration, 'Strucchange Data'), (VSlopeGeneration, 'VSlope Data'), (ExponentialGeneration, 'Exponential Data')]:
        make_fig_scatter(dicts[0], dicts[1], sigma)

    # compute accuracy for each:
    print(f'N = {N}, n = {n}, sigma = {sigma}, alpha = {alpha}')
    compute_accuracy(struc_accuracy, [('struc','exp'), ('struc','vslope')], N)
    compute_accuracy(vslope_accuracy, [('vslope','exp'), ('vslope','struc')], N)
    compute_accuracy(exp_accuracy, [('exp','vslope'), ('exp','struc')], N)

    


if __name__ == '__main__':
    main()



