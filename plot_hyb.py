import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import r2_score
import time
import tkinter as tk
from tkinter import messagebox
import math
import matplotlib as mpl
from scipy.spatial import ConvexHull

# our functions
import predict_Beta_I
import choice_start_day

import warnings
warnings.filterwarnings(action='ignore')


def plot_one(ax, chosen_col,
             predicted_days, seed_df, predicted_I, 
             predicted_Inc,
             beggining_beta, predicted_beta,
             seed_number, execution_time):
    '''
    Plotting the graph for a seed.
    
    Parameters:

    - ax -- area for the plot
    - predicted_days -- predicted days
    - seed_df -- DataFrame of seed, created by the regular network
    - predicted_I -- predicted trajectory of the Infected compartment
    - beggining_beta -- predicted initial values of Beta
    - predicted_beta -- predicted values of Beta
    - seed_number -- seed number        
    - execution_time -- time taken to predict Beta   
    - median_values -- sample mean of predicted_I on a specific day
    - lower_bound -- upper boundary of the interval (3 std of predicted_I on a specific day)
    - upper_bound -- lower boundary of the interval (3 std of predicted_I on a specific day)
    '''
    predicted_Inc[np.isnan(predicted_Inc)] = 0.0  
    # when shifting forecasts, sometimes NaN values appear here
    predicted_I[np.isnan(predicted_I)] = 0.0  
    predicted_beta[np.isnan(predicted_beta)] = 0.0
    beggining_beta[np.isnan(beggining_beta)] = 0.0
 
    # find the maximum and its index
    predicted_peak_I = max(predicted_I[0])
    predicted_peak_day = predicted_days[0] + np.argmax(predicted_I[0])
    predicted_peak_Inc = max(predicted_Inc[0])
    predicted_peak_day_Inc = predicted_days[0] + np.argmax(predicted_Inc[0])
    
    actual_I = seed_df['I'].values 
    actual_peak_I = max(actual_I)
    actual_peak_day = np.argmax(actual_I)+1
    
    actual_Inc = seed_df['incidence'].values 
    actual_peak_Inc = max(actual_Inc)
    actual_peak_day_Inc = np.argmax(actual_Inc)+1
    
    peak = [actual_peak_I, predicted_peak_I,
            actual_peak_Inc, predicted_peak_Inc,
            actual_peak_day, predicted_peak_day,
            actual_peak_day_Inc, predicted_peak_day_Inc] 

    # calculate RMSE 
    actual_I = seed_df.iloc[predicted_days[0]:]['I'].values
    rmse_I = rmse(np.nan_to_num(actual_I, neginf=0, posinf=0),
                  np.nan_to_num(predicted_I[0], neginf=0, posinf=0))
    actual_Inc = seed_df.iloc[predicted_days[0]:]['incidence'].values
    rmse_Inc = rmse(np.nan_to_num(actual_Inc, neginf=0, posinf=0),
                  np.nan_to_num(predicted_Inc[0], neginf=0, posinf=0))
    
    actual_Beta = seed_df.iloc[predicted_days[0]:]['Beta'].values 
    actual_Beta = np.nan_to_num(actual_Beta, neginf=0, posinf=0)
    predicted_beta = predicted_beta[:actual_Beta.shape[0]]
    predicted_beta = np.nan_to_num(predicted_beta, neginf=0, posinf=0)
    rmse_Beta = rmse(actual_Beta, predicted_beta)   
    
    #calc R^2
    r2 = r2_score(np.nan_to_num(actual_I, neginf=0, posinf=0),
              np.nan_to_num(predicted_I[0], neginf=0, posinf=0))
    r2_Inc = r2_score(np.nan_to_num(actual_Inc, neginf=0, posinf=0),
              np.nan_to_num(predicted_Inc[0], neginf=0, posinf=0))
    #calc MAPE
    
    
    # display boundary of switch 
    ax.axvline(predicted_days[0], color='red',ls=':')

    if chosen_col=='incidence':
        to_plot = predicted_Inc
    else:
        to_plot = predicted_I
        
    if to_plot.shape[0] > 1:
    # display trajectories of the stochastic mathematical model
        for i in range(to_plot.shape[0]-1):
            ax.plot(predicted_days, to_plot[i+1], color='tab:orange', ls='--', 
                    alpha=0.3, label=f'Predicted {chosen_col} (stoch.)' if i == 0 else '')

        # median calculation
        mean_values = np.mean(to_plot, axis=0) 
        # standard error
        std_dev = np.std(to_plot, axis=0)
        # boundaries: mean ± 3σ (checked for negative values)
        lower_bound = mean_values - 3 * std_dev
        upper_bound = mean_values + 3 * std_dev
        lower_bound = np.maximum(lower_bound, 0)
        # add vertical lines with tick marks for confidence intervals
        for day in range(0, len(predicted_days), 5): 
            ax.errorbar(predicted_days[day], mean_values[day],
                        yerr=[[mean_values[day] - lower_bound[day]], 
                            [upper_bound[day] - mean_values[day]]], 
                        fmt='o', color='black', capsize=2, markersize=2, elinewidth=1, 
                        alpha=0.6, label='$\mu \pm 3\sigma$' if day == 0 else '')

    # display actual and predicted Infected values
    ax.plot(seed_df.index, seed_df[chosen_col].values , color='tab:blue', 
            label='Actual I')
    ax.plot(predicted_days, to_plot[0],color='red', ls='-', 
              alpha=0.9, label=f'Predicted {chosen_col} (det.)')
    
    # add axis labels
    ax.set_xlabel('Days')
    ax.set_ylabel('Infected', color='tab:blue')
    ax.grid(True, alpha=0.3)
        
    ax_b = ax.twinx()
    # display actual and predicted Beta values
    ax_b.plot(seed_df.index, seed_df['Beta'],  color='gray', ls='--', 
              alpha=0.4, label='Actual Beta')

    if len(beggining_beta) > 0:
        given_days = np.arange(predicted_days[0]+1)
        ax_b.plot(given_days, beggining_beta,color='green', ls='--', 
                  alpha=0.7)
    ax_b.plot(predicted_days, predicted_beta,color='green', ls='--', 
              alpha=0.7, label='Predicted Beta ')
    ax_b.set_ylabel("Beta", color='gray')

    ax_b.set_ylim(0, np.max(actual_Beta[:100])*1.1)
    
    # add legend and titles
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_b.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.set_title(f'Switch day {predicted_days[0]}\n'+
                 f'Peak I (act.):{actual_peak_I:.2f}, '+
                 f'Peak I (pred.):{predicted_peak_I:.2f}, \n'+
                 f'Peak Inc (act.):{actual_peak_Inc:.2f}, '+
                 f'Peak Inc (pred.):{predicted_peak_Inc:.2f}, \n'+
                 f'R2 I:{r2:.2f} '+
                 f'R2 Inc:{r2_Inc:.2f} \n',
                 fontsize=10)
    return rmse_I, rmse_Inc, rmse_Beta, r2, r2_Inc, peak


def main_f(I_prediction_method, stochastic, count_stoch_line, 
           beta_prediction_method, type_start_day, seed_numbers,
           show_fig_flag, seed_dirs='test/', sigma=0.1, gamma=0.08,
           ax = None, model_path='', perc_switch=0.01,
          is_filename=False, on_incidence=False):
    '''
    Main function
    
    Parameters:
    - I_prediction_method -- model for constructing the trajectory of Infected
        ['seir']
    - stochastic -- presence of predicted stochastic trajectories of Infected 
    - count_stoch_line -- number of predicted stochastic trajectories
    - beta_prediction_method -- method for predicting Beta values
        ['last_value',
        'expanding mean last value',
        'median beta',
        'regression beta '
        'lstm']
    - type_start_day -- type of choosing the switching day for the model 
        (changing or constant)
    - seed_numbers -- seed numbers for the experiments
    - show_fig_flag -- flag to show the plots
    
    Output:
        Graph for seeds.
    '''
    features_reg = ''
    if ax is None:
        row_n = len(seed_numbers)//2+math.ceil(len(seed_numbers)%2)
        fig, axes = plt.subplots(row_n, 2, figsize=(13, 3.5*row_n))
        axes = axes.flatten()
    else:
        axes = ax

    print(beta_prediction_method)
    # list of RMSE Beta and I for each seed 
    all_rmse_I, all_rmse_Inc = [], []
    all_rmse_Beta = []
    all_r2, all_r2_Inc = [], []
    all_peak = []
    start_days = []
    execution_time = []
    
    
    for idx, seed_number in enumerate(seed_numbers):
        #print(seed_number)
        # read the DataFrame of the seed: S,[E],I,R,Beta
        if is_filename:
            seed_df = pd.read_csv(seed_dirs+seed_number[0])
            window_size = 4
        else:
            seed_df = pd.read_csv(seed_dirs+f'seir_seed_{seed_number}.csv')
            window_size = 4
            
        seed_df = seed_df.iloc[:,:5].copy()
        seed_df.columns = ['S','E','I','R','Beta']
        
        chosen_col='I'
        # calculating Incidence if needed
        if on_incidence:
            chosen_col='incidence'
            temp = seed_df[['E','S']].shift([0,1])
            # Inc_t = (E_t-1 - E_t) - (S_t - S_t-1)
            seed_df['incidence'] = (temp['E_1'] - temp['E_0']) - \
                                (temp['S_0'] - temp['S_1'])
        else:    
            seed_df['incidence'] = 0
        seed_df = seed_df[(seed_df['E'] > 0)|(seed_df['I'] > 0)
                         ].fillna(0)
        seed_df.replace([np.inf, -np.inf], 0, inplace=True)
        # switch moment
        pop = seed_df.iloc[0,:4].sum()
        n_people = pop*perc_switch
        #if idx==0:
        #    print(pop, perc_switch, n_people)

        start_day = choice_start_day.choose_method(chosen_col,seed_df, 
                                                   type_start_day,
                                                   min_day=window_size,
                                                   frac=perc_switch,
                                                   n_people=n_people)
        # ЗА сколько ДО пика
        if not isinstance(type_start_day, str):
            start_day = seed_df[chosen_col].argmax() - start_day

        start_days.append(start_day)
        # choosing the days for prediction
        predicted_days = np.arange(start_day, seed_df.shape[0])
        start_time = time.time()

        # prediction of Beta values and calculation of prediction time
        beggining_beta, predicted_beta, \
            predicted_I = predict_Beta_I.predict_beta(
                            I_prediction_method, seed_df,
                            beta_prediction_method, 
                            predicted_days, stochastic, 
                            count_stoch_line, sigma, gamma,
                            features_reg, model_path, window_size)
        
        predicted_Inc = np.zeros((count_stoch_line+1, 
                                  predicted_days.shape[0]))
        
        # use predicted Beta values for predicting I
        # extract compartment values on the switch day
        y = seed_df.iloc[predicted_days[0],:4]
        # predict the Infected compartment trajectory
        
        ss,ee,predicted_I[0],\
            _ = predict_Beta_I.predict_I(I_prediction_method, y,
                                          predicted_days-start_day,
                                          predicted_beta,
                                          sigma, gamma, 'det')
        # Inc_t = (E_t-1 - E_t) - (S_t - S_t-1)
        if on_incidence:
            # Inc_t = (E_t-1 - E_t) - (S_t - S_t-1)
            from_last = seed_df['incidence'].iloc[predicted_days[0]]
            predicted_Inc[0] = np.append(from_last, 
                                       (ee[:-1] - ee[1:]) - \
                                           (ss[1:] - ss[:-1]))
            predicted_Inc[0][predicted_Inc[0]<0] = 0
            
        if stochastic:
            for i in range(count_stoch_line):
                ss,ee,predicted_I[i+1],_ \
                    = predict_Beta_I.predict_I(I_prediction_method,
                                               y, predicted_days-start_day, 
                                               predicted_beta,sigma, gamma, 
                                               'stoch')
                if on_incidence:
                    predicted_Inc[i+1] = np.append(from_last, 
                                                 (ee[:-1] - ee[1:]) - \
                                                     (ss[1:] - ss[:-1]))
                    predicted_Inc[i+1][predicted_Inc[i+1]<0] = 0

                    
        end_time = time.time()
        execution_time.append(end_time - start_time)

        if ax is None:
            # plot graph for seed_number
            rmse_I, rmse_Inc, rmse_Beta, r2, r2_Inc, \
                peak = plot_one(axes[idx], chosen_col, predicted_days, 
                                seed_df, predicted_I, predicted_Inc,
                                beggining_beta, predicted_beta, 
                                seed_number, end_time - start_time)  
            
            all_rmse_I.append(rmse_I)
            all_rmse_Inc.append(rmse_Inc)
            all_rmse_Beta.append(rmse_Beta)
            all_r2.append(r2)
            all_r2_Inc.append(r2_Inc)
            all_peak.append(peak)

        
    
    if ax is None:

        # show the plots
        if show_fig_flag:
            plt.tight_layout()
            plt.show()
        else:
            plt.close(fig)

        return all_rmse_I, all_rmse_Inc, all_rmse_Beta, \
                    all_r2, all_r2_Inc, all_peak, \
                    execution_time, start_days
    else:
        return plot_one(axes, predicted_days, seed_df, 
               predicted_I, beggining_beta, predicted_beta, 
               seed_number, end_time - start_time)

