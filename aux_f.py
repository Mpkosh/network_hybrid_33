import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

from matplotlib.colors import LinearSegmentedColormap

from scipy.spatial import ConvexHull
import seaborn as sns



def get_mnames():
    clean_mnames = [['Last value',
        'Cumulative Average','Median'], 
        ['Regression',    
        'LSTM'
        ]]
    methods = [['last_value',
        'expanding_mean_last_value','median_beta'
                  ], 
        ['regression_beta',       
        'lstm_day_E_previous_I'
        ]]
    return clean_mnames, methods


def create_boxplots(folder='', switch='', suff='', 
                    with_inc=False, trim=False,
                    save_fig=False, figsize=(8,4),
                   cut_lim=False):
    
    clean_mnames, methods = get_mnames()
    fig = plt.figure(figsize=figsize) 
    gs = gridspec.GridSpec(1, 2, width_ratios=[3,2]) 

    ax_list = []    
    max_list = []
    
    if with_inc:
        metric = 'rmse_Inc'
    else:
        metric = 'rmse_I'
        
    for i in range(len(methods)):
        rmse_df = pd.DataFrame()
        
        for method, label in zip(methods[i], clean_mnames[i]):
            try:
                # loading data from CSV
                df = pd.read_csv(f'results/{folder}/'+switch+\
                                 f'/{method}_results{suff}.csv')
                if trim:
                    df = df[df['actual_peak_I']>1000]
                
                rmse_df[f"{label}"] = df[metric]
                print(f'Mean RMSE for {label}',
                      df[metric].mean().round(2), '+-',
                     df[metric].std().round(2))
            
            except FileNotFoundError:
                rmse_df[f"{label}"
                       ] = 0#pd.DataFrame([0]*n_seeds)
                print(f'---- No data for {label} ----')
                pass
        q_max = (rmse_df.quantile(.75) + 1.5*\
                    (rmse_df.quantile(.75) - rmse_df.quantile(.25)
                     )
                )
        m_max = rmse_df.quantile(1)
        # sometimes q3 + 1.5*iqr is larger the the max val
        # => we find the min among them
        largest_v = np.min([q_max.values, m_max.values], 0).max()
         
        max_list.append(largest_v)
        # creating a boxplot
        ax = plt.subplot(gs[i])
        box = ax.boxplot(rmse_df[clean_mnames[i]], 
                         showfliers=True, 
                          medianprops=dict(color='OrangeRed',
                                           linewidth=1.5), 
                          widths=0.5, patch_artist=True)
        
        median_c = (1.0, 0.7, 0.7, 0.2)
        for n, patch in enumerate(box['boxes']):
            patch.set(facecolor=median_c, linewidth=1) 

        if i==0:
            ax.set_ylabel('RMSE')
        
        ax.set_xticks(ticks=np.arange(1, len(methods[i])+1), 
                      labels=clean_mnames[i], rotation=30, ha='right')
        ax.grid()  
        ax_list.append(ax)

    plt.tight_layout()
    
    if cut_lim:
        for ax in ax_list:
            ax.set_ylim(-500, np.max(max_list)*1.1) 
    if save_fig:
        plt.savefig(f'results/{folder}/{switch}'+\
                    f'/{metric}_3{suff}.pdf', 
                    format='pdf', bbox_inches='tight')
        
    return rmse_df
        
        
def create_peak_plot(folder = 'new_ba_10k', switch = 'roll_var_npeople',
                    with_outliers=True, same_lims=False, suff='', figsize=(8,4)):
    fig, axes = plt.subplots(1,2, figsize=figsize)
    axes = axes.flatten()

    x_lim = (-130, 20)
    y_lim = (0.6, 2.3)
    size_m=120
    alpha_m=0.4
    alpha_area=0.35
    
    ax = axes[0]

    all_methods = [['Last value',
        #'Cumulative Average',
                    'Median'], 

        ['Regression',    
        'LSTM'
        ]]
    all_new_labels = [['last_value',#'rolling_mean_last_value',
        #'expanding_mean_last_value',
                       'median_beta'
                  ], 
        ['regression_beta',       
        'lstm_day_E_previous_I'
        ]]
    
    
    ymin, ymax, xmin, xmax = 100, -100, 100, -100          
    for i, sub_methods, sub_labels in zip(np.arange(len(all_methods)), 
                                          all_methods, all_new_labels):
        plot_peaks_ax(axes[i], sub_methods, sub_labels, folder, switch,
                     x_lim, y_lim, size_m, alpha_m, alpha_area,
                     with_outliers, suff)
        if same_lims:
            ymin = np.min([ymin, axes[i].get_ylim()[0]])
            ymax = np.max([ymax, axes[i].get_ylim()[1]])
            xmin = np.min([xmin, axes[i].get_xlim()[0]])
            xmax = np.max([xmax, axes[i].get_xlim()[1]])
            
    if same_lims:     
        #pad = 0.2
        for i in np.arange(len(all_methods)):
            #if x_min < 0:
            #axes[i].set_xlim(xmin*(1-pad), xmax*(1+pad))
            #axes[i].set_ylim(ymin*(1-pad), ymax*(1+pad))
            axes[i].set_xlim(xmin, xmax)
            axes[i].set_ylim(ymin, ymax)
    
    axes[0].set_ylabel('Peak height ratio')
    plt.tight_layout()
    plt.savefig(f'results/{folder}/{switch}/peaks_area.pdf', 
                format='pdf', bbox_inches='tight')
    
    
def find_outliers(vals):
    iqr = np.quantile(vals, .75) - np.quantile(vals, .25)
    l_out = np.quantile(vals, .25) - 1.5 * iqr
    h_out = np.quantile(vals, .75) + 1.5 * iqr
    
    if l_out != h_out:
        vals_idx = vals[(l_out<vals)&(vals<h_out)].index
    else:
        vals_idx = vals.index
        
    return vals_idx, l_out, h_out


def plot_peaks_ax(ax, methods, new_labels, folder, switch,
                 x_lim = (-130, 20), y_lim = (0.6, 2.3), size_m=120,
                  alpha_m=0.4, alpha_area=0.35, with_outliers=True, suff=''):
    ax.axvline(x=0, color='black', linestyle='--', 
               linewidth=1)
    ax.axhline(y=1, color='black', linestyle='--', 
               linewidth=1)
    
    cmap = mpl.colormaps['Set2']
    colors_l = cmap(np.linspace(0, 1, 8))
    colors = list(colors_l)[:len(methods)]
    
    for name, method in zip(new_labels, methods):
        try:
            p_df = pd.read_csv(f'results/{folder}/{switch}/{name}_results{suff}.csv')
            #df = df[df[type_start_day]!=0]
            p_df = p_df[p_df['actual_peak_Inc']>10]
            pt = p_df['predicted_peak_day_inc'] - p_df['actual_peak_day_Inc']
            ph = p_df['predicted_peak_inc']/p_df['actual_peak_Inc']
            
            if not with_outliers:
                ph_idx, l_ph, h_ph = find_outliers(ph)
                pt_idx, l_pt, h_pt = find_outliers(pt)
                clean = list(set(ph_idx).intersection(pt_idx))
                ph = ph.loc[clean]
                pt = pt.loc[clean]

            hull = ConvexHull(pd.concat([pt,ph], axis=1))

            col = colors.pop()
            ax.scatter(pt,ph, marker='.', s=size_m,  alpha=alpha_m, 
                       label=method, zorder=10, color=col)

            ax.fill(pt.iloc[hull.vertices], 
                     ph.iloc[hull.vertices], alpha=alpha_area, color=col)
            
        except FileNotFoundError:
            pt = pd.Series([0]*301)
            ph = pd.Series([0]*301)
            print(f'---- No data for {name} ----')

    ax.grid()
    ax.set_xlabel('Peak time difference')
    #ax.set_xlim(x_lim)
    #ax.set_ylim(y_lim)

    leg = ax.legend(prop={'size': 13}, loc='best')
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    leg.set_zorder(20)
    

def df_metrics(folder_name, test_suff='', switch='',
              with_inc=False, trim=False, suff=''):
    methods = ['last_value',
                'expanding_mean_last_value','median_beta', 
                'regression_beta','lstm_day_E_previous_I'
              ]

    sw = pd.read_csv(f'{test_suff}test_files.csv').values[::10]
    sww = pd.Series(sw.flatten()).str.split('_',expand=True)
    sww.columns = ['p','beta','gamma','delta','initi',
                   'alpha','seed','nseed']
    fin = sww[['beta','alpha']].astype(float).round(2)
    fin_m = ['r2','rmse_I','rmse_Beta',
             'pt_err', 'ph_err','time_predict']
    if with_inc:
        fin_m = ['r2','r2_Inc','r2_full','r2_Inc_full',
                 'rmse_I','rmse_Inc',
                'rmse_Beta', 'pt_err', 'ph_err',
                'pt_err_Inc','ph_err_Inc',
                'time_predict']
        
    for label in methods:
        try:
            df = pd.read_csv(f'results/{folder_name}/{switch}/'+\
                             f'{label}_results{suff}.csv')

            if trim:
                df = df[df[switch]!=0]

            df['pt_err'] = df['predicted_peak_day'
                             ] - df['actual_peak_day']
            df['ph_err'] =  df['predicted_peak_I'
                              ]/df['actual_peak_I']
            if with_inc:
                df['pt_err_Inc'] = df['predicted_peak_day_inc'
                                    ] - df['actual_peak_day_Inc']
                df['ph_err_Inc'] =  df['predicted_peak_inc'
                                    ]/df['actual_peak_Inc']

            for met in fin_m:
                fin[f'{met}.{label}'] = df[met]
        except FileNotFoundError:
            pass    
    fin['switch'] = df[switch]      
    fin['days_before_peak'] = df['actual_peak_day']-df[switch]      
    fin['actual_peak_I'] = df['actual_peak_I']
    fin['actual_peak_day'] = df['actual_peak_day']  
    
    if with_inc:
        fin['actual_peak_Inc'] = df['actual_peak_Inc']
        fin['actual_peak_day_Inc'] = df['actual_peak_day_Inc']    
        
    return fin


def flatten(xss):
    return [x for xs in xss for x in xs]

# Andrew's
def nonlinear_norm(x):
    # Быстрый рост от 0 до 0.8 (линейный)
    # Плавный переход от 0.8 до 0.95 (квадратный корень)
    # Очень медленный рост от 0.95 до 1 (логарифмический)
    return x**4


def metric_hmaps(fin, met, suff='', exclude=[]):
    clean_mnames, methods = get_mnames()
    fig = plt.figure(figsize=(15,10))
    gs = gridspec.GridSpec(5, 3) 
    n = ['a)','b)','c)','d)','e)'][::-1]
    
    nice_label = ''
    if 'r2' in met:
        nice_label=r'$R^2$'
    
    cm = plt.cm.RdYlGn
    colors = cm(np.linspace(0, 1, 256))
    new_colors = colors[(nonlinear_norm(np.linspace(0, 1, 256)
                                       ) * 255).astype(int)]
    nonlinear_cmap = LinearSegmentedColormap.from_list('nonlinear_plasma', 
                                                       new_colors)
    
    #rows = [0,2,0,2,1][::-1]
    rows = [0,0,2,2,1][::-1]
    #cols = [0,0,1,1,2][::-1]
    cols = [0,1,0,1,2][::-1]
    
    
    for method, label in zip(flatten(methods),
                             flatten(clean_mnames)):
        if label not in exclude:
            try:
                data = fin.pivot(columns='beta', index='alpha', 
                                 values=f'{met}.{method}')
                r = rows.pop()
                c = cols.pop()
                ax_i = plt.subplot(gs[r:r+2, c])
                sns.heatmap(data.sort_index(level=1, 
                                            ascending=False), 
                            vmin=0, vmax=1,cmap=nonlinear_cmap,
                            ax=ax_i,
                            yticklabels = 10, xticklabels=10,
                            linewidths=0.0, rasterized=True,
                            #cbar_kws={'label': nice_label}
                           )
                ax_i.collections[0].cmap.set_bad('0.7')
                ax_i.set_xlabel(r'$\beta_n$')
                ax_i.set_ylabel(r'$\alpha$')

                ax_i.set_title(label)
                ax_i.text(-0.1, 1.1, n.pop(),
                          transform=ax_i.transAxes, size=15)
                cbar = ax_i.collections[0].colorbar
                cbar.set_label(nice_label, rotation=0)

            except KeyError:
                pass
        
    
    
    plt.tight_layout()
    plt.savefig(f'results/hmap{suff}.pdf', format='pdf', 
                bbox_inches='tight')
    
    
def peaks_hmaps(fin, ax=[], n=['a)','b)'], 
                with_inc=False, title=''):
    fontsize = 14
    if len(ax)==0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        ax=axes.flatten()
    
    cmap = mpl.cm.RdYlGn
    n = n[::-1]
    
    if with_inc:
        suff = '_Inc'
    else:
        suff = ''
        
    
    #bounds = [0, 4, 20, 200]
    #norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    data = fin.pivot(columns='beta', index='alpha', 
                     values=f'actual_peak_day{suff}')
    ax_1 = sns.heatmap(data.sort_index(level=1, ascending=False), 
                       cmap=cmap, ax=ax[0], #norm=norm, 
                       cbar_kws={'extendfrac': .1},
                       #vmax=70,
                      xticklabels = 10, yticklabels=10,
                      linewidths=0.0, rasterized=True,)
    ax_1.set_title('Peak time'+title, 
                   fontsize=1.2*fontsize)
    '''
    colorbar = ax_1.collections[0].colorbar
    tick_locs = np.linspace(bounds[0], bounds[-1], 
                            2 * len(bounds) + 1)[1::2]
    colorbar.set_ticks(np.mean([bounds[1:], bounds[:-1]], 0))
    colorbar.set_ticklabels([f'[1, {bounds[1]})', 
                             f'[{bounds[1]}, {bounds[2]-1})',
                             f'[{bounds[2]}, 150)'])
    
    '''
    
    data = fin.pivot(columns='beta', index='alpha', 
                     values=f'actual_peak_I{suff[2:]}')/100000
    ax_2 = sns.heatmap(data.sort_index(level=1, ascending=False), 
                       cmap=cmap, ax=ax[1], #norm=norm, 
                       cbar_kws={'extendfrac': .1},
                      xticklabels = 10, yticklabels=10,
                      linewidths=0.0, rasterized=True,)
    ax_2.set_title('Peak height'+title,
                   fontsize=1.2*fontsize)

    
    for ax_i in [ax_1, ax_2]:
        ax_i.text(-0.1, 1.1, n.pop(),
                  transform=ax_i.transAxes, size=1.5*fontsize)
        ax_i.collections[0].cmap.set_bad('0.7')
        ax_i.set_xlabel(r'$\beta_n$', fontsize=1.2*fontsize)
        ax_i.set_ylabel(r'$\alpha$', fontsize=1.2*fontsize)
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize)
    for i in [-1,-2]:    
        ax_1.figure.axes[i].tick_params(labelsize=fontsize)
        
    ax_1.figure.axes[-1].set_ylabel('Fraction of new cases', size=fontsize)
    ax_1.figure.axes[-2].set_ylabel('Day', size=fontsize)
    
    plt.tight_layout()
    #plt.savefig(f'results/actual.pdf', format='pdf', bbox_inches='tight')
    
    
def smth_hmaps(fin):
    fontsize = 14
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    ax=axes.flatten()

    cmap = mpl.cm.RdYlGn
    n = ['a)','b)'][::-1]

    #ticks=np.arange(1,22)
    #boundaries = np.arange(1-.5, 21+1.5 )

    data = fin.pivot(columns='beta', index='alpha', 
                     values=f'days_before_peak')
    ax_1 = sns.heatmap(data.sort_index(level=1, ascending=False), 
                       cmap=cmap, ax=ax[0], #norm=norm, 
                       cbar_kws={'extendfrac': .1,
                                #"ticks":ticks, "boundaries":boundaries
                                },
                       vmax=14,
                      xticklabels = 10, yticklabels=10,
                      linewidths=0.0, rasterized=True,)
    ax_1.set_title('Days from switch to peak', fontsize=1.2*fontsize)

    colorbar = ax_1.collections[0].colorbar
    '''
    tick_locs = np.linspace(bounds[0], bounds[-1], 
                            2 * len(bounds) + 1)[1::2]
    colorbar.set_ticks(np.mean([bounds[1:], bounds[:-1]], 0))
    colorbar.set_ticklabels([f'[1, {bounds[1]})', 
                             f'[{bounds[1]}, {bounds[2]-1})',
                             f'[{bounds[2]}, 150)'])
    '''
    data = fin.pivot(columns='beta', index='alpha', 
                     values=f'switch')
    ax_2 = sns.heatmap(data.sort_index(level=1, ascending=False), 
                       cmap=cmap, ax=ax[1], #norm=norm, 
                       cbar_kws={'extendfrac': .1},
                       vmax=14,
                      xticklabels = 10, yticklabels=10,
                      linewidths=0.0, rasterized=True,)
    ax_2.set_title('Day of switch', fontsize=1.2*fontsize)

    colorbar = ax_2.collections[0].colorbar
    '''
    tick_locs = np.linspace(bounds[0], bounds[-1], 
                            2 * len(bounds) + 1)[1::2]
    colorbar.set_ticks(np.mean([bounds[1:], bounds[:-1]], 0))
    
    colorbar.set_ticklabels([r'(0, 1%)', 
                             r'[1%, 5%)', 
                             f'[5%, 10%)', 
                             '[10%, 100%)'])
    '''
    
    for ax_i in [ax_1, ax_2]:
        ax_i.text(-0.1, 1.1, n.pop(),
                  transform=ax_i.transAxes, size=1.5*fontsize)
        ax_i.collections[0].cmap.set_bad('0.7')
        ax_i.set_xlabel(r'$\beta_n$', fontsize=1.2*fontsize)
        ax_i.set_ylabel(r'$\alpha$', fontsize=1.2*fontsize)
        ax_i.tick_params(axis='both', which='major', labelsize=fontsize)
    for i in [-1,-2]:    
        ax_1.figure.axes[i].tick_params(labelsize=fontsize)
        
    ax_1.figure.axes[-1].set_ylabel('Day', size=fontsize)
    ax_1.figure.axes[-2].set_ylabel('Day', size=fontsize)
    
    plt.tight_layout()
    #plt.savefig(f'results/actual.pdf', format='pdf', bbox_inches='tight')
    
    
