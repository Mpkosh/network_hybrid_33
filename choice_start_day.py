from sklearn.preprocessing import MinMaxScaler
import numpy as np


def choose_method(chosen_col, seed_df, start_day, frac=0.01, min_day=14,
                 n_people=1000):
    if start_day == 'fraq_people':
        start_day_v = cpoint_fraq_people(chosen_col, seed_df, 
                                         n_people, min_day=min_day)
    elif start_day == 'roll_var_npeople':
        start_day_v = cpoint_roll_var_npeople(chosen_col, seed_df, 
                                              min_day=min_day, 
                                              n_people=n_people)
    else:
        start_day_v = start_day
    
    return start_day_v     


def cpoint_fraq_people(chosen_col, seed_df, n_people,
                      min_day=4):
    
    switch = seed_df[seed_df[chosen_col]>n_people]
    
    # if there's such a day
    if switch.shape[0]:
        # if it's later than min_day
        if switch.index[0] > min_day:
            return switch.index[0]
        else:
            return min_day
    else:
        return 0
    

# wait until 1% of population is infected, 
# and only then look for a change in variance
def cpoint_roll_var_npeople(chosen_col, seed_df, thresh = 0.1, 
                            n_people=100, min_day=10):
    scaler = MinMaxScaler()

    var_vals = seed_df.Beta.rolling(7).var()
    scaled_varv = scaler.fit_transform(var_vals.values.reshape(-1, 1))
    
    day_with_npeople = seed_df[seed_df[chosen_col] >= n_people].index[0]
    cpoint = np.nanmin(np.where(scaled_varv[day_with_npeople:] < thresh)[0])   
    if cpoint + day_with_npeople < min_day:
        return min_day
    else:
        return cpoint + day_with_npeople