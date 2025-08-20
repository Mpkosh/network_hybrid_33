import pandas as pd
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from scipy.optimize import curve_fit
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# our functions
import seir_discrete 

import warnings
warnings.filterwarnings(action='ignore')


def load_saved_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return joblib.load(model_path)


class LSTMPredictor:
    """
    Wraps the trained LSTM model to predict beta on a rolling window of
    [day, E] (2 features). 
    The model was trained to predict normalized log_beta, so this class
    denormalizes the prediction and returns beta.
    """
    def __init__(self, model, full_scaler, window_size):
        self.model = model
        self.n_feats = 1
        # Create a scaler for input features
        # Corrected feature_indices calculation:
        feature_indices = list(range(self.n_feats))
        self.input_scaler = StandardScaler()
        self.input_scaler.mean_ = full_scaler.mean_[feature_indices]
        self.input_scaler.scale_ = full_scaler.scale_[feature_indices]
        self.input_scaler.var_ = full_scaler.var_[feature_indices]
        self.input_scaler.n_features_in_ = self.n_feats
        self.window_size = window_size
        self.buffer = []
        # Store target parameters for log_beta (7th column)
        self.target_mean = full_scaler.mean_[-1]
        self.target_scale = full_scaler.scale_[-1]
        
    def update_buffer(self, new_data):
        # new_data should be a list with 3 elements: [day, E, prev_I]
        self.buffer.append(new_data)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)
            
    def predict_next(self):
        # Ensure the buffer has window_size rows
        if len(self.buffer) < self.window_size:
            padded = np.zeros((self.window_size, self.n_feats))
            padded[-len(self.buffer):] = self.buffer
        else:
            padded = np.array(self.buffer[-self.window_size:])
            
        scaled = self.input_scaler.transform(padded)
        scaled_window = scaled.reshape(1, self.window_size, self.n_feats)
        normalized_pred = self.model.predict(scaled_window, verbose=0)[0][0]
        # Denormalize to obtain the raw log_beta
        raw_log_beta = normalized_pred * self.target_scale + self.target_mean
        # Compute beta by exponentiating the log_beta
        predicted_beta = np.exp(raw_log_beta)
        return predicted_beta

    
def predict_beta(I_prediction_method, seed_df, beta_prediction_method, predicted_days, 
                 stochastic, count_stoch_line, sigma, gamma, 
                 features_reg='', model_path='', window_size=14):
    
    '''
    Predict Beta values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting Infected trajectories
        ['seir']
    - seed_df -- DataFrame of seed, created by a regular network
    - beta_prediction_method -- method for predicting Beta values
    - predicted_days -- days for prediction
    - stochastic -- indicator of the presence of predicted trajectories by a stochastic mathematical model
    - count_stoch_line -- number of trajectories predicted by the stochastic mathematical model
    - sigma -- parameter of the SEIR-type mathematical model
    - gamma -- parameter of the SEIR-type mathematical model
    '''
    predicted_I = np.zeros((count_stoch_line+1, predicted_days.shape[0]))
    beggining_beta = []

    
    if beta_prediction_method == 'last value':
        predicted_beta = [seed_df.iloc[predicted_days[0]]['Beta'] 
                          for i in range(predicted_days.shape[0])]

    
    elif beta_prediction_method == 'expanding mean last value':
        beggining_beta = seed_df.Beta.iloc[:predicted_days[0]
                                          ].expanding(1).mean()
        last = beggining_beta.iloc[-1]
        predicted_beta = [last for i in range(predicted_days.shape[0])
                         ]


    elif beta_prediction_method == 'median beta':
        betas = pd.read_csv(model_path) #'train/median_beta.csv'
        beggining_beta = betas.iloc[:predicted_days[0],-1].values
        predicted_beta = betas.iloc[predicted_days[0]:,-1].values


    elif beta_prediction_method == 'regression (day)':
        #model_path = 'regression_day_for_seir.joblib'
        model = load_saved_model(model_path)
        x_test = np.arange(0,predicted_days[0]).reshape(-1, 1)
        beggining_beta = np.exp(model.predict(x_test))
        x_test = np.arange(predicted_days[0], seed_df.shape[0]).reshape(-1, 1)
        predicted_beta = np.exp(model.predict(x_test))
    
    
    elif beta_prediction_method == 'regression beta':
        #model_path = 'regression_day_for_seir.joblib'
        model = load_saved_model(model_path)
        ws = 4
        input_b = seed_df[['Beta']].shift(np.arange(1,ws+1)
                                         ).iloc[predicted_days[0]]
        input_b = np.log(input_b + 1e-7).values.reshape(1, -1)
        
        predicted_beta = []
        for day in range(predicted_days[0], seed_df.shape[0]):
            pred = model.predict(input_b)
            predicted_beta.append(np.exp(pred[0]))
            input_b = np.array([*pred,*input_b.flatten()[:-1]]).reshape(1, -1)
    
    
    elif beta_prediction_method == 'arimax':
        #model_path = 'regression_day_for_seir.joblib'
        model = SARIMAXResults.load(model_path)
        to_pred = np.arange(predicted_days[0], seed_df.shape[0])
        our_exog = seed_df['day'].iloc[predicted_days[0], 
                                       seed_df.shape[0]]
        
        beggining_beta = model.predict(0, predicted_days[0],
                                       exog=our_exog)
        beggining_beta[beggining_beta<np.log(1e-7)] = np.log(1e-7)
        beggining_beta = np.exp(beggining_beta)
        
        predicted_beta = model.predict(predicted_days[0], 
                                       seed_df.shape[0],
                                       exog=our_exog)
        predicted_beta[predicted_beta<np.log(1e-7)] = np.log(1e-7)
        predicted_beta = np.exp(predicted_beta)
    
    
    elif beta_prediction_method == 'lstm':
        full_scaler = joblib.load(f'{model_path}.pkl')
        model = load_model(f'{model_path}.keras')
        window_size=4
        predictor = LSTMPredictor(model, full_scaler, 
                                  window_size=window_size)
        inp = seed_df[['Beta']
                     ].shift(np.arange(window_size)
                            ).iloc[predicted_days[0]].values
        inp = np.log(inp+1e-7)
        
        for i in inp[::-1]:
            predictor.update_buffer([i])
        
        predicted_beta = []
        for i in range(predicted_days[0], 
                       seed_df.shape[0]):
            pred = predictor.predict_next()
            predicted_beta.append(pred)
            predictor.update_buffer([np.log(pred+1e-7)])
    
            
        
    elif beta_prediction_method == 'lstm (day, E, previous I)':
        full_scaler = joblib.load(f'{model_path}.pkl')
        model = load_model(f'{model_path}.keras')
        predictor = LSTMPredictor(model, full_scaler, 
                                  window_size=window_size)
        '''
        prev_I = seed_df.iloc[predicted_days[0]-2:predicted_days[0]
                             ]['I'].to_numpy(
            ) if predicted_days[0] > 1 else np.array([0.0, 0.0])
        '''
        seed_df['day'] = range(len(seed_df))
        #seed_df['prev_I'] = seed_df['I'].shift(2).fillna(0)
        predicted_beta = np.empty((0,))
        S = np.zeros((count_stoch_line+1, 2))
        E = np.zeros((count_stoch_line+1, 2))
        R = np.zeros((count_stoch_line+1, 2))

        S[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['S']
        predicted_I[0:count_stoch_line+1,
                    0] = seed_df.iloc[predicted_days[0]]['I']
        R[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['R']  
        E[0:count_stoch_line+1,0] = seed_df.iloc[predicted_days[0]]['E']  
        
        pop = seed_df.iloc[0,:4].sum()
        # Initialize predictor buffer using the last 'window_size' days
        for i in range(predicted_days[0] - predictor.window_size + 1, 
                       predicted_days[0] + 1):
            row = seed_df.iloc[i]
            raw_features = [row['day'], row['E']/pop, #row['prev_I']
                           ]
            predictor.update_buffer(raw_features)
        y = np.array([S[:,0], E[:,0], predicted_I[:,0], R[:,0]])
        y = y.T
        
        for idx in range(predicted_days.shape[0]-1):
            predicted_beta = np.append(predicted_beta, predictor.predict_next())     
            #if idx == predicted_days.shape[0]-1:
            #    break      
            # prediction of the Infected compartment trajectory
            S[0,:], E[0,:], predicted_I[0,idx:idx+2], \
                R[0,:] = predict_I(I_prediction_method, y[0], 
                                    predicted_days[idx:idx+2], 
                                    predicted_beta[idx], sigma, gamma, 
                                    'det', beta_t=False)   
            if stochastic:
                for i in range(count_stoch_line):
                    S[i+1,:], E[i+1,:], predicted_I[i+1,idx:idx+2], \
                        R[i+1,:] = predict_I(I_prediction_method,
                                             y[i+1],
                                             predicted_days[idx:idx+2], 
                                             predicted_beta[idx], 
                                             sigma, gamma, 
                                             'stoch', beta_t=False) 
            y = np.array([S[:,1], E[:,1], predicted_I[:,idx+1], R[:,1]])
            y = y.T
            if idx == 0:
                predictor.update_buffer([predicted_days[idx+1], E[0,1]/pop,
                                         #prev_I[1]
                                        ])
            else:
                predictor.update_buffer([predicted_days[idx+1], E[0,1]/pop,
                                         #predicted_I[0,idx-1]
                                        ])
                
    return np.array(beggining_beta), np.array(predicted_beta), predicted_I 


def predict_I(I_prediction_method, y, 
              predicted_days, 
              predicted_beta, sigma, gamma, stype, beta_t=True):
    '''
    Predict Infected values.

    Parameters:

    - I_prediction_method -- mathematical model for predicting the Infected trajectory
        ['seir']
    - y -- compartment values on the day of switching to the mathematical model
    - predicted_days -- days for prediction
    - predicted_beta -- predicted Beta values
    - sigma -- parameter of the SEIR-type mathematical model
    - gamma -- parameter of the SEIR-type mathematical model
    - stype -- type of mathematical model
        ['stoch', 'det']
    '''
    
    
    S,E,I,R = seir_discrete.seir_model(y, predicted_days, 
                        predicted_beta, sigma, gamma, 
                        stype, beta_t).T

    return S,E,I,R