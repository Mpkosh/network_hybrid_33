# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 17:52:44 2025

@author: MKoshkareva
"""

import networkx as nx 
import numpy as np
import EoN
import pandas as pd
import random


def SEIR_network(G, tau, alpha, gamma, rho, tmax):
    # Initialize states: S=0 (Susceptible), E=1 (Exposed), I=2 (Infected), R=3 (Recovered)
    for node in G.nodes():
        G.nodes[node]['state'] = 0  # Start all nodes as susceptible

    initial_infected = int(rho*len(G.nodes()))
    initial_infected_nodes = random.sample(list(G.nodes()), initial_infected)
    for node in initial_infected_nodes:
        G.nodes[node]['state'] = 2

    susceptible_count = []
    exposed_count = []
    infected_count = []
    recovered_count = []

    for day in range(tmax + 1):
        new_states = {}

        # Count current states
        susceptible = sum(1 for n in G.nodes if G.nodes[n]['state'] == 0)
        exposed = sum(1 for n in G.nodes if G.nodes[n]['state'] == 1)
        infected = sum(1 for n in G.nodes if G.nodes[n]['state'] == 2)
        recovered = sum(1 for n in G.nodes if G.nodes[n]['state'] == 3)

        susceptible_count.append(susceptible)
        exposed_count.append(exposed)
        infected_count.append(infected)
        recovered_count.append(recovered)

        for node in G.nodes():
            if G.nodes[node]['state'] == 2:
                for neighbor in G.neighbors(node):
                    if G.nodes[neighbor]['state'] == 0:
                        if random.random() < tau:
                            new_states[neighbor] = 1
                if random.random() < gamma:
                    new_states[node] = 3

            elif G.nodes[node]['state'] == 1:
                if random.random() < alpha:
                    new_states[node] = 2

        for node, new_state in new_states.items():
            G.nodes[node]['state'] = new_state

    return [index for index in range(tmax + 1)], susceptible_count, \
                exposed_count, infected_count, recovered_count


def save_traj(t, S, E, I, R, top, pop, seed_n):
    S = np.array(S)
    E = np.array(E)
    I = np.array(I)
    R = np.array(R)

    # compute betas values
    beta_values = []
    num_days = len(t)
    for i in range(num_days - 1):
        if S[i] > 0 and I[i] > 0:
            beta = - (S[i + 1] - S[i]) / (S[i] * I[i])
        else:
            beta = np.nan
        beta_values.append(beta)
    beta_values.append(np.nan)
    # store the simulation results in a dataFrame
    df_sim = pd.DataFrame({"S": S, "E": E, "I": I, "R": R, 
                           "Beta": beta_values})
    # save the results to a csv file
    df_sim.to_csv(f'initial_data_{top}_{pop}/seir_seed_{seed_n}.csv', index=False)
            
            
def choose_top(N, s):
    if s=='ba':
        G = nx.barabasi_albert_graph(N, 8) 
    elif s=='sw':
        G = nx.watts_strogatz_graph(N, 8, 0.1)
    elif s=='r':
        # чтобы средняя степень была 8
        G = nx.fast_gnp_random_graph(N, 8/N)
        
    return G
    
            
if __name__ == '__main__':
    
    tmax = 250
    iterations = 50  # run 5 simulations
    tau_boundaries = np.arange(0.04, 0.09, 0.01)         # transmission rate
    gamma = 0.08  # recovery rate
    rho_boundaries = np.arange(0.005, 0.011, 0.001)     # random fraction initially infected
    alpha = 0.1 # latent period rate
    
    for N in [10**4, 10**5, 5*10**4]:
        for top in ['ba','sw','r']:
            G = choose_top(N, top)
            
            seed=0
            
            for tau in tau_boundaries:
                for rho in rho_boundaries:

                    for iter in range(0, iterations):
                        # SEIR
                        t, S, E, I, R = SEIR_network(G, tau, alpha, gamma, rho, tmax)
                        save_traj(t, S, E, I, R, top, str(N), seed)
                        seed += 1