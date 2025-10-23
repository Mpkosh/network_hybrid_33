pop =  N/100
E0 = 0
I0 = 1000
R0 = int(0.0*pop)
S0 = int(pop - E0 - I0 - R0)
y0 = np.array([float(S0), float(E0), float(I0), float(R0)])/pop
print(y0)

yy = np.array([*[0.]*35, 
      *ydata])/pop
print(len(yy),  np.arange(len(yy)).shape)
popt, pcov = optimize.curve_fit(fit_odeint_c, np.arange(len(yy)), yy,
                               bounds=((0,1/5,1/12),(1,1/3,1/8)),
                                #bounds=((0,0,0),(1,1,1)),
                                #method = 'trf',
                               #x_scale = [1, 1e5, 1e5]
                               )

fitted_c = fit_odeint_c(np.arange(len(yy)), popt[0], popt[1], popt[2])



def seir_model0(y, t,  beta, alpha, gamma):
    """
    SEIR differential equation model
    """
    S, E, I, R = y
    dSdt = -beta * S * I
    dEdt = beta * S * I  - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return [dSdt, dEdt, dIdt, dRdt]


def fit_odeint_c(x, beta, alpha, gamma):
    rr = integrate.odeint(seir_model0, y0, x, args=(beta, alpha, gamma))
    inc = [*(rr[:,0][:-1] -  rr[:,0][1:]),0]
    #inc = [*((rr[:,1][:-1] -  rr[:,1][1:]) - (rr[:,0][1:] -  rr[:,0][:-1])),0]
    #inc = [*(alpha *  rr[:,1][:-1]),0]
    return np.array(inc).astype('float64')

