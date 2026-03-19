import numpy as np
from scipy.stats import norm
 
def turnbull_wakeman_call(S0, K, r, sigma, T, t0=0.0):
    """
    Prix d'un Call Asiatique par l'approximation Turnbull-Wakeman.
 
    Paramètres
    ----------
    S0    : float  - Prix initial de l'actif
    K     : float  - Strike (prix d'exercice)
    r     : float  - Taux sans risque (continu)
    sigma : float  - Volatilité du sous-jacent
    T     : float  - Maturité (en années)
    t0    : float  - Date de début de la moyenne (t0 <= 0 si déjà commencée)
 
    Retourne
    --------
    price : float  - Prix de l'option
    """
 
    tau = T - t0

    #Calcul de M1 et M2
    M1 = (np.exp(r * T) - np.exp(r * t0)) / (r * tau) # avec t0 = 0
    
    s2 = sigma**2
    term1 = (2 * np.exp((2*r + s2) * T) / ((r + s2) * (2*r + s2) * tau**2))
    term2 = (2 * np.exp((2*r + s2) * t0) / (r * tau**2)) * (1/(2*r+s2) - np.exp(r * tau)/(r+s2))
    M2 = term1 + term2

    #Expression de rA et sigma_A
    r_A = np.log(M1) / T
    sigma_A = np.sqrt(np.log(M2)/T - 2*r_A)

    #Formule de Black Scholes du prix
    d1 = (np.log(S0/K) + (r_A + 0.5*sigma_A**2)*T) / (sigma_A*np.sqrt(T))
    d2 = d1 - sigma_A*np.sqrt(T)

    price = S0 * np.exp((r_A - r)*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    return price, r_A, sigma_A, M1, M2


def turnbull_wakeman_call_discrete(S0, K, r, sigma, T, t0=0.0): 
    tau = T - t0

    #Calcul de M1 et M2
    M1 =  # avec t0 = 0
    
    s2 = sigma**2
    term1 = (2 * np.exp((2*r + s2) * T) / ((r + s2) * (2*r + s2) * tau**2))
    term2 = (2 * np.exp((2*r + s2) * t0) / (r * tau**2)) * (1/(2*r+s2) - np.exp(r * tau)/(r+s2))
    M2 = term1 + term2

    #Expression de rA et sigma_A
    r_A = np.log(M1) / T
    sigma_A = np.sqrt(np.log(M2)/T - 2*r_A)

    #Formule de Black Scholes du prix
    d1 = (np.log(S0/K) + (r_A + 0.5*sigma_A**2)*T) / (sigma_A*np.sqrt(T))
    d2 = d1 - sigma_A*np.sqrt(T)

    price = S0 * np.exp((r_A - r)*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    return price, r_A, sigma_A, M1, M2