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
    price : float  - Prix de l'option, r_A, sigma_A, M1, M2
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


def M1_discrete(r, delta_t, N):
    e_r  = np.exp(r * delta_t)
    e_rN = np.exp(r * N * delta_t)
    return (1 / N) * e_r * (1 - e_rN) / (1 - e_r)

def M2_discrete(r, sigma, delta_t, N):
    s2 = sigma**2
    dt = delta_t
 
    # Raccourcis exponentiels
    e_2rs2   = np.exp((2*r + s2) * dt)
    e_2rs2_N = np.exp((2*r + s2) * N  * dt)
    e_r      = np.exp(r * dt)
    e_2rs2_N1= np.exp((2*r + s2) * (N-1) * dt)
    e_N1_rs2 = np.exp(((N+1)*r + s2) * dt)
    e_rs2_N1 = np.exp((r + s2) * (N-1) * dt)
 
    terme1 = (1 / N**2) * e_2rs2 * (1 - e_2rs2_N) / (1 - e_2rs2)
 
    facteur = (2 * e_r) / (1 - e_r)
 
    A = e_2rs2 * (1 - e_2rs2_N1) / (1 - e_2rs2)
    B = e_N1_rs2 * (1 - e_rs2_N1) / (1 - np.exp((r + s2) * dt))
 
    terme2 = (1 / N**2) * facteur * (A - B)
 
    return terme1 + terme2
 

def turnbull_wakeman_call_discrete(S0, K, r, sigma, T, N):
    
    delta_t = T / N

    #Calcul de M1 et M2
    M1 = M1_discrete(r, delta_t, N)
    
    M2 = M2_discrete(r, sigma, delta_t, N)

    #Expression de rA et sigma_A
    r_A = np.log(M1) / T
    sigma_A = np.sqrt(np.log(M2)/T - 2*r_A)

    #Formule de Black Scholes du prix
    d1 = (np.log(S0/K) + (r_A + 0.5*sigma_A**2)*T) / (sigma_A*np.sqrt(T))
    d2 = d1 - sigma_A*np.sqrt(T)

    price = S0 * np.exp((r_A - r)*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    return price, r_A, sigma_A, M1, M2



"""
def monte_carlo_asian_call(S0, K, r, sigma, T, N, n_simul=200_000, seed=42):

    #Prix Monte Carlo d'un Call Asiatique discret.
    #Moyenne arithmétique sur N dates équidistantes.
 
    #Retourne : prix, intervalle de confiance 95%

    rng     = np.random.default_rng(seed)
    dt      = T / N
 
    # Simulation vectorisée : (n_simul, N) increments browniens
    Z = rng.standard_normal((n_simul, N))
    # Log-rendements incrémentaux
    log_increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    # Trajectoires cumulées
    log_paths = np.cumsum(log_increments, axis=1)
    S_paths   = S0 * np.exp(log_paths)          # shape (n_simul, N)
 
    # Moyenne arithmétique sur les N dates
    S_mean  = S_paths.mean(axis=1)              # shape (n_simul,)
 
    # Payoff actualisé
    payoffs = np.exp(-r * T) * np.maximum(S_mean - K, 0)
 
    price = payoffs.mean()
    std   = payoffs.std() / np.sqrt(n_simul)
    ic95  = 1.96 * std
 
    return price, ic95
"""