import numpy as np
from scipy.stats import norm


# question 3
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


# question 4
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



# question 5
# Simulation d'UNE trajectoire brownienne
def simuler_trajectoire_brownienne(N, delta_t, seed=None):
    rng = np.random.default_rng(seed)
    # Incréments browniens
    increments = np.sqrt(delta_t) * rng.standard_normal(N)
    # Trajectoire cumulée W(0)=0, W(Δt), ..., W(NΔt)
    W = np.concatenate([[0.0], np.cumsum(increments)])
    return W
 
# Prix de l'actif S(iΔt) à partir de W   
def simuler_trajectoire_S(S0, r, sigma, delta_t, N, W):
    t = np.arange(N + 1) * delta_t                 # [0, Δt, 2Δt, ..., NΔt]
    S = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W)
    return S
 

# Payoff du Call Asiatique discret
def payoff_call_asiatique(S, K, r, T):
    moyenne = S[1:].mean()      # moyenne sur S(Δt), ..., S(NΔt)
    return np.exp(-r * T) * max(moyenne - K, 0.0)
 
# Estimateur Monte Carlo classique            
def monte_carlo_asian_call(S0, K, r, sigma, T, delta_t, n_simul=100_000, seed=42):
    """
    Estimateur Monte Carlo de P^{Δt} pour un Call Asiatique discret.
 
    Pour chaque simulation m = 1..M :
      1. Simuler (W(iΔt))_{i=1..N}
      2. En déduire (S(iΔt))_{i=1..N}
      3. Calculer le payoff actualisé
    Estimateur : moyenne empirique des payoffs
 
    Retourne : prix, écart-type de l'estimateur, IC 95%
    """
    N   = int(T / delta_t)
    rng = np.random.default_rng(seed)
 
    # Simulation vectorisée : (n_simul, N) incréments browniens
    Z = rng.standard_normal((n_simul, N))
    increments  = np.sqrt(delta_t) * Z                          # (M, N)
 
    # Trajectoires browniennes cumulées — W(iΔt) pour i=1..N
    W_paths = np.cumsm(increments, axis=1)                     # (M, N)
 
    # Trajectoires de S(iΔt) pour i=1..N  (on n'a pas besoin de i=0)
    t       = np.arange(1, N + 1) * delta_t                     # (N,)
    S_paths = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W_paths)  # (M, N)
 
    # Moyenne arithmétique sur les N dates pour chaque trajectoire
    S_mean  = S_paths.mean(axis=1)                              # (M,)
 
    # Payoffs actualisés
    payoffs = np.exp(-r * T) * np.maximum(S_mean - K, 0.0)     # (M,)
 
    # Estimateur et statistiques
    prix    = payoffs.mean()
    std_err = payoffs.std() / np.sqrt(n_simul)
    ic95    = 1.96 * std_err
 
    return prix, std_err, ic95
 