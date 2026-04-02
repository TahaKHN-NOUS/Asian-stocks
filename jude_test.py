import numpy as np
from scipy.stats import norm

def M1_discrete(r, dt, N):
    return (1/N) * np.exp(r*dt) * (1 - np.exp(r*N*dt)) / (1 - np.exp(r*dt))

def M2_discrete(r, sigma, dt, N):
    s2 = sigma**2
    a = 2*r + s2
    b = r + s2

    term1 = (1/N**2) * np.exp(a*dt) * (1 - np.exp(a*N*dt)) / (1 - np.exp(a*dt))

    term2 = (1/N**2) * (2*np.exp(r*dt)/(1 - np.exp(r*dt))) * (
        np.exp(a*dt) * (1 - np.exp(a*(N-1)*dt)) / (1 - np.exp(a*dt))
        - np.exp(((N+1)*r + s2)*dt) * (1 - np.exp(b*(N-1)*dt)) / (1 - np.exp(b*dt))
    )

    return term1 + term2

def turnbull_wakeman_call_discrete(S0, K, r, sigma, T, N):
    dt = T / N

    M1 = M1_discrete(r, dt, N)
    M2 = M2_discrete(r, sigma, dt, N)

    r_A = np.log(M1) / T
    sigma_A = np.sqrt(np.log(M2) / T - 2*r_A)

    d1 = (np.log(S0/K) + (r_A + 0.5*sigma_A**2)*T) / (sigma_A*np.sqrt(T))
    d2 = d1 - sigma_A*np.sqrt(T)

    price = S0 * np.exp((r_A - r)*T) * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    return price


# Génération de normales via Box-Muller
def box_muller(n, seed=None):
    rng = np.random.default_rng(seed)
    m = (n + 1) // 2

    U1 = rng.uniform(0.0, 1.0, m)
    U2 = rng.uniform(0.0, 1.0, m)

    U1 = np.maximum(U1, 1e-12)

    R = np.sqrt(-2.0 * np.log(U1))
    Theta = 2.0 * np.pi * U2

    Z1 = R * np.cos(Theta)
    Z2 = R * np.sin(Theta)

    Z = np.empty(2 * m)
    Z[0::2] = Z1
    Z[1::2] = Z2

    return Z[:n]


def monte_carlo_asian_call(S0, K, r, sigma, T, delta_t, n_simul=100_000, seed=42):
    N = int(T / delta_t)

    # Génération des gaussiennes via Box-Muller
    Z = box_muller(n_simul * N, seed=seed).reshape(n_simul, N)

    # Incréments browniens
    increments = np.sqrt(delta_t) * Z

    # Trajectoires browniennes
    W_paths = np.cumsum(increments, axis=1)

    # Temps
    t = np.arange(1, N + 1) * delta_t

    # Trajectoires du sous-jacent
    S_paths = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W_paths)

    # Moyenne arithmétique
    S_mean = S_paths.mean(axis=1)

    # Payoffs actualisés
    payoffs = np.exp(-r * T) * np.maximum(S_mean - K, 0.0)

    # Estimateur MC
    prix = payoffs.mean()
    std_err = payoffs.std(ddof=1) / np.sqrt(n_simul)
    ic95 = 1.96 * std_err

    return prix, std_err, ic95
def geometric_asian_call_closed_form(S0, K, r, sigma, T, N):
    sigma_E = sigma * np.sqrt((N + 1) * (2 * N + 1) / (6 * N**2))
    r_E = (r - 0.5 * sigma**2) * (N + 1) / (2 * N) + 0.5 * sigma_E**2

    d1 = (np.log(S0 / K) + (r_E + 0.5 * sigma_E**2) * T) / (sigma_E * np.sqrt(T))
    d2 = d1 - sigma_E * np.sqrt(T)

    price = S0 * np.exp((r_E - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price, r_E, sigma_E
def monte_carlo_asian_call_control_variate(S0, K, r, sigma, T, delta_t, n_simul=100_000, seed=42):
    N = int(T / delta_t)

    # Génération des normales par Box-Muller
    Z = box_muller(n_simul * N, seed=seed).reshape(n_simul, N)

    # Incréments browniens
    increments = np.sqrt(delta_t) * Z

    # Trajectoires browniennes
    W_paths = np.cumsum(increments, axis=1)

    # Dates t1,...,tN
    t = np.arange(1, N + 1) * delta_t

    # Trajectoires de l'actif
    S_paths = S0 * np.exp((r - 0.5 * sigma**2) * t + sigma * W_paths)

    # -----------------------------------------------------
    # Variable X : payoff arithmétique actualisé
    # -----------------------------------------------------
    A_mean = S_paths.mean(axis=1)
    X = np.exp(-r * T) * np.maximum(A_mean - K, 0.0)

    # -----------------------------------------------------
    # Variable Y : payoff géométrique actualisé
    # -----------------------------------------------------
    G_mean = np.exp(np.mean(np.log(S_paths), axis=1))
    Y = np.exp(-r * T) * np.maximum(G_mean - K, 0.0)

    # Espérance connue de Y grâce à la Q6
    EY, _, _ = geometric_asian_call_closed_form(S0, K, r, sigma, T, N)

    # Coefficient optimal b*
    cov_XY = np.cov(X, Y, ddof=1)[0, 1]
    var_Y = np.var(Y, ddof=1)
    b_opt = cov_XY / var_Y

    # Estimateur corrigé
    Z_ctrl = X - b_opt * (Y - EY)

    # Prix estimé
    prix = Z_ctrl.mean()

    # Erreur standard
    std_err = Z_ctrl.std(ddof=1) / np.sqrt(n_simul)

    # IC asymptotique à 90%
    ic90 = 1.645 * std_err

    return prix, std_err, ic90, b_opt
# Paramètres du sujet
S0 = 1
K = 1
r = 0.01
sigma = 0.3
T = 0.5
delta_t = 1/252

# Simulation Monte Carlo
prix_mc, std_err, ic95 = monte_carlo_asian_call(
    S0, K, r, sigma, T, delta_t,
    n_simul=100_000,
    seed=42
)

# Affichage propre
print("=== Monte Carlo (Q5) ===")
print(f"Prix estimé       = {prix_mc:.6f}")
print(f"Erreur standard   = {std_err:.6f}")
print(f"IC 95%            = [{prix_mc - ic95:.6f}, {prix_mc + ic95:.6f}]")