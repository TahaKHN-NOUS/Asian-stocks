import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from scipy.stats import norm

# =============================================================================
# PARAMÈTRES GLOBAUX
# =============================================================================
S0    = 1.0          # valeur initiale de l'actif
K     = 1.0          # strike
r     = 0.01         # taux sans risque annuel
sigma = 0.3          # volatilité annuelle
T     = 0.5          # maturité (6 mois)
dt    = 1/252        # pas de temps (1 jour ouvré)
N     = int(T/dt)    # nombre d'observations : ~126
eps   = 1            # ε = 1 pour un Call

np.random.seed(42)

# =============================================================================
# UTILITAIRES
# =============================================================================

def normal_cdf(x):
    """
    CDF de la loi normale standard via l'approximation d'Abramowitz & Stegun.
    Uniquement nécessaire si on interdit scipy ; ici on garde scipy pour
    la clarté mais on peut la remplacer par cette fonction.
    """
    b0 = 0.2316419
    b1, b2, b3, b4, b5 = (0.319381530, -0.356563782,
                           1.781477937, -1.821255978, 1.330274429)
    if x >= 0:
        t = 1 / (1 + b0 * x)
        poly = t*(b1 + t*(b2 + t*(b3 + t*(b4 + t*b5))))
        return 1 - (1/np.sqrt(2*np.pi)) * np.exp(-0.5*x**2) * poly
    else:
        return 1 - normal_cdf(-x)

def N_cdf(x):
    """Wrapper vectorisé pour la CDF normale."""
    return np.vectorize(normal_cdf)(x)

def bs_call(S, K, r_bs, sigma_bs, T_bs):
    """
    Prix d'un Call Black-Scholes classique.
    Utilisé pour l'approximation TW et le prix de contrôle (Q6).
    """
    if sigma_bs <= 0 or T_bs <= 0:
        return max(S - K * np.exp(-r_bs * T_bs), 0)
    d1 = (np.log(S/K) + (r_bs + 0.5*sigma_bs**2)*T_bs) / (sigma_bs*np.sqrt(T_bs))
    d2 = d1 - sigma_bs*np.sqrt(T_bs)
    return S * np.exp((r_bs - r)*T_bs) * N_cdf(d1) - K * np.exp(-r*T_bs) * N_cdf(d2)

def uniform_to_normal(u1, u2):
    """
    Transformation de Box-Muller : deux U(0,1) → deux N(0,1).
    On utilise UNIQUEMENT un générateur uniforme comme demandé.
    """
    z1 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    z2 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
    return z1, z2

# =============================================================================
# Q1 : SOLUTION DE L'EDS DE BLACK-SCHOLES
# =============================================================================
def S_exact(t, W_t):
    """S(t) = S0 * exp((r - σ²/2)*t + σ*W(t))"""
    return S0 * np.exp((r - 0.5*sigma**2)*t + sigma*W_t)

# =============================================================================
# Q2 : OPTION ASIATIQUE DÉJÀ COMMENCÉE (t0 < 0)
# =============================================================================
"""
    K' = [K(T-t0) + t0 * A_past] / T   (strike ajusté, t0 < 0)
    P_0(t0<0) = T/(T-t0) * P_0^{K'}(t0=0)

On se ramène ainsi à une option asiatique standard (t0=0) avec strike K'.
"""

# =============================================================================
# Q3 : APPROXIMATION TURNBULL-WAKEMAN CONTINUE
# =============================================================================
"""
Q3. Approximation TW pour le cas continu.
Le prix TW est alors le prix Black-Scholes avec (r_A, σ_A) :
    P_0,TW = S0 * e^{(r_A - r)T} N(d1) - K * e^{-rT} N(d2)
    d1 = [ln(S0/K) + (r_A + σ_A²/2)T] / (σ_A√T)
    d2 = d1 - σ_A√T
"""

def TW_continuous(S0, K, r, sigma, T):
    """Prix TW pour le cas continu (moyenne continue)."""
    M1 = (np.exp(r*T) - 1) / (r*T)
    M2 = (2*np.exp((2*r + sigma**2)*T) /
          ((r + sigma**2)*(2*r + sigma**2)*T**2)
          + 2/(r*T**2) * (1/(2*r+sigma**2) - np.exp(r*T)/(r+sigma**2)))
    r_A = np.log(M1) / T
    sigma_A2 = np.log(M2) / T - 2*r_A
    if sigma_A2 <= 0:
        sigma_A = 0.0
    else:
        sigma_A = np.sqrt(sigma_A2)
    return bs_call(S0, K, r_A, sigma_A, T)

print(f"[Q3] P_0,TW (continu) = {TW_continuous(S0, K, r, sigma, T):.6f}")

# =============================================================================
# Q4 : APPROXIMATION TURNBULL-WAKEMAN DISCRÈTE
# =============================================================================
"""
Q4. Approximation TW pour le cas discret.
Mêmes formules pour r_A et σ_A qu'en continu.
"""

def TW_discrete(S0, K, r, sigma, T, dt):
    """Prix TW pour le cas discret avec pas dt."""
    N = int(round(T / dt))
    if N == 0:
        return TW_continuous(S0, K, r, sigma, T)

    # Terme M1
    if abs(r) < 1e-10:
        M1 = 1.0
    else:
        M1 = (1/N) * np.exp(r*dt) * (1 - np.exp(r*N*dt)) / (1 - np.exp(r*dt))

    # Terme M2 — partie 1
    alpha1 = (2*r + sigma**2)*dt
    if abs(np.expm1(alpha1)) < 1e-10:
        part1 = (1/N**2) * N
    else:
        part1 = (1/N**2) * np.exp(alpha1) * (1 - np.exp(alpha1*N)) / (1 - np.exp(alpha1))

    # Terme M2 — partie 2
    alpha2 = (2*r + sigma**2)*dt
    alpha3 = (r + sigma**2)*dt
    if abs(np.expm1(r*dt)) < 1e-10 or abs(np.expm1(alpha2)) < 1e-10 or abs(np.expm1(alpha3)) < 1e-10:
        part2 = 0.0
    else:
        A = np.exp(alpha2) * (1 - np.exp(alpha2*(N-1))) / (1 - np.exp(alpha2))
        B = np.exp(((N+1)*r + sigma**2)*dt) * (1 - np.exp((r+sigma**2)*(N-1)*dt)) / (1 - np.exp(alpha3))
        part2 = (1/N**2) * 2*np.exp(r*dt)/(1 - np.exp(r*dt)) * (A - B)

    M2 = part1 + part2

    r_A = np.log(M1) / T
    sigma_A2 = np.log(M2) / T - 2*r_A
    sigma_A = np.sqrt(max(sigma_A2, 0))
    return bs_call(S0, K, r_A, sigma_A, T)

P_TW_disc = TW_discrete(S0, K, r, sigma, T, dt)
print(f"[Q4] P_∆t,TW (discret, ∆t=1/252) = {P_TW_disc:.6f}")

# =============================================================================
# Q5 : SIMULATION MONTE CARLO CLASSIQUE
# =============================================================================
"""
Q5. Simulation d'une trajectoire (W(i∆t))_{i=1..N}.

Idée : les accroissements W(i∆t) - W((i-1)∆t) sont i.i.d. N(0, ∆t).
On génère des N(0,1) par la méthode de Box-Muller à partir de U(0,1) :
    Z = √(-2 ln U1) cos(2π U2)
Puis W(i∆t) = Σ_{k=1}^i √∆t * Z_k  (marche aléatoire brownienne)

L'estimateur MC du prix est :
    P̂_∆t,MC = e^{-rT} * 1/n * Σ_{j=1}^n ε(Ā_j - K)+

où Ā_j = 1/N Σ_{i=1}^N S_j(i∆t) est la moyenne arithmétique de la trajectoire j.
"""

def simulate_paths(n_paths, N=N, dt=dt):
    """
    Génère n_paths trajectoires (S(i∆t))_{i=1..N}.
    Utilise uniquement un générateur uniforme (Box-Muller).
    Retourne : S_paths shape (n_paths, N), W_paths shape (n_paths, N)
    """
    # Génération des uniformes (N_paths * N points)
    # On a besoin de 2 uniformes par incrément pour Box-Muller
    # Si N*n_paths est impair, on génère un de plus
    total = n_paths * N
    u1 = np.random.uniform(0, 1, size=total)
    u2 = np.random.uniform(0, 1, size=total)
    z, _ = uniform_to_normal(np.clip(u1, 1e-10, 1-1e-10), u2)
    increments = z.reshape(n_paths, N) * np.sqrt(dt)   # shape (n_paths, N)

    # Trajectoires Browniennes cumulées
    W = np.cumsum(increments, axis=1)    # W[j, i] = W(i∆t) pour trajectoire j

    # Temps
    t_grid = np.arange(1, N+1) * dt     # (i∆t)_{i=1..N}

    # S(i∆t) = S0 * exp((r - σ²/2)*i∆t + σ*W(i∆t))
    S = S0 * np.exp((r - 0.5*sigma**2)*t_grid + sigma*W)
    return S, W

def MC_price_classic(n_paths, N=N, dt=dt):
    """Estimateur MC classique et son écart-type."""
    S, _ = simulate_paths(n_paths, N, dt)
    A = S.mean(axis=1)                          # moyenne arithmétique
    payoffs = np.exp(-r*T) * np.maximum(eps*(A - K), 0)
    price = payoffs.mean()
    std_err = payoffs.std() / np.sqrt(n_paths)
    return price, std_err

P_MC, se_MC = MC_price_classic(10_000)
print(f"[Q5] P_∆t,MC (n=10000) = {P_MC:.6f} ± {1.645*se_MC:.6f} (IC 90%)")

# =============================================================================
# Q6 : DISTRIBUTION DE LA MOYENNE GÉOMÉTRIQUE
# =============================================================================
"""
Q6. Montrer que exp(1/N * Σ ln S(i∆t)) suit une loi log-normale.
    (σ^E)² = σ²∆t(N+1)(2N+1) / (6N*T) = σ²(N+1)(2N+1) / (6N²)
    r^E    = (r - σ²/2)*(N+1)/(2N) + (σ^E)²/2

Le prix de la moyenne géométrique est un prix Black-Scholes :
    e^{-rT} E[ε(G_T - K)+] = BS_Call(S0, K, r^E, σ^E, T)
"""

def geometric_asian_analytic(S0, K, r, sigma, T, N):
    """Prix analytique de l'option sur moyenne géométrique (Q6)."""
    sigma_E2 = sigma**2 * (N+1)*(2*N+1) / (6*N**2)
    sigma_E  = np.sqrt(sigma_E2)
    r_E = (r - 0.5*sigma**2)*(N+1)/(2*N) + 0.5*sigma_E2
    return bs_call(S0, K, r_E, sigma_E, T)

P_geom = geometric_asian_analytic(S0, K, r, sigma, T, N)
print(f"[Q6] Prix géométrique analytique = {P_geom:.6f}")

# =============================================================================
# Q7 : ESTIMATEUR PAR VARIABLE DE CONTRÔLE
# =============================================================================
"""
Q7. Variable de contrôle.

Idée : la moyenne géométrique G_T = exp(1/N Σ ln S(i∆t)) est corrélée
avec la moyenne arithmétique A_T = 1/N Σ S(i∆t), et son prix est connu
analytiquement (Q6).

On définit l'estimateur à variable de contrôle :
    P̂_ctrl = (1/n) Σ [h_i - c*(g_i - E[g])]

avec :
    h_i = e^{-rT} * (A_i - K)+       (payoff arithmétique)
    g_i = e^{-rT} * (G_i - K)+       (payoff géométrique)
    E[g] = P_geom                     (valeur analytique Q6)
    c_opt = Cov(h, g) / Var(g)        (coefficient optimal)

c_opt réduit la variance en exploitant la corrélation entre A et G.
En pratique, on estime c sur le même batch de simulations.
"""

def MC_price_control_variate(n_paths, N=N, dt=dt, S0=S0, K=K, r=r, sigma=sigma, T=T):
    """
    Estimateur MC avec variable de contrôle (moyenne géométrique).
    Retourne : prix, écart-type de Monte Carlo.
    """
    S, _ = simulate_paths(n_paths, N, dt)

    # Moyenne arithmétique et géométrique par trajectoire
    A = S.mean(axis=1)
    G = np.exp(np.log(S).mean(axis=1))   # moyenne géométrique

    # Payoffs actualisés
    h = np.exp(-r*T) * np.maximum(eps*(A - K), 0)
    g = np.exp(-r*T) * np.maximum(eps*(G - K), 0)

    # Valeur analytique du contrôle
    E_g = geometric_asian_analytic(S0, K, r, sigma, T, N)

    # Coefficient optimal c = Cov(h,g) / Var(g)
    cov_hg = np.cov(h, g)[0, 1]
    var_g  = np.var(g, ddof=1)
    c_opt  = cov_hg / var_g if var_g > 1e-14 else 1.0

    # Estimateur corrigé
    h_ctrl = h - c_opt * (g - E_g)
    price  = h_ctrl.mean()
    std_err = h_ctrl.std(ddof=1) / np.sqrt(n_paths)
    return price, std_err

P_ctrl, se_ctrl = MC_price_control_variate(10_000)
print(f"[Q7] P_∆t,MC,ctrl (n=10000) = {P_ctrl:.6f} ± {1.645*se_ctrl:.6f} (IC 90%)")
print(f"     Réduction de variance : {(se_MC/se_ctrl)**2:.1f}x")

# =============================================================================
# Q8 : CONVERGENCE EN FONCTION DU NOMBRE DE TRAJECTOIRES
# =============================================================================
"""
Q8. Tracer P_∆t,MC et P_∆t,MC,ctrl en fonction de n, avec IC 90%.

On trace les deux estimateurs et leurs IC asymptotiques à 90% (z_{0.05}=1.645)
en fonction du nombre de trajectoires, ainsi que la valeur TW discrète.
"""

def plot_Q8():
    n_values = np.logspace(1, 4, 30).astype(int)
    prices_mc, se_mc = [], []
    prices_ctrl, se_ctrl_arr = [], []

    for n in n_values:
        p, s = MC_price_classic(n)
        prices_mc.append(p); se_mc.append(s)
        p2, s2 = MC_price_control_variate(n)
        prices_ctrl.append(p2); se_ctrl_arr.append(s2)

    prices_mc = np.array(prices_mc)
    prices_ctrl = np.array(prices_ctrl)
    se_mc = np.array(se_mc)
    se_ctrl_arr = np.array(se_ctrl_arr)
    z = 1.645

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Q8 — Convergence des estimateurs Monte Carlo", fontsize=13, fontweight='bold')

    for ax, prices, ses, label, color in zip(
        axes,
        [prices_mc, prices_ctrl],
        [se_mc, se_ctrl_arr],
        ["MC classique", "MC variable de contrôle"],
        ["steelblue", "darkorange"]
    ):
        ax.fill_between(n_values, prices - z*ses, prices + z*ses,
                        alpha=0.25, color=color, label="IC 90%")
        ax.plot(n_values, prices, color=color, lw=1.8, label=label)
        ax.axhline(P_TW_disc, color='crimson', lw=1.5, ls='--', label=r"$P_{\Delta t,TW}$")
        ax.set_xscale('log')
        ax.set_xlabel("Nombre de trajectoires n", fontsize=11)
        ax.set_ylabel("Prix estimé", fontsize=11)
        ax.set_title(label)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/Q8_convergence.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[Q8] Figure sauvegardée : Q8_convergence.png")

plot_Q8()

# =============================================================================
# Q9 : EN FONCTION DE K
# =============================================================================
"""
Q9. Tracer P_∆t,MC,ctrl et P_∆t,TW en fonction de K ∈ [0,2].

On choisit n = 50 000 trajectoires pour un IC 90% de demi-largeur < 0.001
(critère de précision : IC < 0.2% du sous-jacent).
On trace aussi la différence entre les deux estimateurs et on zoome autour de K=1.
"""

def plot_Q9(n_traj=50_000):
    K_values = np.linspace(0, 2, 60)
    prices_ctrl, se_ctrl_arr, prices_TW = [], [], []

    # Simulations une seule fois pour toutes les valeurs de K
    S, _ = simulate_paths(n_traj, N, dt)
    A = S.mean(axis=1)
    G = np.exp(np.log(S).mean(axis=1))

    for Kv in K_values:
        h = np.exp(-r*T) * np.maximum(eps*(A - Kv), 0)
        g = np.exp(-r*T) * np.maximum(eps*(G - Kv), 0)
        E_g = geometric_asian_analytic(S0, Kv, r, sigma, T, N)
        cov_hg = np.cov(h, g)[0, 1]
        var_g  = np.var(g, ddof=1)
        c_opt  = cov_hg / var_g if var_g > 1e-14 else 1.0
        h_ctrl = h - c_opt*(g - E_g)
        prices_ctrl.append(h_ctrl.mean())
        se_ctrl_arr.append(h_ctrl.std(ddof=1)/np.sqrt(n_traj))
        prices_TW.append(TW_discrete(S0, Kv, r, sigma, T, dt))

    prices_ctrl = np.array(prices_ctrl)
    prices_TW   = np.array(prices_TW)
    se_ctrl_arr = np.array(se_ctrl_arr)
    diff = prices_ctrl - prices_TW
    z = 1.645

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Q9 — Prix en fonction de K (n={n_traj})", fontsize=13, fontweight='bold')

    for idx, (ax, K_lim, title) in enumerate(zip(
        [axes[0,0], axes[0,1], axes[1,0], axes[1,1]],
        [(0, 2), (0.8, 1.2), (0, 2), (0.8, 1.2)],
        ["Prix vs K", "Zoom K∈[0.8,1.2]", "Différence TW - MC,ctrl", "Zoom différence K∈[0.8,1.2]"]
    )):
        mask = (K_values >= K_lim[0]) & (K_values <= K_lim[1])
        Kv = K_values[mask]
        if idx < 2:
            ax.fill_between(Kv, prices_ctrl[mask]-z*se_ctrl_arr[mask],
                            prices_ctrl[mask]+z*se_ctrl_arr[mask],
                            alpha=0.25, color='darkorange', label="IC 90%")
            ax.plot(Kv, prices_ctrl[mask], color='darkorange', lw=2, label=r"$P_{\Delta t,MC,ctrl}$")
            ax.plot(Kv, prices_TW[mask], 'r--', lw=2, label=r"$P_{\Delta t,TW}$")
            ax.set_ylabel("Prix"); ax.legend(fontsize=9)
        else:
            ax.fill_between(Kv, diff[mask]-z*se_ctrl_arr[mask],
                            diff[mask]+z*se_ctrl_arr[mask],
                            alpha=0.3, color='gray', label="IC MC 90%")
            ax.plot(Kv, diff[mask], color='purple', lw=2, label="MC,ctrl - TW")
            ax.axhline(0, color='black', lw=0.8, ls=':')
            ax.set_ylabel("Différence"); ax.legend(fontsize=9)
        ax.set_title(title); ax.set_xlabel("K"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/Q9_vs_K.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[Q9] Figure sauvegardée : Q9_vs_K.png")

plot_Q9()

# =============================================================================
# Q10 : EN FONCTION DE σ
# =============================================================================
"""
Q10. Tracer P_∆t,MC,ctrl et P_∆t,TW en fonction de σ ∈ [0, 0.8].

On simule pour chaque valeur de σ (les trajectoires changent avec σ).
On garde les autres paramètres fixes.
"""

def plot_Q10(n_traj=30_000):
    sigma_values = np.linspace(0, 0.8, 40)
    prices_ctrl, se_ctrl_arr, prices_TW = [], [], []

    for sv in sigma_values:
        # Resimulation avec sigma=sv
        total = n_traj * N
        u1 = np.random.uniform(0, 1, size=total)
        u2 = np.random.uniform(0, 1, size=total)
        z_norm, _ = uniform_to_normal(np.clip(u1, 1e-10, 1-1e-10), u2)
        increments = z_norm.reshape(n_traj, N) * np.sqrt(dt)
        W = np.cumsum(increments, axis=1)
        t_grid = np.arange(1, N+1) * dt
        S = S0 * np.exp((r - 0.5*sv**2)*t_grid + sv*W)

        A = S.mean(axis=1)
        G = np.exp(np.log(S + 1e-300).mean(axis=1))

        h = np.exp(-r*T) * np.maximum(eps*(A - K), 0)
        g = np.exp(-r*T) * np.maximum(eps*(G - K), 0)
        E_g = geometric_asian_analytic(S0, K, r, sv, T, N)
        cov_hg = np.cov(h, g)[0, 1]
        var_g  = np.var(g, ddof=1)
        c_opt  = cov_hg / var_g if var_g > 1e-14 else 1.0
        h_ctrl = h - c_opt*(g - E_g)
        prices_ctrl.append(h_ctrl.mean())
        se_ctrl_arr.append(h_ctrl.std(ddof=1)/np.sqrt(n_traj))
        prices_TW.append(TW_discrete(S0, K, r, sv, T, dt))

    prices_ctrl = np.array(prices_ctrl)
    prices_TW   = np.array(prices_TW)
    se_ctrl_arr = np.array(se_ctrl_arr)
    diff = prices_ctrl - prices_TW
    z = 1.645

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Q10 — Prix en fonction de σ (n={n_traj})", fontsize=13, fontweight='bold')

    ax1.fill_between(sigma_values, prices_ctrl-z*se_ctrl_arr,
                     prices_ctrl+z*se_ctrl_arr, alpha=0.25, color='darkorange', label="IC 90%")
    ax1.plot(sigma_values, prices_ctrl, color='darkorange', lw=2, label=r"$P_{\Delta t,MC,ctrl}$")
    ax1.plot(sigma_values, prices_TW, 'r--', lw=2, label=r"$P_{\Delta t,TW}$")
    ax1.set_xlabel("σ"); ax1.set_ylabel("Prix"); ax1.set_title("Prix en fonction de σ")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.fill_between(sigma_values, diff-z*se_ctrl_arr,
                     diff+z*se_ctrl_arr, alpha=0.3, color='gray', label="IC MC 90%")
    ax2.plot(sigma_values, diff, color='purple', lw=2, label="MC,ctrl - TW")
    ax2.axhline(0, color='black', lw=0.8, ls=':')
    ax2.set_xlabel("σ"); ax2.set_ylabel("Différence"); ax2.set_title("Différence MC,ctrl − TW")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/Q10_vs_sigma.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[Q10] Figure sauvegardée : Q10_vs_sigma.png")

plot_Q10()

# =============================================================================
# Q11 : INFLUENCE DE LA DISCRÉTISATION (TW UNIQUEMENT)
# =============================================================================
"""
Q11. Tracer P_∆t,TW en fonction de K pour plusieurs valeurs de ∆t.

Les valeurs de ∆t testées :
- ∆t = 0      : cas continu (formule TW_continuous)
- ∆t = 1/252  : observations journalières
- ∆t = 1/52   : observations hebdomadaires
- ∆t = 1/12   : observations mensuelles

On zoome autour de K=1 pour voir les différences.
"""

def plot_Q11():
    K_values = np.linspace(0, 2, 200)
    dt_cases = {
        r"$\Delta t=0$ (continu)": None,
        r"$\Delta t=1/252$ (jour)": 1/252,
        r"$\Delta t=1/52$ (sem.)": 1/52,
        r"$\Delta t=1/12$ (mois)": 1/12,
    }
    colors = ['black','steelblue','darkorange','crimson']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Q11 — Approximation TW pour différentes discrétisations", fontsize=13, fontweight='bold')

    for (label, dti), color in zip(dt_cases.items(), colors):
        prices = [TW_continuous(S0, Kv, r, sigma, T) if dti is None
                  else TW_discrete(S0, Kv, r, sigma, T, dti)
                  for Kv in K_values]
        for ax, Klim in zip([ax1, ax2], [(0,2),(0.8,1.2)]):
            mask = (K_values >= Klim[0]) & (K_values <= Klim[1])
            ax.plot(K_values[mask], np.array(prices)[mask],
                    color=color, lw=2, label=label)

    for ax, title in zip([ax1, ax2], ["Prix TW vs K", "Zoom K∈[0.8,1.2]"]):
        ax.set_xlabel("K"); ax.set_ylabel("Prix")
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/Q11_TW_discretisation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[Q11] Figure sauvegardée : Q11_TW_discretisation.png")

plot_Q11()

# =============================================================================
# Q12 : DIFFÉRENCE MC,ctrl - TW POUR PLUSIEURS ∆t
# =============================================================================
"""
Q12. Tracer la différence P_∆t,MC,ctrl - P_∆t,TW pour ∆t ∈ {1/252, 1/52, 1/12}.

On compare la différence à la largeur de l'IC de l'estimateur MC,ctrl.
Si la différence est dans l'IC → TW est précis à ce niveau.
Si elle dépasse → biais non négligeable de l'approximation TW.
"""

def plot_Q12(n_traj=50_000):
    K_values = np.linspace(0, 2, 60)
    dt_cases = {
        r"$\Delta t=1/252$": 1/252,
        r"$\Delta t=1/52$":  1/52,
        r"$\Delta t=1/12$":  1/12,
    }
    colors = ['steelblue', 'darkorange', 'crimson']
    z = 1.645

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Q12 — Différence MC,ctrl − TW (n={n_traj})", fontsize=13, fontweight='bold')

    for (label, dti), color in zip(dt_cases.items(), colors):
        Ni = int(round(T / dti))
        # Simulation unique pour ce ∆t
        total = n_traj * Ni
        u1 = np.random.uniform(0, 1, size=total)
        u2 = np.random.uniform(0, 1, size=total)
        z_norm, _ = uniform_to_normal(np.clip(u1, 1e-10, 1-1e-10), u2)
        increments = z_norm.reshape(n_traj, Ni) * np.sqrt(dti)
        W = np.cumsum(increments, axis=1)
        t_grid = np.arange(1, Ni+1) * dti
        S = S0 * np.exp((r - 0.5*sigma**2)*t_grid + sigma*W)
        A = S.mean(axis=1)
        G = np.exp(np.log(S + 1e-300).mean(axis=1))

        diffs, widths_IC = [], []
        for Kv in K_values:
            h = np.exp(-r*T) * np.maximum(eps*(A - Kv), 0)
            g = np.exp(-r*T) * np.maximum(eps*(G - Kv), 0)
            E_g = geometric_asian_analytic(S0, Kv, r, sigma, T, Ni)
            cov_hg = np.cov(h, g)[0, 1]
            var_g  = np.var(g, ddof=1)
            c_opt  = cov_hg / var_g if var_g > 1e-14 else 1.0
            h_ctrl = h - c_opt*(g - E_g)
            p_mc   = h_ctrl.mean()
            se_mc  = h_ctrl.std(ddof=1)/np.sqrt(n_traj)
            p_tw   = TW_discrete(S0, Kv, r, sigma, T, dti)
            diffs.append(p_mc - p_tw)
            widths_IC.append(z * se_mc)

        diffs = np.array(diffs)
        widths_IC = np.array(widths_IC)

        for ax, Klim in zip(axes, [(0,2),(0.8,1.2)]):
            mask = (K_values >= Klim[0]) & (K_values <= Klim[1])
            ax.plot(K_values[mask], diffs[mask], color=color, lw=2, label=label)
            ax.fill_between(K_values[mask],
                            diffs[mask]-widths_IC[mask],
                            diffs[mask]+widths_IC[mask],
                            alpha=0.12, color=color)

    for ax, title in zip(axes, ["Différence vs K", "Zoom K∈[0.8,1.2]"]):
        ax.axhline(0, color='black', lw=0.8, ls=':')
        ax.set_xlabel("K"); ax.set_ylabel("MC,ctrl − TW")
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/mnt/user-data/outputs/Q12_diff_discretisation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("[Q12] Figure sauvegardée : Q12_diff_discretisation.png")

plot_Q12()

print("\n✓ Toutes les figures ont été générées dans /mnt/user-data/outputs/")