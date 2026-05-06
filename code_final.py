"""
PRB222 - Options Asiatiques
Implémentation complète : Q1 à Q16
Paramètres : sigma=0.3, S0=1, K=1, dt=1/252, r=0.01, T=6 mois
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres globaux (défauts)
# ─────────────────────────────────────────────────────────────────────────────
S0       = 1.0
K_DEF    = 1.0
r        = 0.01
SIGMA    = 0.3
T        = 0.5          # 6 mois
DT_DEF   = 1/252
EPSILON  = 1            # 1 = Call, -1 = Put

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# Approximation Abramowitz & Stegun de la CDF normale centrée réduite
# ─────────────────────────────────────────────────────────────────────────────
def norm_cdf(x: np.ndarray) -> np.ndarray:
    """CDF N(0,1) via Abramowitz & Stegun, précision < 7.5e-8."""
    b0 = 0.2316419
    b1 =  0.319381530
    b2 = -0.356563782
    b3 =  1.781477937
    b4 = -1.821255978
    b5 =  1.330274429

    x = np.asarray(x, dtype=float)
    ax = np.abs(x)
    t  = 1.0 / (1.0 + b0 * ax)
    poly = t*(b1 + t*(b2 + t*(b3 + t*(b4 + t*b5))))
    cdf_pos = 1.0 - (1.0/np.sqrt(2*np.pi)) * np.exp(-0.5*ax**2) * poly
    return np.where(x >= 0, cdf_pos, 1.0 - cdf_pos)

# ─────────────────────────────────────────────────────────────────────────────
# Q1 – Solution explicite de l'EDS de Black-Scholes
# S(t) = S0 * exp((r - σ²/2)*t + σ*W(t))
# ─────────────────────────────────────────────────────────────────────────────
def BS_path(t, W_t, S0=S0, r=r, sigma=SIGMA):
    """S(t) = S0 exp((r - σ²/2)t + σ W(t))."""
    return S0 * np.exp((r - 0.5*sigma**2)*t + sigma*W_t)

# ─────────────────────────────────────────────────────────────────────────────
# Formule de Black-Scholes standard (pour usage interne TW / variable contrôle)
# ─────────────────────────────────────────────────────────────────────────────
def bs_call_put(S, K, r_bs, sigma_bs, T_bs, eps=1):
    """Prix BS d'un call (eps=1) ou put (eps=-1)."""
    if T_bs <= 1e-12 or sigma_bs <= 1e-12 or S <= 0 or K <= 0:
        return float(max(eps*(S - K), 0.0))
    d1 = (np.log(S/K) + (r_bs + 0.5*sigma_bs**2)*T_bs) / (sigma_bs*np.sqrt(T_bs))
    d2 = d1 - sigma_bs*np.sqrt(T_bs)
    return eps * (S * norm_cdf(eps*d1) - K*np.exp(-r_bs*T_bs)*norm_cdf(eps*d2))

# ─────────────────────────────────────────────────────────────────────────────
# Q3 – Approximation Turnbull & Wakeman CONTINUE
# ─────────────────────────────────────────────────────────────────────────────
def tw_continuous(S0, K, r, sigma, T, eps=EPSILON):
    """Prix TW continu : on égalise les 2 premiers moments de la moyenne continue."""
    rT = r * T
    # M1 = E[1/T ∫₀ᵀ S(t)dt] / S0
    if abs(r) < 1e-12:
        M1 = 1.0
    else:
        M1 = (np.exp(rT) - 1.0) / rT

    # M2 = E[(1/T ∫₀ᵀ S(t)dt)²] / S0²
    r2s = 2*r + sigma**2
    M2 = (2.0*np.exp(r2s*T) / ((r + sigma**2)*r2s*T**2)
          + 2.0/(rT*T) * (1.0/r2s - np.exp(rT)/(r + sigma**2)))

    rA     = np.log(M1) / T
    sigA2  = max(np.log(M2)/T - 2.0*rA, 1e-14)
    sigA   = np.sqrt(sigA2)

    return bs_call_put(S0, K, rA, sigA, T, eps)

# ─────────────────────────────────────────────────────────────────────────────
# Q4 – Approximation Turnbull & Wakeman DISCRETISÉE
# ─────────────────────────────────────────────────────────────────────────────
def tw_discrete(S0, K, r, sigma, T, dt, eps=EPSILON):
    """Prix TW discret : on égalise les moments de 1/N Σ S(i·Δt)."""
    N   = max(1, int(round(T / dt)))
    rdt = r * dt
    s2dt = (2*r + sigma**2) * dt

    # M1
    if abs(r) < 1e-12:
        M1 = 1.0
    else:
        M1 = (1.0/N) * np.exp(rdt) * (1.0 - np.exp(rdt*N)) / (1.0 - np.exp(rdt))

    # M2 : terme diagonal + terme croisé
    denom1 = 1.0 - np.exp(s2dt)
    T1 = (1.0/N**2) * np.exp(s2dt) * (1.0 - np.exp(s2dt*N)) / denom1

    denom2 = 1.0 - np.exp(rdt)
    A = np.exp(s2dt) * (1.0 - np.exp(s2dt*(N-1))) / denom1
    B = np.exp(((N+1)*r + sigma**2)*dt) * (1.0 - np.exp((r+sigma**2)*(N-1)*dt)) / (1.0 - np.exp((r+sigma**2)*dt))
    T2 = (2.0*np.exp(rdt) / (denom2 * N**2)) * (A - B)

    M2 = T1 + T2

    rA    = np.log(M1) / T
    sigA2 = max(np.log(M2)/T - 2.0*rA, 1e-14)
    sigA  = np.sqrt(sigA2)

    return bs_call_put(S0, K, rA, sigA, T, eps)

# ─────────────────────────────────────────────────────────────────────────────
# Q5 – Simulation de trajectoires (loi uniforme uniquement → Box-Muller)
#       et estimateur Monte Carlo classique
# ─────────────────────────────────────────────────────────────────────────────
def simulate_S_paths(M_paths, N_steps, dt, r, sigma, S0):
    """
    Retourne S_paths (M, N) et W_paths (M, N).
    Utilise uniquement un générateur Uniforme (Box-Muller).
    """
    U1 = np.random.uniform(0.0, 1.0, (M_paths, N_steps))
    U2 = np.random.uniform(0.0, 1.0, (M_paths, N_steps))
    # Box-Muller : Z ~ N(0,1)
    Z  = np.sqrt(-2.0 * np.log(np.clip(U1, 1e-300, None))) * np.cos(2.0*np.pi*U2)

    dW = np.sqrt(dt) * Z
    W  = np.cumsum(dW, axis=1)               # (M, N)

    ts = np.arange(1, N_steps+1) * dt        # (N,)
    log_S = (np.log(S0)
             + (r - 0.5*sigma**2) * ts[np.newaxis, :]
             + sigma * W)
    S_paths = np.exp(log_S)
    return S_paths, W

def mc_asian(M_paths, dt, r, sigma, S0, K, T, eps=EPSILON):
    """Estimateur Monte Carlo classique de P^{Δt}."""
    N = max(1, int(round(T/dt)))
    S_paths, _ = simulate_S_paths(M_paths, N, dt, r, sigma, S0)

    arith_avg = S_paths.mean(axis=1)                       # (M,)
    payoffs   = np.maximum(eps*(arith_avg - K), 0.0)

    disc   = np.exp(-r*T)
    price  = disc * payoffs.mean()
    se     = disc * payoffs.std(ddof=1) / np.sqrt(M_paths)
    return price, se, payoffs

# ─────────────────────────────────────────────────────────────────────────────
# Q6 – Moyenne géométrique : loi et prix analytique
#
# e^{1/N Σ ln S(iΔt)}  =^{loi}  S0 · exp((r^E - (σ^E)²/2)T + σ^E W^E_T)
#
# (σ^E)² = σ² (N+1)(2N+1) / (6N²)
# r^E    = (r - σ²/2)(N+1)/(2N) + (σ^E)²/2
# ─────────────────────────────────────────────────────────────────────────────
def geom_avg_params(r, sigma, T, N):
    """Retourne (r^E, σ^E) pour la moyenne géométrique discrète."""
    sigE2 = sigma**2 * (N+1)*(2*N+1) / (6*N**2)
    sigE  = np.sqrt(sigE2)
    rE    = (r - 0.5*sigma**2)*(N+1)/(2*N) + 0.5*sigE2
    return rE, sigE

def geom_avg_price(S0, K, r, sigma, T, N, eps=EPSILON):
    """Prix analytique de l'option sur moyenne géométrique discrète."""
    rE, sigE = geom_avg_params(r, sigma, T, N)
    return bs_call_put(S0, K, rE, sigE, T, eps)

# ─────────────────────────────────────────────────────────────────────────────
# Q7 – Estimateur Monte Carlo avec variable de contrôle
#       Variable de contrôle : payoff sur moyenne géométrique
# ─────────────────────────────────────────────────────────────────────────────
def mc_asian_ctrl(M_paths, dt, r, sigma, S0, K, T, eps=EPSILON):
    """
    Estimateur MC avec variable de contrôle (moyenne géométrique).
    β optimal estimé sur l'échantillon.
    """
    N = max(1, int(round(T/dt)))
    S_paths, _ = simulate_S_paths(M_paths, N, dt, r, sigma, S0)

    arith_avg = S_paths.mean(axis=1)
    log_S     = np.log(S_paths)
    geom_avg  = np.exp(log_S.mean(axis=1))

    payoff_arith = np.maximum(eps*(arith_avg - K), 0.0)
    payoff_geom  = np.maximum(eps*(geom_avg  - K), 0.0)

    # Espérance analytique de la variable de contrôle (non actualisée)
    geom_cf = geom_avg_price(S0, K, r, sigma, T, N, eps) * np.exp(r*T)

    # Coefficient β optimal
    cov_mat = np.cov(payoff_arith, payoff_geom)
    var_g   = cov_mat[1, 1]
    beta    = cov_mat[0, 1] / var_g if var_g > 1e-14 else 0.0

    payoffs_ctrl = payoff_arith - beta*(payoff_geom - geom_cf)

    disc  = np.exp(-r*T)
    price = disc * payoffs_ctrl.mean()
    se    = disc * payoffs_ctrl.std(ddof=1) / np.sqrt(M_paths)
    return price, se, payoffs_ctrl

# ─────────────────────────────────────────────────────────────────────────────
# Q8 – Convergence en nombre de trajectoires  (CI à 90%)
# ─────────────────────────────────────────────────────────────────────────────
def plot_q8(M_max=50_000, step=500):
    print("Q8 : Convergence en nombre de trajectoires...")
    path_counts = np.arange(step, M_max+step, step)
    q90 = 1.6449                              # quantile z à 90%

    # Pre-simule tout en une fois pour cohérence
    N = max(1, int(round(T / DT_DEF)))
    S_all, _ = simulate_S_paths(M_max, N, DT_DEF, r, SIGMA, S0)
    arith     = S_all.mean(axis=1)
    log_S_all = np.log(S_all)
    geom      = np.exp(log_S_all.mean(axis=1))

    # Pré-calcul géom analytique
    geom_cf = geom_avg_price(S0, K_DEF, r, SIGMA, T, N) * np.exp(r*T)
    cov_mat = np.cov(np.maximum(arith - K_DEF, 0),
                     np.maximum(geom  - K_DEF, 0))
    beta = cov_mat[0,1] / max(cov_mat[1,1], 1e-14)

    disc    = np.exp(-r*T)
    tw_val  = tw_discrete(S0, K_DEF, r, SIGMA, T, DT_DEF)

    prices_mc, ci_mc, prices_ctrl, ci_ctrl = [], [], [], []
    for m in path_counts:
        p   = np.maximum(arith[:m] - K_DEF, 0)
        g   = np.maximum(geom[:m]  - K_DEF, 0)
        pc  = p - beta*(g - geom_cf)

        prices_mc.append(disc * p.mean())
        ci_mc.append(q90 * disc * p.std(ddof=1) / np.sqrt(m))

        prices_ctrl.append(disc * pc.mean())
        ci_ctrl.append(q90 * disc * pc.std(ddof=1) / np.sqrt(m))

    prices_mc   = np.array(prices_mc)
    ci_mc       = np.array(ci_mc)
    prices_ctrl = np.array(prices_ctrl)
    ci_ctrl     = np.array(ci_ctrl)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, prices, ci, label, col in zip(
            axes,
            [prices_mc, prices_ctrl],
            [ci_mc, ci_ctrl],
            ["MC classique $P^{\\Delta t,MC}$", "MC ctrl $P^{\\Delta t,MC,ctrl}$"],
            ["tab:blue", "tab:orange"]):
        ax.plot(path_counts, prices, color=col, label=label, lw=1.5)
        ax.fill_between(path_counts, prices-ci, prices+ci, alpha=0.25, color=col,
                        label="IC 90%")
        ax.axhline(tw_val, color="red", ls="--", lw=1.5,
                   label=f"TW discret = {tw_val:.5f}")
        ax.set_xlabel("Nombre de trajectoires")
        ax.set_ylabel("Prix estimé")
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Q8 – Convergence des estimateurs (IC 90%)", fontsize=13)
    plt.tight_layout()
    plt.savefig("q8_convergence.png", dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Q9 – Prix en fonction de K
# ─────────────────────────────────────────────────────────────────────────────
def plot_q9(M_paths=30_000, n_K=60):
    print("Q9 : Prix en fonction de K...")
    K_vals = np.linspace(0.01, 2.0, n_K)
    q90    = 1.6449
    N      = max(1, int(round(T/DT_DEF)))
    disc   = np.exp(-r*T)

    S_all, _  = simulate_S_paths(M_paths, N, DT_DEF, r, SIGMA, S0)
    arith_all = S_all.mean(axis=1)
    log_S_all = np.log(S_all)
    geom_all  = np.exp(log_S_all.mean(axis=1))

    # β global (K=1) pour stabilité
    p0 = np.maximum(arith_all - K_DEF, 0)
    g0 = np.maximum(geom_all  - K_DEF, 0)
    c0 = np.cov(p0, g0)
    beta = c0[0,1] / max(c0[1,1], 1e-14)

    prices_ctrl, ci_ctrl, prices_tw = [], [], []
    for K_ in K_vals:
        p  = np.maximum(arith_all - K_, 0)
        g  = np.maximum(geom_all  - K_, 0)
        cf = geom_avg_price(S0, K_, r, SIGMA, T, N) * np.exp(r*T)
        pc = p - beta*(g - cf)

        prices_ctrl.append(disc * pc.mean())
        ci_ctrl.append(q90 * disc * pc.std(ddof=1) / np.sqrt(M_paths))
        prices_tw.append(tw_discrete(S0, K_, r, SIGMA, T, DT_DEF))

    prices_ctrl = np.array(prices_ctrl)
    ci_ctrl     = np.array(ci_ctrl)
    prices_tw   = np.array(prices_tw)
    diff        = prices_ctrl - prices_tw

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1 : prix
    axes[0].plot(K_vals, prices_ctrl, label="MC ctrl", color="tab:blue")
    axes[0].fill_between(K_vals, prices_ctrl-ci_ctrl, prices_ctrl+ci_ctrl,
                         alpha=0.2, color="tab:blue", label="IC 90%")
    axes[0].plot(K_vals, prices_tw, "--r", label="TW discret")
    axes[0].set_xlabel("K"); axes[0].set_ylabel("Prix"); axes[0].set_title("Prix vs K")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    # Panel 2 : différence
    axes[1].plot(K_vals, diff, color="green", label="MC ctrl − TW")
    axes[1].fill_between(K_vals, diff-ci_ctrl, diff+ci_ctrl, alpha=0.2, color="green")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("K"); axes[1].set_ylabel("Différence"); axes[1].set_title("Différence MC ctrl − TW")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    # Panel 3 : zoom autour K=1
    mask = (K_vals >= 0.8) & (K_vals <= 1.2)
    axes[2].plot(K_vals[mask], diff[mask], color="green")
    axes[2].fill_between(K_vals[mask], diff[mask]-ci_ctrl[mask], diff[mask]+ci_ctrl[mask],
                         alpha=0.2, color="green")
    axes[2].axhline(0, color="k", ls="--", lw=0.8)
    axes[2].set_xlabel("K"); axes[2].set_title("Zoom K ∈ [0.8, 1.2]")
    axes[2].grid(alpha=0.3)

    plt.suptitle("Q9 – Prix et différence en fonction de K", fontsize=13)
    plt.tight_layout()
    plt.savefig("q9_vs_K.png", dpi=150)
    plt.close()
    

# ─────────────────────────────────────────────────────────────────────────────
# Q10 – Prix en fonction de σ
# ─────────────────────────────────────────────────────────────────────────────
def plot_q10(M_paths=30_000, n_sig=50):
    print("Q10 : Prix en fonction de σ...")
    sig_vals = np.linspace(0.001, 0.8, n_sig)
    q90 = 1.6449
    N   = max(1, int(round(T/DT_DEF)))
    disc = np.exp(-r*T)

    prices_ctrl, ci_ctrl, prices_tw = [], [], []
    for sig in sig_vals:
        S_all, _ = simulate_S_paths(M_paths, N, DT_DEF, r, sig, S0)
        arith = S_all.mean(axis=1)
        geom  = np.exp(np.log(S_all).mean(axis=1))

        p  = np.maximum(arith - K_DEF, 0)
        g  = np.maximum(geom  - K_DEF, 0)
        cf = geom_avg_price(S0, K_DEF, r, sig, T, N) * np.exp(r*T)
        cv = np.cov(p, g)
        beta = cv[0,1] / max(cv[1,1], 1e-14)
        pc = p - beta*(g - cf)

        prices_ctrl.append(disc * pc.mean())
        ci_ctrl.append(q90 * disc * pc.std(ddof=1) / np.sqrt(M_paths))
        prices_tw.append(tw_discrete(S0, K_DEF, r, sig, T, DT_DEF))

    prices_ctrl = np.array(prices_ctrl)
    ci_ctrl     = np.array(ci_ctrl)
    prices_tw   = np.array(prices_tw)
    diff        = prices_ctrl - prices_tw

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(sig_vals, prices_ctrl, label="MC ctrl", color="tab:blue")
    axes[0].fill_between(sig_vals, prices_ctrl-ci_ctrl, prices_ctrl+ci_ctrl,
                         alpha=0.2, color="tab:blue", label="IC 90%")
    axes[0].plot(sig_vals, prices_tw, "--r", label="TW discret")
    axes[0].set_xlabel("σ"); axes[0].set_ylabel("Prix"); axes[0].set_title("Prix vs σ")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(sig_vals, diff, color="green")
    axes[1].fill_between(sig_vals, diff-ci_ctrl, diff+ci_ctrl, alpha=0.2, color="green")
    axes[1].axhline(0, color="k", ls="--", lw=0.8)
    axes[1].set_xlabel("σ"); axes[1].set_ylabel("Différence"); axes[1].set_title("MC ctrl − TW vs σ")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Q10 – Prix et différence en fonction de σ", fontsize=13)
    plt.tight_layout()
    plt.savefig("q10_vs_sigma.png", dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Q11 – Influence de la discrétisation (TW discret vs K)
# ─────────────────────────────────────────────────────────────────────────────
def plot_q11(n_K=80):
    print("Q11 : Influence de Δt sur P^{Δt,TW} ...")
    K_vals = np.linspace(0.01, 2.0, n_K)
    dts = {"Continu (Δt→0)": 0,
           "Δt = 1/252 (jour)": 1/252,
           "Δt = 1/52 (sem.)":  1/52,
           "Δt = 1/12 (mois)":  1/12}
    styles = ["k-", "b--", "g-.", "r:"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (label, dt_v), sty in zip(dts.items(), styles):
        prices = []
        for K_ in K_vals:
            if dt_v == 0:
                prices.append(tw_continuous(S0, K_, r, SIGMA, T))
            else:
                prices.append(tw_discrete(S0, K_, r, SIGMA, T, dt_v))
        axes[0].plot(K_vals, prices, sty, label=label, lw=1.5)

        mask = (K_vals >= 0.85) & (K_vals <= 1.15)
        axes[1].plot(K_vals[mask], np.array(prices)[mask], sty, label=label, lw=1.5)

    for ax in axes:
        ax.set_xlabel("K"); ax.set_ylabel("Prix TW"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_title("P^{Δt,TW} vs K (toutes discrétisations)")
    axes[1].set_title("Zoom K ∈ [0.85, 1.15]")

    plt.suptitle("Q11 – Influence de la discrétisation sur P^{Δt,TW}", fontsize=13)
    plt.tight_layout()
    plt.savefig("q11_discretisation.png", dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Q12 – Différence MC ctrl − TW pour plusieurs Δt
# ─────────────────────────────────────────────────────────────────────────────
def plot_q12(M_paths=30_000, n_K=50):
    print("Q12 : Différence MC ctrl − TW pour différents Δt...")
    K_vals = np.linspace(0.05, 2.0, n_K)
    q90    = 1.6449
    disc   = np.exp(-r*T)
    dts    = {"1/252": 1/252, "1/52": 1/52, "1/12": 1/12}
    styles = ["b-", "g--", "r-."]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (dt_label, dt_v), sty in zip(dts.items(), styles):
        N = max(1, int(round(T/dt_v)))
        S_all, _ = simulate_S_paths(M_paths, N, dt_v, r, SIGMA, S0)
        arith = S_all.mean(axis=1)
        geom  = np.exp(np.log(S_all).mean(axis=1))

        diffs, ci_list = [], []
        for K_ in K_vals:
            p  = np.maximum(arith - K_, 0)
            g  = np.maximum(geom  - K_, 0)
            cf = geom_avg_price(S0, K_, r, SIGMA, T, N) * np.exp(r*T)
            cv = np.cov(p, g)
            beta = cv[0,1] / max(cv[1,1], 1e-14)
            pc = p - beta*(g - cf)

            mc_p = disc * pc.mean()
            ci   = q90 * disc * pc.std(ddof=1) / np.sqrt(M_paths)
            tw_p = tw_discrete(S0, K_, r, SIGMA, T, dt_v)
            diffs.append(mc_p - tw_p)
            ci_list.append(ci)

        diffs   = np.array(diffs)
        ci_list = np.array(ci_list)

        axes[0].plot(K_vals, diffs, sty, label=f"Δt={dt_label}", lw=1.5)
        axes[0].fill_between(K_vals, diffs-ci_list, diffs+ci_list, alpha=0.1)

        mask = (K_vals >= 0.85) & (K_vals <= 1.15)
        axes[1].plot(K_vals[mask], diffs[mask], sty, label=f"Δt={dt_label}", lw=1.5)
        axes[1].fill_between(K_vals[mask], (diffs-ci_list)[mask], (diffs+ci_list)[mask], alpha=0.1)

    for ax in axes:
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xlabel("K"); ax.set_ylabel("MC ctrl − TW"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_title("Différence MC ctrl − TW vs K")
    axes[1].set_title("Zoom K ∈ [0.85, 1.15]")

    plt.suptitle("Q12 – Comparaison MC ctrl vs TW (plusieurs Δt)", fontsize=13)
    plt.tight_layout()
    plt.savefig("q12_diff_discret.png", dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Q13 – Comparaison des temps de calcul
# ─────────────────────────────────────────────────────────────────────────────
def benchmark_q13(M_paths=10_000, n_rep=5):
    print("Q13 : Temps de calcul ...")
    dts = {"1/252": 1/252, "1/52": 1/52, "1/12": 1/12}
    results = {}
    for label, dt_v in dts.items():
        t_mc_list, t_tw_list = [], []
        for _ in range(n_rep):
            t0 = time.perf_counter()
            mc_asian_ctrl(M_paths, dt_v, r, SIGMA, S0, K_DEF, T)
            t_mc_list.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            tw_discrete(S0, K_DEF, r, SIGMA, T, dt_v)
            t_tw_list.append(time.perf_counter() - t0)

        results[label] = {
            "MC ctrl (s)": f"{np.mean(t_mc_list):.4f} ± {np.std(t_mc_list):.4f}",
            "TW (s)":      f"{np.mean(t_tw_list):.6f} ± {np.std(t_tw_list):.6f}",
        }
        print(f"   Δt = {label}: MC ctrl = {results[label]['MC ctrl (s)']}"
              f"  |  TW = {results[label]['TW (s)']}")

    # Graphique barres
    labels   = list(dts.keys())
    mc_means = [float(results[l]["MC ctrl (s)"].split("±")[0]) for l in labels]
    tw_means = [float(results[l]["TW (s)"].split("±")[0])      for l in labels]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - 0.2, mc_means, 0.35, label="MC ctrl", color="tab:blue")
    ax.bar(x + 0.2, tw_means, 0.35, label="TW discret", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels([f"Δt={l}" for l in labels])
    ax.set_ylabel("Temps (s)")
    ax.set_title("Q13 – Temps de calcul (10 000 trajectoires)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("q13_timing.png", dpi=150)
    plt.close()
    return results

# ─────────────────────────────────────────────────────────────────────────────
# Q15 – Parité Call/Put asiatique
#
# e^{-rT} E[(1/N Σ S(iΔt) - K)⁺] - e^{-rT} E[(K - 1/N Σ S(iΔt))⁺]
#   = e^{-rT}(S0·M1 - K)
# ─────────────────────────────────────────────────────────────────────────────
def asian_parity(S0, K, r, sigma, T, dt):
    """
    Retourne le membre droit de la parité call-put asiatique:
    C - P = e^{-rT}(S0·M1 - K)
    où M1 est le premier moment normalisé de la moyenne arithmétique.
    """
    N   = max(1, int(round(T/dt)))
    rdt = r*dt
    if abs(r) < 1e-12:
        M1 = 1.0
    else:
        M1 = (1.0/N) * np.exp(rdt) * (1.0 - np.exp(rdt*N)) / (1.0 - np.exp(rdt))
    return np.exp(-r*T) * (S0*M1 - K)

# ─────────────────────────────────────────────────────────────────────────────
# Q16 – Réduction de variance par parité + variable de contrôle géométrique
# ─────────────────────────────────────────────────────────────────────────────
def plot_q16(M_paths=20_000, n_K=60):
    """
    Pour K < S0·M1 (call dans la monnaie) → parité C-P = cst+call géom → variance ↓
    Pour K > S0·M1 (put dans la monnaie)  → parité met en jeu le put géom  → variance ↑
    Paramètre discriminant : moneyness K/S0.
    """
    print("Q16 : Réduction de variance par parité call-put...")
    K_vals = np.linspace(0.3, 1.7, n_K)
    q90    = 1.6449
    N      = max(1, int(round(T/DT_DEF)))
    disc   = np.exp(-r*T)

    # Parity constant
    M1_disc = tw_discrete(S0, 1.0, r, SIGMA, T, DT_DEF)   # proxy pour M1·S0

    S_all, _ = simulate_S_paths(M_paths, N, DT_DEF, r, SIGMA, S0)
    arith = S_all.mean(axis=1)
    geom  = np.exp(np.log(S_all).mean(axis=1))

    ci_call, ci_put, ci_parity = [], [], []
    for K_ in K_vals:
        # --- Call avec ctrl géom standard
        p_c  = np.maximum(arith - K_, 0)
        g_c  = np.maximum(geom  - K_, 0)
        cf_c = geom_avg_price(S0, K_, r, SIGMA, T, N) * np.exp(r*T)
        cv_c = np.cov(p_c, g_c)
        b_c  = cv_c[0,1] / max(cv_c[1,1], 1e-14)
        pc_c = p_c - b_c*(g_c - cf_c)
        ci_call.append(q90 * disc * pc_c.std(ddof=1) / np.sqrt(M_paths))

        # --- Put avec ctrl géom standard
        p_p  = np.maximum(K_ - arith, 0)
        g_p  = np.maximum(K_ - geom,  0)
        cf_p = geom_avg_price(S0, K_, r, SIGMA, T, N, eps=-1) * np.exp(r*T)
        cv_p = np.cov(p_p, g_p)
        b_p  = cv_p[0,1] / max(cv_p[1,1], 1e-14)
        pc_p = p_p - b_p*(g_p - cf_p)
        ci_put.append(q90 * disc * pc_p.std(ddof=1) / np.sqrt(M_paths))

        # --- Call par parité (C = P + disc*(S0*M1 - K))
        #     on estime P par MC ctrl et on ajoute la constante
        ci_parity.append(ci_put[-1])   # même variance que le put estimé par ctrl

    ci_call    = np.array(ci_call)
    ci_put     = np.array(ci_put)
    ci_parity  = np.array(ci_parity)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(K_vals, ci_call,   "b-",  label="IC call MC ctrl (direct)")
    ax.plot(K_vals, ci_put,    "r--", label="IC put  MC ctrl (direct)")
    ax.plot(K_vals, ci_parity, "g:",  label="IC call via parité (= IC put ctrl)", lw=2)
    ax.axvline(1.0, color="k", ls="--", lw=0.8, label="K = S0 = 1")
    ax.set_xlabel("K"); ax.set_ylabel("Demi-largeur IC 90%")
    ax.set_title("Q16 – Comparaison des intervalles de confiance\n"
                 "(Call direct vs Call par parité)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("q16_parity_ctrl.png", dpi=150)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN – affichage résumé numérique + génération de tous les graphiques
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  PRB222 – Options Asiatiques – Résultats numériques")
    print("=" * 60)

    # ── Q3 : TW continu
    p_tw_cont = tw_continuous(S0, K_DEF, r, SIGMA, T)
    print(f"\nQ3  – TW continu (Call)     = {p_tw_cont:.6f}")

    # ── Q4 : TW discret
    p_tw_disc = tw_discrete(S0, K_DEF, r, SIGMA, T, DT_DEF)
    print(f"Q4  – TW discret (Call)     = {p_tw_disc:.6f}")

    # ── Q5 : MC classique
    p_mc, se_mc, _ = mc_asian(20_000, DT_DEF, r, SIGMA, S0, K_DEF, T)
    print(f"Q5  – MC classique (20k)    = {p_mc:.6f}  ±  {se_mc:.6f}")

    # ── Q6 : prix géométrique analytique
    N_def = int(round(T/DT_DEF))
    p_geom = geom_avg_price(S0, K_DEF, r, SIGMA, T, N_def)
    rE, sE = geom_avg_params(r, SIGMA, T, N_def)
    print(f"Q6  – Prix géom. analytique = {p_geom:.6f}  (r^E={rE:.4f}, σ^E={sE:.4f})")

    # ── Q7 : MC ctrl
    p_ctrl, se_ctrl, _ = mc_asian_ctrl(20_000, DT_DEF, r, SIGMA, S0, K_DEF, T)
    print(f"Q7  – MC ctrl (20k)         = {p_ctrl:.6f}  ±  {se_ctrl:.6f}")
    print(f"     Réduction variance : × {(se_mc/se_ctrl):.1f}")

    # ── Q15 : parité
    par = asian_parity(S0, K_DEF, r, SIGMA, T, DT_DEF)
    p_put_ctrl, _, _ = mc_asian_ctrl(20_000, DT_DEF, r, SIGMA, S0, K_DEF, T, eps=-1)
    print(f"\nQ15 – Parité C-P théorique  = {par:.6f}")
    print(f"     Call MC ctrl - Put MC ctrl = {p_ctrl - p_put_ctrl:.6f}")

    # ── Graphiques
    print("\n--- Génération des graphiques ---")
    plot_q8()
    plot_q9()
    plot_q10()
    plot_q11()
    plot_q12()
    benchmark_q13()
    plot_q16()

