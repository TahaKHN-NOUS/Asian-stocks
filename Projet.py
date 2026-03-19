import numpy as np
import scipy.stats as si

def turnbull_wakeman_continu_call(S0, K, T, r, sigma):
    # 1. Calcul des moments M1 et M2
    M1 = (np.exp(r * T) - 1) / (r * T)
    
    terme1_M2 = (2 * np.exp((2 * r + sigma**2) * T)) / ((r + sigma**2) * (2 * r + sigma**2) * T**2)
    terme2_M2 = (2 / (r * T**2)) * (1 / (2 * r + sigma**2) - np.exp(r * T) / (r + sigma**2))
    M2 = terme1_M2 + terme2_M2
    
    # 2. Calcul des paramètres synthétiques r_A et sigma_A
    r_A = np.log(M1) / T
    sigma_A = np.sqrt(np.log(M2) / T - 2 * r_A)
    
    # 3. Calcul de d1 et d2
    d1 = (np.log(S0 / K) + (r_A + 0.5 * sigma_A**2) * T) / (sigma_A * np.sqrt(T))
    d2 = d1 - sigma_A * np.sqrt(T)
    
    # 4. Fonction de répartition de la loi normale (Phi)
    # L'utilisation de si.norm.cdf remplace l'approximation de l'annexe
    N_d1 = si.norm.cdf(d1, 0.0, 1.0)
    N_d2 = si.norm.cdf(d2, 0.0, 1.0)
    
    # 5. Calcul du prix final
    prix = np.exp(-r * T) * (S0 * np.exp(r_A * T) * N_d1 - K * N_d2)
    
    return prix

# Test avec les valeurs de la question Q6 (pour vérification future) :
# S0=1, K=1, r=0.01, sigma=0.3, T=0.5 (6 mois)
prix_test = turnbull_wakeman_continu_call(1.0, 1.0, 0.5, 0.01, 0.3)
print(f"Prix analytique continu du Call Asiatique : {prix_test:.5f}")