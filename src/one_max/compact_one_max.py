import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PARAMÈTRES
# =============================================================================
ONE_MAX_LENGTH = 1000
VIRTUAL_POP_SIZE = ONE_MAX_LENGTH // 10
MAX_GENERATIONS = 20000
STEP = 1.0 / VIRTUAL_POP_SIZE
NB_RUNS = 20

# =============================================================================
# MOTEUR CGA VECTORISÉ
# =============================================================================

def run_cga_numpy(run_id, n_bits, virtual_pop_n, max_iters):
    """
    Exécute un run de Compact GA.
    n_bits : Taille du génome (L)
    virtual_pop_n : Taille de la population virtuelle (n) -> Pas de mise à jour = 1/n
    """
    np.random.seed(run_id)

    # Initialisation du vecteur de probabilité à 0.5
    probability_vector = np.ones(n_bits) * 0.5

    fitness_history = np.zeros(max_iters)

    for it in range(max_iters):
        candidate_a = (np.random.rand(n_bits) < probability_vector).astype(int)
        candidate_b = (np.random.rand(n_bits) < probability_vector).astype(int)

        fit_a = np.sum(candidate_a)
        fit_b = np.sum(candidate_b)

        if fit_a >= fit_b:
            winner, loser = candidate_a, candidate_b
            current_best = fit_a
        else:
            winner, loser = candidate_b, candidate_a
            current_best = fit_b

        fitness_history[it] = current_best

        if fit_a != fit_b:

            # Masque des différences (XOR)
            diff_mask = (winner != loser)

            # Logique cGA :
            # Si Winner a 1 et Loser a 0 -> Proba augmente de 1/n
            # Si Winner a 0 et Loser a 1 -> Proba diminue de 1/n
            # Astuce math : (2*winner - 1) donne +1 si 1, et -1 si 0.

            update_direction = (2 * winner[diff_mask] - 1) * STEP
            probability_vector[diff_mask] += update_direction

            # --- CLAMPING CRUCIAL ---
            # Contrairement à l'EDA, le cGA peut converger totalement vers 0 ou 1.
            # Mais pour éviter les erreurs d'arrondi float, on clip.
            # On peut laisser converger à 0.0 et 1.0 (convergence finale)
            probability_vector = np.clip(probability_vector, 0.0, 1.0)

            # Condition d'arrêt anticipée (si le vecteur a convergé partout)
            # (Optionnel, ici on laisse tourner pour voir la courbe plate)

    return fitness_history, probability_vector


# =============================================================================
# BOUCLE PRINCIPALE (MULTI-RUNS)
# =============================================================================

def main_cga():
    print(f"--- Démarrage cGA (N={ONE_MAX_LENGTH}, Virtual Pop={VIRTUAL_POP_SIZE}) ---")
    print(f"Update Step = 1/{VIRTUAL_POP_SIZE} = {1.0 / VIRTUAL_POP_SIZE}")

    # Matrice de résultats (Runs x Iterations)
    all_runs = np.zeros((NB_RUNS, MAX_GENERATIONS))

    for r in range(NB_RUNS):
        print(f"Run {r + 1}/{NB_RUNS}...", end="\r")
        hist, final_prob = run_cga_numpy(r, ONE_MAX_LENGTH, VIRTUAL_POP_SIZE, MAX_GENERATIONS)
        all_runs[r, :] = hist

    print(f"\nTerminé.")

    # Calcul des stats
    mean_curve = np.mean(all_runs, axis=0)
    std_curve = np.std(all_runs, axis=0)

    return mean_curve, std_curve


# =============================================================================
# VISUALISATION
# =============================================================================
if __name__ == "__main__":
    mean_data, std_data = main_cga()
    x_axis = range(MAX_GENERATIONS)

    plt.figure(figsize=(10, 6))

    plt.plot(x_axis, mean_data, label="Fitness Winner (Moyenne)", color='purple', linewidth=1.5)
    plt.fill_between(x_axis,
                     mean_data - std_data,
                     mean_data + std_data,
                     color='purple', alpha=0.15, label="Ecart-type")

    plt.axhline(y=ONE_MAX_LENGTH, color='green', linestyle=':', label="Optimum")

    plt.title(f"Compact GA (cGA) : Convergence (N={ONE_MAX_LENGTH}, n={VIRTUAL_POP_SIZE})")
    plt.xlabel("Itérations (Evaluations / 2)")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
