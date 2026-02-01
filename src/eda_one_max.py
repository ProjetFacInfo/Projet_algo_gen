import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
import random

# =============================================================================
# 1. PARAMÈTRES
# =============================================================================
# ONE_MAX_LENGTH = 200                  # Taille du problème (N)
ONE_MAX_LENGTH = 1000                   # Taille du problème (N)
POPULATION_SIZE = ONE_MAX_LENGTH // 10   # Taille de la population (M)
MAX_GENERATIONS = 500                   # Durée du run
NB_SEL = POPULATION_SIZE // 2            # Nombre de meilleurs pour l'apprentissage (50% ici)
NB_RUNS = 20                            # Nombre de répétitions pour la moyenne

# Clamping (Sécurité)
# MIN_PROBA = 0.02
# MAX_PROBA = 0.98
MIN_PROBA = 1.0 / ONE_MAX_LENGTH
MAX_PROBA = 1.0 - MIN_PROBA

# =============================================================================
# 2. SETUP DEAP
# =============================================================================
if "FitnessMax" in dir(creator): del creator.FitnessMax
if "Individual" in dir(creator): del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def eval_numpy(ind):
    return float(np.sum(ind)),

toolbox.register("evaluate", eval_numpy)

# =============================================================================
# 3. FONCTIONS CŒUR (EDA)
# =============================================================================
def genere_population_vectorisee(distribution, pop_size):
    random_matrix = np.random.rand(pop_size, len(distribution))
    pop_matrix = (random_matrix < distribution).astype(int)
    return [creator.Individual(row) for row in pop_matrix]

def maj_distribution_vectorisee(population, k):
    best_inds = tools.selBest(population, k)
    best_matrix = np.array(best_inds)
    new_distrib = np.mean(best_matrix, axis=0)
    new_distrib = np.clip(new_distrib, MIN_PROBA, MAX_PROBA)
    return new_distrib

# =============================================================================
# 4. GESTION DES RUNS (MODIFIÉE POUR CAPTER MAX ET AVG)
# =============================================================================

def run_single_eda(run_id):
    """ Exécute 1 run et renvoie DEUX listes : historique Max et historique Avg. """
    np.random.seed(run_id)
    random.seed(run_id)

    distribution = np.ones(ONE_MAX_LENGTH) * 0.5

    hist_max = []
    hist_avg = []

    for gen in range(MAX_GENERATIONS):
        # 1. Sampling & Eval
        population = genere_population_vectorisee(distribution, POPULATION_SIZE)
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # 2. Update
        distribution = maj_distribution_vectorisee(population, NB_SEL)

        # 3. Stats (On capture les deux métriques)
        fits = [ind.fitness.values[0] for ind in population]
        hist_max.append(max(fits))
        hist_avg.append(np.mean(fits))

    return hist_max, hist_avg

def main_multi_runs():
    print(f"--- Démarrage EDA (Runs={NB_RUNS}, N={ONE_MAX_LENGTH}) ---")

    # Matrices pour stocker TOUS les runs
    mat_max = np.zeros((NB_RUNS, MAX_GENERATIONS))
    mat_avg = np.zeros((NB_RUNS, MAX_GENERATIONS))

    for r in range(NB_RUNS):
        print(f"Run {r+1}/{NB_RUNS}...", end="\r")
        # On récupère les deux courbes du run r
        r_max, r_avg = run_single_eda(r)
        mat_max[r, :] = r_max
        mat_avg[r, :] = r_avg

    print(f"\nCalcul des statistiques...")

    # Moyennes et Ecart-types sur l'ensemble des runs
    stats = {
        "mean_max": np.mean(mat_max, axis=0),
        "std_max": np.std(mat_max, axis=0),
        "mean_avg": np.mean(mat_avg, axis=0),
        "std_avg": np.std(mat_avg, axis=0)
    }

    return stats

# =============================================================================
# 5. AFFICHAGE (Double courbe avec ombres)
# =============================================================================
if __name__ == "__main__":
    s = main_multi_runs()
    gens = range(MAX_GENERATIONS)

    plt.figure(figsize=(10, 6))

    # --- COURBE 1 : La Performance (Le meilleur individu) ---
    plt.plot(gens, s["mean_max"], label="Meilleur Individu (Moyenne)", color='blue', linewidth=2)
    plt.fill_between(gens, s["mean_max"] - s["std_max"], s["mean_max"] + s["std_max"],
                     color='blue', alpha=0.15)

    # --- COURBE 2 : La Convergence (La moyenne de la population) ---
    plt.plot(gens, s["mean_avg"], label="Moyenne Population (Moyenne)", color='red', linestyle='--', linewidth=2)
    plt.fill_between(gens, s["mean_avg"] - s["std_avg"], s["mean_avg"] + s["std_avg"],
                     color='red', alpha=0.15)

    # Optimum
    plt.axhline(y=ONE_MAX_LENGTH, color='green', linestyle=':', label="Optimum")

    plt.title(f"EDA : Performance vs Convergence (Moyenne sur {NB_RUNS} runs)")
    plt.xlabel("Générations")
    plt.ylabel("Fitness")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()
