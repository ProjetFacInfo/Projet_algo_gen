import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
from functools import partial

# ONE_MAX_LENGTH = 200
ONE_MAX_LENGTH = 1000
POPULATION_SIZE = 20
MAX_GENERATIONS = 10000
N_RUNS = 20


# Création des types (FitnessMax et Individual)
if "FitnessMax" in dir(creator): del creator.FitnessMax
if "Individual" in dir(creator): del creator.Individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)


# Fontions
# Selection
def selRandom(individuals, k):
    """
    Sélection aléatoire pure (optimisée).
    Utilise np.random.choice qui est plus rapide que random.sample pour les grands nombres.
    """
    # replace=False pour simuler le comportement par défaut (pas de doublons si k <= len)
    # Note: np.random.choice attend un array 1D, on lui donne les indices
    chosen_indices = np.random.choice(len(individuals), k, replace=False)
    return [individuals[i] for i in chosen_indices]


def selBest(individuals, k):
    """
    Sélection élitiste (optimisée).
    Trie les indices basé sur les fitnesses au lieu de trier les objets.
    """
    # Extraction vectorisée des fitnesses (suppose 1 seul objectif)
    # On met un signe - pour utiliser argsort (qui trie croissant) si on veut maximiser
    fits = np.array([ind.fitness.values[0] for ind in individuals])

    # ArgSort est beaucoup plus rapide qu'un sort() python sur des objets complexes
    # On prend les k derniers indices (les plus grands) car argsort trie croissant
    # Ou alors on trie -fits pour avoir les meilleurs en premier
    top_indices = np.argsort(fits)[::-1][:k]

    return [individuals[i] for i in top_indices]


def selTournament(individuals, k, tournsize):
    """
    Tournoi vectorisé (Ultra rapide).
    Au lieu de faire k boucles for, on tire une matrice d'indices d'un coup.
    """
    n_individuals = len(individuals)

    # 1. Extraction des fitnesses
    fits = np.array([ind.fitness.values[0] for ind in individuals])

    # 2. Création de la matrice de tournoi (k combats de 'tournsize' participants)
    # On génère k * tournsize indices aléatoires
    competitors_indices = np.random.randint(0, n_individuals, (k, tournsize))

    # 3. Récupération des fitnesses correspondantes
    competitors_fitnesses = fits[competitors_indices]

    # 4. Trouver le gagnant de chaque ligne (argmax sur l'axe 1)
    # winners_local_indices[i] donne l'index (0..tournsize-1) du gagnant du i-ème tournoi
    winners_local_indices = np.argmax(competitors_fitnesses, axis=1)

    # 5. Retrouver l'index global du gagnant
    # On sélectionne l'index global correspondant au gagnant local pour chaque ligne
    winners_global_indices = competitors_indices[np.arange(k), winners_local_indices]

    return [individuals[i] for i in winners_global_indices]

# Croisement
def cxOnePoint(ind1, ind2):
    """
    Croisement en un point (Vectorisé).
    """
    size = len(ind1)
    # Choix du point de coupure (entre 1 et size-1)
    cxpoint = np.random.randint(1, size)

    # Swap vectorisé avec copie pour éviter les effets de bord (références partagées)
    # ind1[cxpoint:] renvoie une vue, .copy() assure qu'on stocke les données
    temp = ind1[cxpoint:].copy()
    ind1[cxpoint:] = ind2[cxpoint:]
    ind2[cxpoint:] = temp

    return ind1, ind2


def cxUniform(ind1, ind2, indpb):
    """
    Croisement uniforme (Vectorisé).
    Au lieu de iterer bit par bit, on génère un masque booléen complet.
    """
    size = len(ind1)

    # 1. Génération du masque de swap (True là où on doit échanger)
    # C'est beaucoup plus rapide que d'appeler random() size fois
    swap_mask = np.random.random(size) < indpb

    # 2. Échange vectorisé
    # On utilise un buffer temporaire pour les valeurs de ind1 à ces endroits
    temp = ind1[swap_mask].copy()
    ind1[swap_mask] = ind2[swap_mask]
    ind2[swap_mask] = temp

    return ind1, ind2


def mutFlipBit(individual, indpb):
    """
    Mutation BitFlip (Vectorisée).
    Inverse les bits selon une probabilité, sans aucune boucle Python.
    """
    # 1. Génération du masque de mutation
    mutation_mask = np.random.random(len(individual)) < indpb

    # 2. Application de la mutation par XOR (Bitwise Exclusive OR)
    # Si le masque est True (1), le bit change (0->1, 1->0). Si False (0), il reste.
    # L'opérateur ^= est ultra-rapide en C.
    individual[mutation_mask] ^= 1

    return individual,

def mutKFlip(individual, k):
    """
    Version optimisée avec Numpy.
    Inverse exactement k bits.
    """
    # 1. Génération rapide des k indices uniques
    # replace=False est crucial pour ne pas flipper 2 fois le même bit (ce qui l'annulerait)
    indices = np.random.choice(len(individual), k, replace=False)

    # 2. Application de la mutation
    # Vérification : Si ton individu est déjà un array numpy (idéal), c'est instantané
    if isinstance(individual, np.ndarray):
        individual[indices] ^= 1  # Vectorisation totale : 0->1, 1->0
    else:
        # Si tu utilises des listes standard DEAP, on doit boucler,
        # mais c'est quand même plus rapide grâce au XOR
        for i in indices:
            individual[i] ^= 1

    return individual,


# Générateurs
toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0, 1)

def init_numpy_ind(icls, size):
    # Génère des 0 ou 1 aléatoires directement en vecteur
    return icls(np.random.randint(0, 2, size))

toolbox.register("individual", init_numpy_ind, creator.Individual, ONE_MAX_LENGTH)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: (float(np.sum(ind)),))

# SÉLECTION
toolbox.register("selRandom", selRandom)
toolbox.register("selBest", selBest)
toolbox.register("selTournament", selTournament, tournsize=3)

# CROISEMENT (Crossover)
toolbox.register("cxOnePoint", cxOnePoint)
toolbox.register("cxUniform", cxUniform, indpb=0.5)


toolbox.register("mutBitFlip", mutFlipBit, indpb=1.0/ONE_MAX_LENGTH)
toolbox.register("mut1Flip", mutKFlip, k=1)
toolbox.register("mut3Flip", mutKFlip, k=3)
toolbox.register("mut5Flip", mutKFlip, k=5)


def run_steady_state(pop_size, max_iterations, selection_op, crossover_op, mutation_op, insertion_strategy, pc=1.0, pm=1.0):
    """
    Exécute un AG Steady-State Optimisé NumPy.
    Gain de performance majeur sur les stratégies d'insertion.
    """
    # Initialisation
    population = toolbox.population(n=pop_size)

    # Évaluation initiale vectorisée (si evaluate renvoie un tuple, on map)
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    logbook = tools.Logbook()
    logbook.header = ['gen', 'best_fit']

    # Initialisation stats
    best_ind = tools.selBest(population, 1)[0]
    logbook.record(gen=0, best_fit=best_ind.fitness.values[0])

    # Pointeur pour la stratégie "age" (Remplace le pop(0))
    age_index = 0

    # Boucle principale
    for it in range(1, max_iterations + 1):

        # --- SÉLECTION ---
        # Note: on clone pour éviter de modifier les parents dans la population
        offspring = [toolbox.clone(ind) for ind in selection_op(population, 2)]

        # --- CROISEMENT ---
        if random.random() < pc:
            crossover_op(offspring[0], offspring[1])
            del offspring[0].fitness.values
            del offspring[1].fitness.values

        # --- MUTATION ---
        for ind in offspring:
            if random.random() < pm:
                mutation_op(ind)
                del ind.fitness.values

        # --- ÉVALUATION ---
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # --- INSERTION ---
        if insertion_strategy == "fitness":
            # On extrait les fitness actuelles dans un tableau numpy
            current_fits = np.array([ind.fitness.values[0] for ind in population])

            # np.argpartition met les k plus petits éléments au début (non triés)
            # C'est beaucoup plus rapide (O(N)) que sort (O(N log N))
            worst_indices = np.argpartition(current_fits, 2)[:2]

            # Remplacement direct
            population[worst_indices[0]] = offspring[0]
            population[worst_indices[1]] = offspring[1]

        elif insertion_strategy == "age":
            # On remplace le plus vieux (pointé par age_index)
            population[age_index] = offspring[0]
            age_index = (age_index + 1) % pop_size # Incrément modulo taille

            # On remplace le suivant
            population[age_index] = offspring[1]
            age_index = (age_index + 1) % pop_size

        current_best_fit = np.max([ind.fitness.values[0] for ind in population])
        logbook.record(gen=it, best_fit=current_best_fit)

    return logbook

def lancer_comparatif(titre, param_nom, variantes_dict, params_fixes):
    """
    Fonction générique pour comparer des variantes d'un composant de l'AG.

    :param titre: Titre du graphique (str)
    :param param_nom: Le nom exact de l'argument à varier dans run_steady_state (str)
                      Ex: "selection_op", "crossover_op", "insertion_strategy", "pc"...
    :param variantes_dict: Dictionnaire {"Nom sur le graphe": Valeur de l'argument}
    :param params_fixes: Dictionnaire des autres paramètres fixes pour run_steady_state
    """
    results = {}
    n_runs = params_fixes.get("n_runs", 20)
    iterations = params_fixes.get("max_iterations", 500)

    print(f"--- Comparatif : {titre} ---")

    for label, valeur_variable in variantes_dict.items():
        print(f"  > Test de : {label}")
        fitness_matrix = []

        for r in range(n_runs):
            random.seed(r)
            np.random.seed(r)

            args = params_fixes.copy()
            if "n_runs" in args: del args["n_runs"]

            args[param_nom] = valeur_variable
            lb = run_steady_state(**args)
            fitness_matrix.append([x['best_fit'] for x in lb])

        results[label] = np.mean(fitness_matrix, axis=0)

    plt.figure(figsize=(10, 6))
    x_axis = range(iterations + 1)

    for label, data in results.items():
        plt.plot(x_axis, data, label=label)

    plt.title(titre)
    plt.xlabel("Itérations (Mises à jour)")
    plt.ylabel("Fitness (Meilleur Individu)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()


def lancer_analyse(titre, params_dict):
    """
    Lance un set de runs avec une configuration unique de paramètres.
    Affiche la courbe moyenne et l'écart-type.

    :param titre: Le titre du graphique
    :param params_dict: Le dictionnaire contenant tous les hyperparamètres
                        (pop_size, max_iterations, ops, n_runs, etc.)
    """
    print(f"--- Analyse : {titre} ---")

    n_runs = params_dict.get("n_runs", 20)
    max_iters = params_dict["max_iterations"]
    fitness_history = []

    for r in range(n_runs):
        print(f"Run {r + 1}/{n_runs}...", end="\r")

        random.seed(r)
        np.random.seed(r)
        args = params_dict.copy()
        if "n_runs" in args: del args["n_runs"]
        logbook = run_steady_state(**args)
        fitness_history.append([entry['best_fit'] for entry in logbook])

    print("\nCalcul des statistiques...")

    data_matrix = np.array(fitness_history)
    mean_curve = np.mean(data_matrix, axis=0)
    std_curve = np.std(data_matrix, axis=0)
    x_axis = np.arange(len(mean_curve))
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, mean_curve, label="Fitness Moyenne", color='#2980b9', linewidth=2)
    plt.fill_between(x_axis,
                     mean_curve - std_curve,
                     mean_curve + std_curve,
                     color='#2980b9', alpha=0.2, label="Ecart-type")
    plt.axhline(y=ONE_MAX_LENGTH, color='green', linestyle=':', label="Optimum Global")
    plt.title(f"{titre}\n(Pop={params_dict['pop_size']}, Iters={max_iters})")
    plt.xlabel("Itérations (Mises à jour)")
    plt.ylabel("Fitness (Meilleur Individu)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()


# ============================= Test : selection (Aléa, Best, Tournament) ========================================
params_base = {
    "pop_size": POPULATION_SIZE,
    "max_iterations": MAX_GENERATIONS,
    "crossover_op": toolbox.cxOnePoint,
    "mutation_op": toolbox.mutBitFlip,
    "insertion_strategy": "age",
    "pc": 0.0,
    "pm": 1.0,
    "n_runs": N_RUNS
}

variantes_sel = {
    "Aleatoire": toolbox.selRandom,
    "Best": toolbox.selBest,
    "Tournoi": toolbox.selTournament
}

lancer_comparatif(
    titre=f"Impact de la Sélection (OneMax N={ONE_MAX_LENGTH})",
    param_nom="selection_op",
    variantes_dict=variantes_sel,
    params_fixes=params_base
)

# ============================= Test : Tournoi (3, 20%, 50%, 100%) ========================
variantes_tournoi = {
    "Tournoi Standard (3)": partial(tools.selTournament, tournsize=3),
    "Tournoi 20% (20)": partial(tools.selTournament, tournsize=4),
    "Tournoi 50% (50)": partial(tools.selTournament, tournsize=10),
    "Tournoi 100% (100)": partial(tools.selTournament,tournsize=POPULATION_SIZE)
}

params_tournoi = {
    "pop_size": POPULATION_SIZE,
    "max_iterations": MAX_GENERATIONS,
    "crossover_op": toolbox.cxOnePoint,
    "mutation_op": toolbox.mutBitFlip,
    "insertion_strategy": "age",
    "pc": 1.0,
    "pm": 1.0,
    "n_runs": 20
}

lancer_comparatif(
    titre="Impact de la Taille du Tournoi (Pression de Sélection)",
    param_nom="selection_op",
    variantes_dict=variantes_tournoi,
    params_fixes=params_tournoi
)

# ============================= Test insertion :  (Age, Fitness) ========================================
params_base["selection_op"] = toolbox.selTournament

variantes_ins = {
    "Insertion Age (Remplacer les vieux)": "age",
    "Insertion Fitness (Remplacer les pires)": "fitness"
}

lancer_comparatif(
    titre=f"Impact de la Stratégie d'Insertion (N = {ONE_MAX_LENGTH})",
    param_nom="insertion_strategy",
    variantes_dict=variantes_ins,
    params_fixes=params_base
)

# ============================= Test croisement :  (mono-point, uniforme) ========================================
params_cross = params_base.copy()
params_cross["pc"] = 1.0
params_cross["pm"] = 0.0
params_cross["insertion_strategy"] = "age"

variantes_cross = {
    "Mono-Point": toolbox.cxOnePoint,
    "Uniforme": toolbox.cxUniform
}

lancer_comparatif(
    titre=f"Impact du Type de Croisement (avec Pc=1.0) (N = {ONE_MAX_LENGTH})",
    param_nom="crossover_op",
    variantes_dict=variantes_cross,
    params_fixes=params_cross
)

# ============================= Test mutation :  (1,3,5-flips, bit-flip) ========================================
params_cross = params_base.copy()
params_cross["pc"] = 0.0
params_cross["pm"] = 1.0

variantes_cross = {
    "1-flips": toolbox.mut1Flip,
    "3-flips": toolbox.mut3Flip,
    "5-flips": toolbox.mut5Flip,
    "bit-flips": toolbox.mutBitFlip
}

lancer_comparatif(
    titre=f"Impact du Type de mutation (avec pm=1.0) (N = {ONE_MAX_LENGTH})",
    param_nom="mutation_op",
    variantes_dict=variantes_cross,
    params_fixes=params_cross
)

# # ============================= Test : Taille Population (20, 50, 100, 200) ========================
params_pop = {
    "pop_size": POPULATION_SIZE,
    "max_iterations": MAX_GENERATIONS,
    "selection_op": toolbox.selTournament,
    "crossover_op": toolbox.cxUniform,
    "mutation_op": toolbox.mutBitFlip,
    "insertion_strategy": "fitness",
    "pc": 1.0,
    "pm": 1.0,
    "n_runs": 20
}

variantes_pop = {
    "Pop=20": 20,
    "Pop=50": 50,
    "Pop=100": 100,
    "Pop=200": 200
}

lancer_comparatif(
    titre=f"Impact Taille Population (N = {ONE_MAX_LENGTH})",
    param_nom="pop_size",
    variantes_dict=variantes_pop,
    params_fixes=params_pop
)


# # ============================= Meilleur ========================


params_meilleur = {
    "pop_size": POPULATION_SIZE,
    "max_iterations": MAX_GENERATIONS,
    "selection_op": toolbox.selBest,
    "crossover_op": toolbox.cxUniform,
    "mutation_op": toolbox.mutBitFlip,
    "insertion_strategy": "fitness",
    "pc": 1.0,
    "pm": 1.0,
    "n_runs": 20
}

lancer_analyse("Configuration Optimale", params_meilleur)
