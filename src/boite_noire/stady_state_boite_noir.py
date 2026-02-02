import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools


# =============================================================================
# PRÉPARATION DU PROBLÈME (IMAGE CIBLE)
# =============================================================================

def generate_letter_image(letter, img_shape=(32, 32), font_size=28):
    """ Génère l'image cible (Lettre noire sur fond blanc). """
    fig = plt.figure(figsize=(1, 1), dpi=img_shape[0])
    plt.text(0.5, 0.5, letter, fontsize=font_size, ha='center', va='center', color='black')
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # Conversion Niveaux de gris
    img_gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
    return img_gray.astype(np.uint8)


# Paramètres Image
IMG_SHAPE = (32, 32)
NB_PIXELS = IMG_SHAPE[0] * IMG_SHAPE[1]
target_img = generate_letter_image('A', IMG_SHAPE)
target_flat = target_img.flatten()

# =============================================================================
# PARAMÈTRES AG
# =============================================================================
POPULATION_SIZE = 20
MAX_ITERATIONS = 100000


# =============================================================================
# SETUP DEAP (MINIMISATION)
# =============================================================================
if "FitnessMin" in dir(creator): del creator.FitnessMin
if "Individual" in dir(creator): del creator.Individual

# IMPORTANT : weights=(-1.0,) car on veut MINIMISER l'erreur (MSE)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)


# =============================================================================
# OPÉRATEURS VECTORISÉS (ADAPTÉS POUR MINIMISATION & PIXELS)
# =============================================================================

def eval_img(individual):
    """ Calcule la MSE (Mean Squared Error) entre l'individu et la cible. """
    # Astuce : conversion float pour éviter l'overflow des uint8 lors du calcul
    mse = np.mean((individual.astype(float) - target_flat.astype(float)) ** 2)
    return (mse,)


# --- SÉLECTION (Adaptée pour MINIMISATION : on cherche les plus petits scores) ---

def selBest(individuals, k):
    """ Sélection élitiste (Minimisation). """
    fits = np.array([ind.fitness.values[0] for ind in individuals])
    # Argsort trie du plus petit au plus grand.
    # Pour minimiser, on prend les k premiers (les plus petits)
    top_indices = np.argsort(fits)[:k]
    return [individuals[i] for i in top_indices]


def selTournament(individuals, k, tournsize):
    """ Tournoi vectorisé (Minimisation). """
    n_individuals = len(individuals)
    fits = np.array([ind.fitness.values[0] for ind in individuals])

    competitors_indices = np.random.randint(0, n_individuals, (k, tournsize))
    competitors_fitnesses = fits[competitors_indices]

    # ArgMIN car on veut le plus petit MSE
    winners_local_indices = np.argmin(competitors_fitnesses, axis=1)
    winners_global_indices = competitors_indices[np.arange(k), winners_local_indices]

    return [individuals[i] for i in winners_global_indices]


# --- CROISEMENT  ---
def cxOnePoint(ind1, ind2):
    size = len(ind1)
    cxpoint = np.random.randint(1, size)
    temp = ind1[cxpoint:].copy()
    ind1[cxpoint:] = ind2[cxpoint:]
    ind2[cxpoint:] = temp
    return ind1, ind2

def cxUniform(ind1, ind2, indpb):
    """
    Croisement uniforme (Vectorisé).
    Au lieu de iterer, on génère un masque booléen complet.
    """
    size = len(ind1)
    swap_mask = np.random.random(size) < indpb
    temp = ind1[swap_mask].copy()
    ind1[swap_mask] = ind2[swap_mask]
    ind2[swap_mask] = temp

    return ind1, ind2

# --- MUTATION (Nouvelle : Modification de pixels) ---
def mutPixelRandom(individual, indpb):
    """
    Remplace des pixels par une valeur aléatoire [0, 255].
    Équivalent du BitFlip pour des entiers.
    """
    size = len(individual)
    # Masque des gènes à muter
    mask = np.random.random(size) < indpb

    # On assigne une nouvelle valeur aléatoire aux pixels sélectionnés
    # np.random.randint(0, 256) génère des entiers [0, 255]
    individual[mask] = np.random.randint(0, 256, size=np.sum(mask))

    return individual,


# =============================================================================
# TOOLBOX
# =============================================================================
toolbox = base.Toolbox()

# Génération d'attributs : Pixels entre 0 et 255
toolbox.register("attr_pixel", random.randint, 0, 255)


# Initialisation individu : Array numpy de taille NB_PIXELS
def init_numpy_ind(icls, size):
    return icls(np.random.randint(0, 256, size, dtype=np.uint8))


toolbox.register("individual", init_numpy_ind, creator.Individual, NB_PIXELS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", eval_img)

# Enregistrement des opérateurs
toolbox.register("selBest", selBest)
toolbox.register("selTournament", selTournament, tournsize=3)
toolbox.register("cxOnePoint", cxOnePoint)
toolbox.register("cxUniform", cxUniform, indpb=0.5)

# Taux de mutation : 1/L est classique (ici 1/1024)
toolbox.register("mutPixel", mutPixelRandom, indpb=1.0 / NB_PIXELS)


# =============================================================================
# MOTEUR STEADY STATE
# =============================================================================
def run_steady_state_img(pop_size, max_iterations, selection_op, crossover_op, mutation_op, insertion_strategy, pc=1.0,
                         pm=1.0):
    # Init Population
    population = toolbox.population(n=pop_size)

    # Évaluation Initiale
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    logbook = tools.Logbook()
    logbook.header = ['gen', 'best_fit']

    # Stats initiales (Minimisation -> min)
    best_ind = tools.selBest(population, 1)[0]
    logbook.record(gen=0, best_fit=best_ind.fitness.values[0])

    age_index = 0

    for it in range(1, max_iterations + 1):
        # SÉLECTION (Clonage des parents)
        offspring = [toolbox.clone(ind) for ind in selection_op(population, 2)]

        # CROISEMENT
        if random.random() < pc:
            crossover_op(offspring[0], offspring[1])
            del offspring[0].fitness.values
            del offspring[1].fitness.values

        # MUTATION
        for ind in offspring:
            if random.random() < pm:
                mutation_op(ind)
                del ind.fitness.values

        # ÉVALUATION
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)

        # INSERTION
        if insertion_strategy == "fitness":
            current_fits = np.array([ind.fitness.values[0] for ind in population])

            # REPLACE WORST : On cherche les plus GRANDS scores (pire erreur)
            # argpartition met les k plus grands à la fin
            # On veut les indices des 2 plus grands
            worst_indices = np.argpartition(current_fits, -2)[-2:]

            population[worst_indices[0]] = offspring[0]
            population[worst_indices[1]] = offspring[1]

        elif insertion_strategy == "age":
            population[age_index] = offspring[0]
            age_index = (age_index + 1) % pop_size
            population[age_index] = offspring[1]
            age_index = (age_index + 1) % pop_size

        # Enregistrement (Minimisation -> min)
        current_best_fit = np.min([ind.fitness.values[0] for ind in population])

        # Log réduit pour ne pas spammer la console (tous les 1000 iters)
        if it % 1000 == 0:
            print(f"Iter {it}/{max_iterations} - Best MSE: {current_best_fit:.2f}", end='\r')

        logbook.record(gen=it, best_fit=current_best_fit)

    return logbook, population


# =============================================================================
# VISUALISATION
# =============================================================================

# Paramètres choisis pour le test
params = {
    "pop_size": POPULATION_SIZE,
    "max_iterations": MAX_ITERATIONS,
    "selection_op": toolbox.selTournament,  # Tournoi est robuste
    "crossover_op": toolbox.cxUniform,
    "mutation_op": toolbox.mutPixel,
    "insertion_strategy": "fitness",  # Convergence plus rapide
    "pc": 0.9,
    "pm": 1.0  # Mutation systématique (taux par gène géré par indpb)
}

print("--- Démarrage de l'optimisation d'image ---")
logbook, final_pop = run_steady_state_img(**params)

# Récupération du meilleur individu final
best_ind = toolbox.selBest(final_pop, 1)[0]
best_img = best_ind.reshape(IMG_SHAPE)

print(f"\nTerminé. Meilleure MSE finale : {best_ind.fitness.values[0]:.2f}")

# --- AFFICHAGE ---
plt.figure(figsize=(12, 4))

# Courbe de convergence
plt.subplot(1, 3, 1)
gen = logbook.select("gen")
fit = logbook.select("best_fit")
plt.plot(gen, fit, color='blue')
plt.title("Convergence (MSE)")
plt.xlabel("Itérations")
plt.ylabel("Erreur (MSE)")
plt.yscale('log')  # Échelle log souvent plus lisible pour MSE
plt.grid(True, alpha=0.3)

# Image Cible
plt.subplot(1, 3, 2)
plt.imshow(target_img, cmap='gray', vmin=0, vmax=255)
plt.title("Cible (Target)")
plt.axis('off')

# Meilleure Image Trouvée
plt.subplot(1, 3, 3)
plt.imshow(best_img, cmap='gray', vmin=0, vmax=255)
plt.title(f"Meilleur Individu\nMSE: {best_ind.fitness.values[0]:.1f}")
plt.axis('off')

plt.tight_layout()
plt.show()