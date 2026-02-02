import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from deap import base, creator, tools


# ==============================================================================
# CLASSE TSP (Chargement Local Dynamique)
# ==============================================================================
class TravelingSalesmanProblem:
    def __init__(self, name):
        self.name = name
        self.locations = []
        self.distances = []
        self.tspSize = 0
        self.__loadLocalData()

    def __loadLocalData(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "..", "tsp-data")

        loc_file = os.path.join(data_dir, f"{self.name}-loc.pickle")
        dist_file = os.path.join(data_dir, f"{self.name}-dist.pickle")

        if not os.path.exists(loc_file):
            raise FileNotFoundError(f"Fichier non trouvé: {loc_file}")

        with open(loc_file, 'rb') as f:
            self.locations = pickle.load(f)
        with open(dist_file, 'rb') as f:
            self.distances = pickle.load(f)

        self.tspSize = len(self.locations)

    def getTotalDistance(self, indices):
        distance = 0
        for i in range(self.tspSize):
            distance += self.distances[indices[i]][indices[(i + 1) % self.tspSize]]
        return distance

    def plotData(self, indices, ax):
        x = [self.locations[i][0] for i in indices] + [self.locations[indices[0]][0]]
        y = [self.locations[i][1] for i in indices] + [self.locations[indices[0]][1]]
        ax.plot(x, y, 'o-', mfc='r', markersize=3, linewidth=1)
        ax.set_aspect('equal')

# ==============================================================================
# PARAMÈTRES Q-LEARNING (Demandés)
# ==============================================================================
QL_ALPHA = 0.8  # Taux d'apprentissage
QL_GAMMA = 0.99  # Facteur d'actualisation (plus de poids sur le futur)
QL_EPSILON = 0.01  # Exploration (très faible, on privilégie l'exploitation)
N_ZONES = 10  # Discrétisation de la Fitness


# ==============================================================================
# CONFIGURATION DEAP
# ==============================================================================
if "FitnessMin" in dir(creator): del creator.FitnessMin
if "Individual" in dir(creator): del creator.Individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin,
               prev_fit=float, current_island=int)


# ==============================================================================
# AGENT Q-LEARNING MIS À JOUR
# ==============================================================================
class QLearningAgent:
    def __init__(self, n_islands, n_zones, alpha, gamma, epsilon):
        self.n_islands = n_islands
        self.n_zones = n_zones
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Table Q[état_source, zone_fitness, action_destination]
        self.q_table = np.random.uniform(0, 0.01, (n_islands, n_zones, n_islands))
        self.initial_baseline = None

    def get_state(self, fitness):
        if self.initial_baseline is None or self.initial_baseline == 0:
            return 0
        # Calcul du gain relatif par rapport au début
        improvement = (self.initial_baseline - fitness) / self.initial_baseline
        zone = int(improvement * self.n_zones * 1.5)
        return max(0, min(zone, self.n_zones - 1))

    def choose_action(self, island_idx, fitness):
        # Epsilon-Greedy
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_islands)
        zone = self.get_state(fitness)
        return np.argmax(self.q_table[island_idx, zone])

    def learn(self, old_isl, old_fit, new_isl, new_fit):
        old_zone = self.get_state(old_fit)
        new_zone = self.get_state(new_fit)
        reward = max(0, old_fit - new_fit)  # On récompense uniquement l'amélioration

        current_q = self.q_table[old_isl, old_zone, new_isl]
        max_next_q = np.max(self.q_table[new_isl, new_zone])

        # Formule du Q-Learning avec les nouveaux paramètres
        self.q_table[old_isl, old_zone, new_isl] += self.alpha * (reward + self.gamma * max_next_q - current_q)


# Opérateurs
def mutSwap(ind):
    idx1, idx2 = random.sample(range(len(ind)), 2)
    ind[idx1], ind[idx2] = ind[idx2], ind[idx1]
    return ind,


def mutScramble(ind):
    tools.mutShuffleIndexes(ind, indpb=0.05)
    return ind,


def mutInversion(ind):
    a, b = random.sample(range(len(ind)), 2)
    if a > b: a, b = b, a
    ind[a:b + 1] = ind[a:b + 1][::-1]
    return ind,



operators_map = {
    0: ("Swap", mutSwap),
    1: ("Scramble", mutScramble),
    2: ("2-Opt", mutInversion)
}


# ==============================================================================
# FONCTION D'EXÉCUTION
# ==============================================================================
def run_instance(instance_name):
    print(f"\nExécution : {instance_name}...")
    tsp = TravelingSalesmanProblem(instance_name)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(tsp.tspSize), tsp.tspSize)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (tsp.getTotalDistance(ind),))

    POP_SIZE = 50 if tsp.tspSize < 100 else 100
    MAX_GEN = 2500 if tsp.tspSize < 100 else 5000

    # Instanciation de l'agent avec tes paramètres
    agent = QLearningAgent(len(operators_map), N_ZONES, QL_ALPHA, QL_GAMMA, QL_EPSILON)

    population = toolbox.population(n=POP_SIZE)

    for i, ind in enumerate(population):
        ind.fitness.values = toolbox.evaluate(ind)
        ind.prev_fit = ind.fitness.values[0]
        ind.current_island = i % len(operators_map)

    agent.initial_baseline = np.mean([ind.fitness.values[0] for ind in population])

    history_dist = []
    history_migr = {op[0]: [] for op in operators_map.values()}

    for gen in range(MAX_GEN):
        offspring = [toolbox.clone(ind) for ind in population]

        for ind in offspring:
            _, op_func = operators_map[ind.current_island]
            op_func(ind)
            del ind.fitness.values

        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        for i in range(len(offspring)):
            child, parent = offspring[i], population[i]
            agent.learn(parent.current_island, parent.prev_fit,
                        child.current_island, child.fitness.values[0])
            child.current_island = agent.choose_action(child.current_island, child.fitness.values[0])
            child.prev_fit = child.fitness.values[0]

        for i in range(len(population)):
            if offspring[i].fitness.values[0] < population[i].prev_fit:
                population[i] = offspring[i]
            else:
                population[i].current_island = agent.choose_action(population[i].current_island, population[i].prev_fit)

        best_val = min(ind.fitness.values[0] for ind in population)
        history_dist.append(best_val)

        counts = np.zeros(len(operators_map))
        for ind in population: counts[ind.current_island] += 1
        for idx, count in enumerate(counts):
            history_migr[operators_map[idx][0]].append(count)

        if gen % 1000 == 0:
            print(f"  Gen {gen} | Meilleur : {best_val:.2f}")

    return history_dist, history_migr, tools.selBest(population, 1)[0], tsp


# ==============================================================================
# LANCEMENT
# ==============================================================================
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "..", "tsp-data", "*-loc.pickle")
instance_names = [os.path.basename(f).replace("-loc.pickle", "") for f in glob.glob(data_path)]
instance_names.sort()

for name in instance_names:
    h_dist, h_migr, best_sol, tsp_obj = run_instance(name)

    # Plotting simplifié pour la console/notebook
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Instance : {name} (Alpha={QL_ALPHA}, Gamma={QL_GAMMA})", fontsize=16)

    ax1.plot(h_dist)
    ax1.set_title("Convergence Distance")

    labels = list(h_migr.keys())
    ax2.stackplot(range(len(h_dist)), *[h_migr[k] for k in labels], labels=labels)
    ax2.set_title("Répartition des Opérateurs")
    ax2.legend(loc='upper right')

    tsp_obj.plotData(best_sol, ax3)
    ax3.set_title("Meilleur Chemin Trouvé")

    plt.tight_layout()
    plt.show()