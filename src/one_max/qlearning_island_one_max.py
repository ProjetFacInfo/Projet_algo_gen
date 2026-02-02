import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
import random

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ONE_MAX_LENGTH = 1000
TOTAL_POP_SIZE = 100
MAX_GENERATIONS = 10000

# Paramètres Q-Learning
QL_ALPHA = 0.8  # Taux d'apprentissage
QL_GAMMA = 0.99  # Facteur d'actualisation
QL_EPSILON = 0.01  # Exploration
N_ZONES = 10  # Discrétisation Fitness

# ==============================================================================
# SETUP DEAP & OPÉRATEURS
# ==============================================================================
if "FitnessMax" in dir(creator): del creator.FitnessMax
if "Individual" in dir(creator): del creator.Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax,
               prev_fit=float, current_island=int)

toolbox = base.Toolbox()


# --- Opérateurs ---
def mutKFlip(ind, k):
    idx = np.random.choice(len(ind), k, replace=False)
    ind[idx] = 1 - ind[idx]
    return ind,


def mutFlipBit(ind, p):
    mask = np.random.random(len(ind)) < p
    ind[mask] = 1 - ind[mask]
    return ind,


def mutIdentity(ind):
    return ind,


# Mapping des îles
operators_map = {
    0: ("bit-flip", lambda ind: mutFlipBit(ind, 1.0 / ONE_MAX_LENGTH)),
    1: ("1-flip", lambda ind: mutKFlip(ind, k=1)),
    2: ("3-flips", lambda ind: mutKFlip(ind, k=3)),
    3: ("5-flips", lambda ind: mutKFlip(ind, k=5)),
    4: ("Identity", lambda ind: mutIdentity(ind))
}
N_ISLANDS = len(operators_map)

toolbox.register("individual", lambda icls, size: icls(np.random.randint(0, 2, size)), creator.Individual,
                 ONE_MAX_LENGTH)
toolbox.register("evaluate", lambda ind: (float(np.sum(ind)),))


# ==============================================================================
# AGENT Q-LEARNING
# ==============================================================================
class QLearningAgent:
    def __init__(self, n_islands, n_zones):
        self.n_islands = n_islands
        self.n_zones = n_zones
        # Init aléatoire léger
        self.q_table = np.random.uniform(low=0.0, high=0.001, size=(n_islands, n_zones, n_islands))

    def get_state(self, island_idx, fitness):
        zone = int((fitness / ONE_MAX_LENGTH) * self.n_zones)
        if zone >= self.n_zones: zone = self.n_zones - 1
        return island_idx, zone

    def choose_action(self, island_idx, fitness):
        # Epsilon-Greedy
        if np.random.random() < QL_EPSILON:
            return np.random.randint(self.n_islands)
        _, zone = self.get_state(island_idx, fitness)
        return np.argmax(self.q_table[island_idx, zone])

    def learn(self, old_island, old_fitness, action_island, reward, new_fitness):
        _, old_zone = self.get_state(old_island, old_fitness)
        _, new_zone = self.get_state(action_island, new_fitness)

        current_q = self.q_table[old_island, old_zone, action_island]
        max_next_q = np.max(self.q_table[action_island, new_zone])

        # Bellman Equation
        new_q = current_q + QL_ALPHA * (reward + QL_GAMMA * max_next_q - current_q)
        self.q_table[old_island, old_zone, action_island] = new_q


# ==============================================================================
# SIMULATION
# ==============================================================================
def run_simulation():
    population = [toolbox.individual() for _ in range(TOTAL_POP_SIZE)]
    agent = QLearningAgent(N_ISLANDS, N_ZONES)

    # Init Population
    for i, ind in enumerate(population):
        ind.fitness.values = toolbox.evaluate(ind)
        ind.prev_fit = ind.fitness.values[0]
        ind.current_island = i % N_ISLANDS

    island_history = {op_name: [] for _, (op_name, _) in operators_map.items()}

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    stats.register("avg", np.mean)
    logbook = tools.Logbook()

    print("Démarrage de la simulation Q-Learning...")

    for gen in range(MAX_GENERATIONS):
        # Mutation (Locale sur l'île)
        offspring = [toolbox.clone(ind) for ind in population]
        for ind in offspring:
            _, op_func = operators_map[ind.current_island]
            op_func(ind)
            del ind.fitness.values

        # Evaluation
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)

        # Apprentissage & Migration
        for i in range(len(offspring)):
            ind = offspring[i]
            parent = population[i]

            # Reward Clipped (0 si pas d'amélioration)
            raw_gain = ind.fitness.values[0] - parent.prev_fit
            reward = max(0, raw_gain)

            agent.learn(
                old_island=parent.current_island,
                old_fitness=parent.prev_fit,
                action_island=ind.current_island,
                reward=reward,
                new_fitness=ind.fitness.values[0]
            )

            # Décision pour le futur
            ind.current_island = agent.choose_action(ind.current_island, ind.fitness.values[0])
            ind.prev_fit = ind.fitness.values[0]

        # Remplacement (Elitisme)
        for i in range(len(population)):
            if offspring[i].fitness.values[0] >= population[i].prev_fit:
                population[i] = offspring[i]
            else:
                # Le parent reste mais met à jour sa destination
                new_dest = agent.choose_action(population[i].current_island, population[i].prev_fit)
                population[i].current_island = new_dest

        record = stats.compile(population)
        logbook.record(gen=gen, **record)

        counts = np.zeros(N_ISLANDS)
        for ind in population: counts[ind.current_island] += 1
        for idx, count in enumerate(counts):
            island_history[operators_map[idx][0]].append(count)

        if record["max"] >= ONE_MAX_LENGTH:
            print(f"-> Solution trouvée à la génération {gen}")
            break

    return logbook, island_history


log, history = run_simulation()

# ==============================================================================
# VISUALISATION
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
gen = log.select("gen")
fit_max = np.array(log.select("max"))
fit_avg = np.array(log.select("avg"))
ax1.plot(gen, fit_max, label="Best Fitness", color='#d62728', linewidth=2)
ax1.plot(gen, fit_avg, label="Avg Fitness", color='#ff7f0e', linestyle='--', alpha=0.8)
ax1.set_ylabel("Fitness (0 - 1000)")
ax1.set_ylim(bottom=0, top=1050)  # On laisse une petite marge en haut
ax1.set_title(f"Convergence Q-Learning (Max atteint : {fit_max[-1]:.0f})")
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)
labels = list(history.keys())
data = [history[k] for k in labels]
colors = ['#1f77b4', '#aec7e8', '#2ca02c', '#98df8a', '#9467bd']
ax2.stackplot(range(len(data[0])), *data, labels=labels, colors=colors, alpha=0.85)
ax2.set_ylabel("Population par Île")
ax2.set_xlabel("Générations")
ax2.set_title("Répartition de la Population (Choix de l'opérateur)")
ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
ax2.margins(0, 0)
plt.tight_layout()
plt.show()