import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools

# ==============================================================================
# CONFIGURATION
# ==============================================================================
ONE_MAX_LENGTH = 1000
POPULATION_SIZE = 20
MAX_GENERATIONS = 10000

# Hyper-paramètres des stratégies
ALPHA = 0.2  # Roulette : Vitesse d'oubli
P_MIN = 0.05  # Roulette : Exploration mini
UCB_C = 0.5  # UCB/DMAB : Facteur d'exploration
PH_DELTA = 0.1  # DMAB : Marge de tolérance (Page-Hinkley)
PH_GAMMA = 5.0  # DMAB : Seuil de déclenchement du Restart


# ==============================================================================
# CLASSES DE STRATÉGIES (BANDITS)
# ==============================================================================
class BanditStrategy:
    def __init__(self, n_arms, names):
        self.n_arms = n_arms
        self.names = names
        self.history_probs = {name: [] for name in names}

    def select(self): raise NotImplementedError

    def update(self, arm_idx, reward): pass

    def get_name(self): return self.__class__.__name__



class UniformStrategy(BanditStrategy):
    def select(self):
        self._log_history(1.0 / self.n_arms)  # Log théorique
        return np.random.randint(self.n_arms)

    def _log_history(self, prob):
        for name in self.names: self.history_probs[name].append(prob)



# --- Baseline : Opérateur Unique ---
class SingleOpStrategy(BanditStrategy):
    def __init__(self, n_arms, names, fixed_op_name="1-flip"):
        super().__init__(n_arms, names)
        self.fixed_idx = names.index(fixed_op_name)

    def select(self):
        for i, name in enumerate(self.names):
            self.history_probs[name].append(1.0 if i == self.fixed_idx else 0.0)
        return self.fixed_idx


# --- Stratégie 3 : Roulette Adaptative ---
class AdaptiveRoulette(BanditStrategy):
    def __init__(self, n_arms, names, alpha=0.1, p_min=0.05):
        super().__init__(n_arms, names)
        self.alpha = alpha
        self.p_min = p_min
        self.utilities = np.ones(n_arms)
        self.probs = np.ones(n_arms) / n_arms

    def select(self):
        for i, name in enumerate(self.names):
            self.history_probs[name].append(self.probs[i])
        return np.random.choice(self.n_arms, p=self.probs)

    def update(self, arm_idx, reward):
        self.utilities[arm_idx] = (1 - self.alpha) * self.utilities[arm_idx] + self.alpha * reward
        sum_u = np.sum(self.utilities)
        if sum_u == 0: sum_u = 1.0
        self.probs = self.p_min + (1 - self.n_arms * self.p_min) * (self.utilities / sum_u)


# --- Stratégie 4 : UCB Standard ---
class UCB(BanditStrategy):
    def __init__(self, n_arms, names, c=0.5):
        super().__init__(n_arms, names)
        self.c = c
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0

    def select(self):
        self.total_counts += 1
        # Exploration forcée au début
        for i in range(self.n_arms):
            if self.counts[i] == 0:
                self._log_choice(i)
                return i

        # Calcul scores UCB
        ucb = self.values + self.c * np.sqrt(2 * np.log(self.total_counts) / self.counts)
        choice = np.argmax(ucb)
        self._log_choice(choice)
        return choice

    def _log_choice(self, choice):
        for i, name in enumerate(self.names):
            self.history_probs[name].append(1.0 if i == choice else 0.0)

    def update(self, arm_idx, reward):
        self.counts[arm_idx] += 1
        n = self.counts[arm_idx]
        self.values[arm_idx] = ((n - 1) / n) * self.values[arm_idx] + (1 / n) * reward

    def reset(self):
        """Remise à zéro pour le DMAB"""
        self.counts.fill(0)
        self.values.fill(0)
        self.total_counts = 0


# --- Stratégie 5 : DMAB (UCB + Page-Hinkley) ---
class DMAB(UCB):
    def __init__(self, n_arms, names, c=0.5, delta=0.1, gamma=5.0):
        super().__init__(n_arms, names, c)
        self.delta = delta
        self.gamma = gamma
        # Mémoire Page-Hinkley
        self.ph_mean = np.zeros(n_arms)
        self.ph_sum = np.zeros(n_arms)
        self.ph_min = np.zeros(n_arms)
        self.restarts = 0

    def update(self, arm_idx, reward):
        super().update(arm_idx, reward)  # Mise à jour UCB classique

        # Test de Page-Hinkley (Détection de dégradation)
        # On update la moyenne observée
        n = self.counts[arm_idx]
        self.ph_mean[arm_idx] += (reward - self.ph_mean[arm_idx]) / n

        # On accumule la dérive négative (quand reward < moyenne)
        self.ph_sum[arm_idx] += (reward - self.ph_mean[arm_idx]) + self.delta

        if self.ph_sum[arm_idx] < self.ph_min[arm_idx]:
            self.ph_min[arm_idx] = self.ph_sum[arm_idx]

        # Si la différence dépasse le seuil Gamma -> RESTART
        if self.ph_sum[arm_idx] - self.ph_min[arm_idx] > self.gamma:
            # print(f"DMAB Restart déclenché à cause de {self.names[arm_idx]}!")
            self.restarts += 1
            self.reset()
            # Reset spécifique PH
            self.ph_sum.fill(0)
            self.ph_min.fill(0)
            self.ph_mean.fill(0)


# ==============================================================================
# MOTEUR ÉVOLUTIONNAIRE
# ==============================================================================
if "FitnessMax" in dir(creator): del creator.FitnessMax
if "Individual" in dir(creator): del creator.Individual
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


# Opérateurs demandés
def mutKFlip(ind, k):
    idx = np.random.choice(len(ind), k, replace=False)
    ind[idx] = 1 - ind[idx]
    return ind,


def mutFlipBit(ind, p):
    mask = np.random.random(len(ind)) < p
    ind[mask] = 1 - ind[mask]
    return ind,


def mutIdentity(ind): return ind,  # Opérateur inutile


operators_pool = {
    "bit-flip": lambda ind: mutFlipBit(ind, 1.0 / ONE_MAX_LENGTH),
    "1-flip": lambda ind: mutKFlip(ind, k=1),
    "3-flips": lambda ind: mutKFlip(ind, k=3),
    "5-flips": lambda ind: mutKFlip(ind, k=5),
    "Identity": lambda ind: mutIdentity(ind)  # Test
}
op_names = list(operators_pool.keys())
N_OPS = len(op_names)

toolbox.register("individual", lambda icls, size: icls(np.random.randint(0, 2, size)), creator.Individual,
                 ONE_MAX_LENGTH)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", lambda ind: (float(np.sum(ind)),))


def run_experiment(strategy, label):
    print(f"Exécution : {label}...")
    pop = toolbox.population(n=POPULATION_SIZE)
    for ind in pop: ind.fitness.values = toolbox.evaluate(ind)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("max", np.max)
    logbook = tools.Logbook()

    for gen in range(MAX_GENERATIONS):
        # Sélection & Clonage
        parents = tools.selTournament(pop, 2, tournsize=3)
        offspring = [toolbox.clone(p) for p in parents]
        f_parents_max = max(p.fitness.values[0] for p in parents)

        # Le Bandit choisit
        op_idx = strategy.select()
        op_func = operators_pool[op_names[op_idx]]

        # Application
        reward = 0
        for child in offspring:
            op_func(child)
            child.fitness.values = toolbox.evaluate(child)
            gain = child.fitness.values[0] - f_parents_max
            if gain > 0: reward += gain

        # Le Bandit apprend
        strategy.update(op_idx, reward)

        # Remplacement Steady State
        pop[:] = pop + offspring
        pop.sort(key=lambda x: x.fitness.values[0], reverse=True)
        pop[:] = pop[:POPULATION_SIZE]

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)
        if record["max"] >= ONE_MAX_LENGTH: break

    return logbook, strategy.history_probs


# ==============================================================================
# LANCEMENT ET VISUALISATION
# ==============================================================================

# Instanciation des 5 stratégies
strats = [
    (UniformStrategy(N_OPS, op_names), "Uniforme", 'orange'),
    (SingleOpStrategy(N_OPS, op_names), "Single (1-flip)", 'gray'),
    (AdaptiveRoulette(N_OPS, op_names, alpha=ALPHA, p_min=P_MIN), "Roulette", 'blue'),
    (UCB(N_OPS, op_names, c=UCB_C), "UCB (Statique)", 'green'),
    (DMAB(N_OPS, op_names, c=UCB_C, delta=PH_DELTA, gamma=PH_GAMMA), "DMAB (Dynamique)", 'red')
]

results = {}
for strat_obj, name, color in strats:
    log, probs = run_experiment(strat_obj, name)
    results[name] = {"log": log, "probs": probs, "color": color}

fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3)
ax1 = fig.add_subplot(gs[0, :])
for name, res in results.items():
    log = res["log"]
    ax1.plot(log.select("gen"), log.select("max"), label=name, color=res["color"], linewidth=2)
ax1.set_title("Comparaison des Performances (Fitness)")
ax1.set_xlabel("Itérations")
ax1.set_ylabel("Fitness Max")
ax1.legend()
ax1.grid(True, alpha=0.3)


def plot_probs(ax, probs, title):
    for op in op_names:
        data = probs[op]
        window = 50 if len(data) > 100 else 1
        y_smooth = np.convolve(data, np.ones(window) / window, mode='same')
        ax.plot(y_smooth, label=op)
    ax.set_title(title)
    ax.legend(fontsize='small')
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
plot_probs(ax2, results["Roulette"]["probs"], "Détail : Roulette Adaptative")
ax3 = fig.add_subplot(gs[1, 1])
plot_probs(ax3, results["UCB (Statique)"]["probs"], "Détail : UCB (Standard)")
ax4 = fig.add_subplot(gs[1, 2])
plot_probs(ax4, results["DMAB (Dynamique)"]["probs"], "Détail : DMAB (Avec Restarts)")
plt.tight_layout()
plt.show()

dmab_restarts = strats[4][0].restarts
print(f"\n[INFO] Le DMAB a effectué {dmab_restarts} redémarrages (Restarts) au cours de la recherche.")