import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools


# =============================================================================
# PRÉPARATION (IMAGE CIBLE)
# =============================================================================
def generate_letter_image(letter, img_shape=(32, 32), font_size=28):
    fig = plt.figure(figsize=(1, 1), dpi=img_shape[0])
    plt.text(0.5, 0.5, letter, fontsize=font_size, ha='center', va='center', color='black')
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)
    return np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


IMG_SHAPE = (32, 32)
NB_PIXELS = 1024
target_img = generate_letter_image('A', IMG_SHAPE)
target_flat = target_img.flatten()

# =============================================================================
# SETUP DEAP (PBIL Binaire)
# =============================================================================
if "FitnessMin" in dir(creator): del creator.FitnessMin
if "Individual" in dir(creator): del creator.Individual

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def eval_img(individual):
    # L'individu est binaire (0 ou 1). On convertit en niveaux de gris (0 ou 255)
    real_pixels = individual * 255
    mse = np.mean((real_pixels.astype(float) - target_flat.astype(float)) ** 2)
    return (mse,)


toolbox.register("evaluate", eval_img)


# =============================================================================
# MOTEUR PBIL
# =============================================================================

def run_pbil_binary(pop_size=100, max_generations=1000, alpha=0.1):
    # Sélection Elite : 10% des meilleurs
    nb_sel = int(pop_size * 0.1)
    if nb_sel < 1: nb_sel = 1

    # Vecteur de probabilités (Init à 0.5 = Incertitude totale)
    probs = np.ones(NB_PIXELS) * 0.5

    logbook = tools.Logbook()
    logbook.header = ['gen', 'best_fit']

    # Stockage du meilleur absolu
    best_ind_ever = None
    best_fit_ever = float('inf')

    print(f"Démarrage PBIL (Pop={pop_size}, Gen={max_generations})...")

    for gen in range(max_generations):

        # SAMPLING (Génération Vectorisée)
        rand_matrix = np.random.random((pop_size, NB_PIXELS))
        # Si rand < proba, alors 1 (Blanc), sinon 0 (Noir)
        pop_matrix = (rand_matrix < probs).astype(np.uint8)
        population = [creator.Individual(row) for row in pop_matrix]

        # EVALUATION
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # SAUVEGARDE DU MEILLEUR (HALL OF FAME)
        current_best = tools.selBest(population, 1)[0]
        current_fit = current_best.fitness.values[0]

        if current_fit < best_fit_ever:
            best_fit_ever = current_fit
            # On stocke l'individu converti en 0-255 pour l'affichage direct
            best_ind_ever = current_best.copy() * 255

            # MISE A JOUR DU MODELE (PBIL Update Rule)
        best_inds = tools.selBest(population, nb_sel)
        best_matrix = np.array(best_inds)

        # Fréquence des '1' dans l'élite
        freq_best = np.mean(best_matrix, axis=0)

        # Lissage : Le vecteur apprend doucement de l'élite
        probs = (1.0 - alpha) * probs + alpha * freq_best

        logbook.record(gen=gen, best_fit=best_fit_ever)
        if gen % 100 == 0:
            print(f"Gen {gen} - Best MSE: {best_fit_ever:.2f}", end='\r')

    return logbook, best_ind_ever


# =============================================================================
# LANCEMENT ET AFFICHAGE
# =============================================================================

logbook, best_ind = run_pbil_binary(pop_size=100, max_generations=1000, alpha=0.1)
best_img = best_ind.reshape(IMG_SHAPE).astype(np.uint8)
final_mse = logbook[-1]['best_fit']
print(f"\nTerminé. Meilleure MSE finale : {final_mse:.2f}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
gen = logbook.select("gen")
fit = logbook.select("best_fit")
plt.plot(gen, fit, color='blue')
plt.title("Convergence (MSE)")
plt.xlabel("Itérations")
plt.ylabel("Erreur (MSE)")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.subplot(1, 3, 2)
plt.imshow(target_img, cmap='gray', vmin=0, vmax=255)
plt.title("Cible")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(best_img, cmap='gray', vmin=0, vmax=255)
plt.title(f"Meilleur Individu\nMSE: {final_mse:.1f}") # On utilise final_mse ici aussi
plt.axis('off')
plt.tight_layout()
plt.show()