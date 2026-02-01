import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools


# =============================================================================
# SETUP TARGET
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
# FONCTION D'EVALUATION
# =============================================================================
def evaluate_binary_mse(candidate_binary):
    """
    candidate_binary : vecteur de 0 et 1
    Retourne la MSE par rapport à la cible (0 et 255)
    """
    real_pixels = candidate_binary * 255
    return np.mean((real_pixels.astype(float) - target_flat.astype(float)) ** 2)


# =============================================================================
# MOTEUR CGA (Compact Genetic Algorithm)
# =============================================================================

def run_cga_image(virtual_pop_size=500, max_iterations=30000):
    """
    cGA pour reconstruction d'image.
    step = 1 / virtual_pop_size
    """
    # Pas de mise à jour (Update Step)
    step = 1.0 / virtual_pop_size

    # Vecteur de probabilités (Init à 0.5)
    probs = np.ones(NB_PIXELS) * 0.5

    # Historique pour le graphe
    history_best_fit = []
    iterations_axis = []

    best_ind_ever = None
    best_fit_ever = float('inf')

    print(f"Démarrage cGA (Virtual Pop={virtual_pop_size}, Iter={max_iterations})...")

    # Boucle principale (Duel par Duel)
    for it in range(max_iterations):

        # Génération de 2 Candidats (Vectorisé)
        # On tire 2 masques aléatoires pour comparer aux probas
        rand_a = np.random.random(NB_PIXELS)
        rand_b = np.random.random(NB_PIXELS)

        cand_a = (rand_a < probs).astype(np.uint8)
        cand_b = (rand_b < probs).astype(np.uint8)

        # Evaluation
        fit_a = evaluate_binary_mse(cand_a)
        fit_b = evaluate_binary_mse(cand_b)

        # Tournoi (Minimisation MSE)
        if fit_a <= fit_b:
            winner, loser = cand_a, cand_b
            current_best_mse = fit_a
        else:
            winner, loser = cand_b, cand_a
            current_best_mse = fit_b

        # Mise à jour du "Hall of Fame"
        if current_best_mse < best_fit_ever:
            best_fit_ever = current_best_mse
            best_ind_ever = winner.copy() * 255  # Sauvegarde en 0-255

        # Mise à jour du vecteur de probabilité
        if fit_a != fit_b:
            # On ne met à jour que là où les gènes sont différents
            diff_mask = (winner != loser)

            # Formule cGA : Probs += 1/n vers le gagnant
            # Si winner[i]=1, on ajoute step. Si winner[i]=0, on enlève step.
            # Astuce math : (2*winner - 1) donne +1 ou -1
            update_direction = (2.0 * winner[diff_mask] - 1.0) * step

            probs[diff_mask] += update_direction

            # Clamping pour ne pas dépasser [0, 1] - Important sinon bug
            probs = np.clip(probs, 0.0, 1.0)

        # Log (On n'enregistre pas tout le temps pour ne pas alourdir la mémoire)
        if it % 100 == 0:
            history_best_fit.append(best_fit_ever)
            iterations_axis.append(it)
            print(f"Iter {it} - Best MSE: {best_fit_ever:.2f}", end='\r')

    return iterations_axis, history_best_fit, best_ind_ever


# =============================================================================
# LANCEMENT ET AFFICHAGE
# =============================================================================
iters, fits, best_ind = run_cga_image(virtual_pop_size=1000, max_iterations=40000)
best_img = best_ind.reshape(IMG_SHAPE).astype(np.uint8)
final_mse = fits[-1]
print(f"\nTerminé. Meilleure MSE finale : {final_mse:.2f}")
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(iters, fits, color='purple')
plt.title("Convergence cGA")
plt.xlabel("Itérations (Duels)")
plt.ylabel("MSE")
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.subplot(1, 3, 2)
plt.imshow(target_img, cmap='gray', vmin=0, vmax=255)
plt.title("Cible")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(best_img, cmap='gray', vmin=0, vmax=255)
plt.title(f"Meilleur cGA\nMSE: {final_mse:.1f}")
plt.axis('off')
plt.tight_layout()
plt.show()