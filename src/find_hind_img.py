import os
import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools


def generate_letter_image(letter, img_shape=(32, 32), font_size=28):
    """
    Génère une image numpy en niveaux de gris contenant la lettre spécifiée,
    lettre noire sur fond blanc.
    """
    fig = plt.figure(figsize=(1, 1), dpi=img_shape[0])
    plt.text(0.5, 0.5, letter, fontsize=font_size, ha='center', va='center', color='black')
    plt.gca().set_facecolor('white')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.canvas.draw()
    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    argb = argb.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    plt.close(fig)

    # ARGB -> RGB (ignorer alpha) puis niveaux de gris
    rgb = argb[..., 1:4]
    img_gray = np.dot(rgb, [0.299, 0.587, 0.114]).astype(np.uint8)
    return img_gray


def eval_img(individual):
    arr = np.array(individual, dtype=np.uint8)
    img = arr.reshape(IMG_SHAPE)
    mse = np.mean((img.astype(np.float32) - target_img.astype(np.float32)) ** 2)
    return (float(mse),)


# Image cible et constantes
target_img = generate_letter_image('A', img_shape=(32, 32))
IMG_SHAPE = target_img.shape
PIXELS = IMG_SHAPE[0] * IMG_SHAPE[1]

# DEAP setup
if not hasattr(creator, 'FitnessMin'):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, 'Individual'):
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_pixel", random.randint, 0, 255)
toolbox.register("individual", lambda: creator.Individual([toolbox.attr_pixel() for _ in range(PIXELS)]))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", eval_img)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutUniformInt, low=0, up=255, indpb=1 / PIXELS)


def run_evolution(n_pop=30, n_gen=2000, cxpb=0.5, mutpb=0.2, disp_every=100, save_frames=False):
    random.seed(42)
    pop = toolbox.population(n=n_pop)
    hof = tools.HallOfFame(1)

    # Évaluation initiale
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    hof.update(pop)

    # Historique des meilleures MSE
    mse_history = []
    best_mse = hof[0].fitness.values[0]
    mse_history.append(best_mse)
    print(f'gen=0 nevals={len(invalid)} mse={best_mse:.4f}')

    # Figure interactive
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].set_title('Target')
    axes[0].imshow(target_img, cmap='gray')
    axes[0].axis('off')
    axes[1].axis('off')
    best_img = np.array(hof[0], dtype=np.uint8).reshape(IMG_SHAPE)
    im_best = axes[1].imshow(best_img, cmap='gray')
    axes[1].set_title(f'Gen 0  mse={best_mse:.4f}')
    fig.canvas.draw()
    plt.pause(0.001)

    if save_frames:
        os.makedirs('frames', exist_ok=True)
        fig.savefig('frames/gen_000.png', dpi=150)

    # Boucle d’évolution
    for gen in range(1, n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Croisement
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Mutation
        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # Réévaluation
        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)
        best_mse = hof[0].fitness.values[0]
        mse_history.append(best_mse)
        print(f'gen={gen} nevals={len(invalid)} mse={best_mse:.4f}')

        # Mise à jour de l’affichage toutes les `disp_every` générations (et dernière)
        if gen % disp_every == 0 or gen == n_gen:
            best_img = np.array(hof[0], dtype=np.uint8).reshape(IMG_SHAPE)
            im_best.set_data(best_img)
            axes[1].set_title(f'Gen {gen}  mse={best_mse:.4f}')
            fig.canvas.draw()
            plt.pause(0.001)
            if save_frames:
                fig.savefig(f'frames/gen_{gen:03d}.png', dpi=150)

    # Sauvegardes et plots finaux
    plt.ioff()
    fig.savefig('evolved_vs_target.png', dpi=150)

    # Courbe MSE
    fig_mse, axm = plt.subplots(1, 1, figsize=(6, 3))
    axm.plot(range(len(mse_history)), mse_history, color='tab:blue')
    axm.set_xlabel('Génération')
    axm.set_ylabel('MSE')
    axm.set_title('Évolution de la MSE')
    axm.grid(True, alpha=0.3)
    fig_mse.tight_layout()
    fig_mse.savefig('mse_curve.png', dpi=150)

    plt.show()


if __name__ == '__main__':
    run_evolution(n_pop=30, n_gen=10000, cxpb=0.5, mutpb=0.2, disp_every=100, save_frames=False)