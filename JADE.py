from random import uniform
from scipy.stats import cauchy
import numpy as np
from sklearn.metrics import pairwise_distances

# --- FUNCTIONS ----------------------------------------------------------------+


def ensure_bounds(vec, bounds):
    vec = np.asarray(vec)
    vec_new = np.copy(vec)
    # cycle through each variable in vector
    bounds = np.asarray(bounds)
    indx_low = np.where(vec < bounds[:, 0])
    indx_high = np.where(vec > bounds[:, 1])
    vec_new[indx_low] = bounds[:, 0][indx_low]
    vec_new[indx_high] = bounds[:, 1][indx_high]
    return vec_new


def recombine(x_parent, x_trail, cr):
    v_trial = np.zeros(len(x_parent))
    j_random = np.random.randint(0, len(x_parent))
    for j in range(len(x_parent)):
        crossover = np.random.random()
        if crossover <= cr or j_random == j:
            v_trial[j] = x_trail[j]
        else:
            v_trial[j] = x_parent[j]
    return v_trial


# --- MAIN ---------------------------------------------------------------------+

def minimize(cost_func, bounds, popsize, maxiter=100, mu_CR=0.5, mu_F=0.5, p_best=0.05, scale_c=0.1,
             n_novelty_solution=10, n_neighbors=15):
    # --- INITIALIZE A POPULATION (step #1) ----------------+
    n_p_best = int(p_best * popsize)
    population = []
    for i in range(0, popsize):
        indv  = []
        for j in range(len(bounds)):
            indv.append(uniform(bounds[j][0], bounds[j][1]))
        population.append(indv)
    population = np.asarray(population)
    # --- SOLVE --------------------------------------------+

    # cycle through each generation (step #2)
    best_solution_ = None
    bset_fitness_all = []
    archive_A = np.empty([0, len(bounds)])
    # # evaluate fitness
    fitness_parents = []
    for i in range(popsize):
        fitness_parents.append(cost_func(population[i]))
    fitness_parents = np.asarray(fitness_parents)
    for i in range(1, maxiter + 1):
        # print("GENERATION:", i)
        # print(population)
        gen_scores = []  # score_Cora keeping
        succesed_set_A = []
        failed_set_B = []
        failed_pop_fittness = []
        failed_indx = []
        CR_archive, F_archive = [], []
        # cycle through each individual in the population

        CR_gen = np.random.normal(loc=mu_CR, scale=0.1, size=popsize)
        CR_gen[np.where(CR_gen > 1)] = 1
        CR_gen[np.where(CR_gen < 0)] = 0
        F_gen = cauchy.rvs(loc=mu_F, scale=0.1, size=popsize)
        while len(np.where(F_gen < 0)[0]) != 0:
            F_gen[np.where(F_gen < 0)[0][0]] = cauchy.rvs(loc=mu_F, scale=0.1)
        F_gen[np.where(F_gen > 1)] = 1

        CR_gen = CR_gen[np.argsort(CR_gen)]

        for j in range(0, popsize):
            best_indx_gen = np.argsort(fitness_parents)[:n_p_best]
            # --- MUTATION (step #3.A) ---------------------+
            # select three random vector index positions [0, popsize), not including current vector (j)
            candidates_1 = list(range(0, popsize))
            candidates_1.remove(j)
            indx_1 = np.random.choice(candidates_1, 1, replace=False)[0]
            pop_temp = np.append(population, archive_A,  axis=0)
            candidates_2 = list(range(0, len(pop_temp)))
            candidates_2.remove(j), candidates_2.remove(indx_1)
            indx_2 = np.random.choice(candidates_2, 1, replace=False)[0]
            x_1 = population[indx_1]
            x_2 = pop_temp[indx_2]
            x_parent = population[j]  # target individual
            random_best_index = np.random.choice(best_indx_gen, 1, replace=False)[0]
            x_best = population[random_best_index]
            # # x_t + F(i)*(Xpbest-X(i,:))+F(i)*(Xr1-Xr2);
            x_diff = x_parent + F_gen[j] * (x_best - x_parent) + F_gen[j] * (x_1 - x_2)
            v_donor = ensure_bounds(x_diff, bounds)

            # --- RECOMBINATION (step #3.B) ----------------+
            v_trial = recombine(x_parent, v_donor, CR_gen[j])

            # --- GREEDY SELECTION (step #3.C) -------------+
            score_trial = cost_func(v_trial)
            score_target = fitness_parents[j]

            if score_trial < score_target:
                population[j] = v_trial
                gen_scores.append(score_trial)
                fitness_parents[j] = score_trial
                # print('   >', score_trial, v_trial)
                # print('   >', score_trial)
                # # add set A
                succesed_set_A.append(v_trial)
                # # archive parents vector and CR/F
                archive_A = np.append(archive_A, x_parent.reshape([1, -1]), axis=0)
                CR_archive.append(CR_gen[j])
                F_archive.append(F_gen[j])
            else:
                # print('   >', score_target, x_parent)
                # print('   >', score_target)
                gen_scores.append(score_target)
                # # add set B
                failed_set_B.append(v_trial)
                failed_indx.append(j)
                failed_pop_fittness.append(score_trial)

        # ----------- novelty search ----------------------+
        replaced_indx = novelty_search(succesed_set_A, failed_set_B, n_novelty_solution, n_neighbors)
        population[failed_indx][replaced_indx] = np.asarray(failed_set_B)[replaced_indx]
        # print(replaced_indx)
        fitness_parents[failed_indx][replaced_indx] = np.asarray(failed_pop_fittness)[replaced_indx]

        # --- post processing --------------------------------+
        # # 1: calculate mean mu_CR and mu_F
        F_archive = np.asarray(F_archive)
        if len(CR_archive) != 0:
           mu_CR = (1. - scale_c) * mu_CR + scale_c * np.mean(CR_archive)
        if len(F_archive) != 0:
            mu_F = (1. - scale_c) * mu_F + scale_c * np.sum(F_archive ** 2) / (np.sum(F_archive))

        # # 2: maintain archive
        if len(archive_A) > popsize:
            indx_keep = np.random.choice(np.arange(0, len(archive_A)), popsize, replace=False)
            archive_A = archive_A[indx_keep]

        # --- SCORE KEEPING --------------------------------+

        gen_avg = sum(gen_scores) / popsize  # current generation avg. fitness
        gen_best = min(gen_scores)  # fitness of best individual
        bset_fitness_all.append(gen_best)
        best_solution_ = population[np.argmin(fitness_parents)]  # solution of best individual
        # print('      > GENERATION AVERAGE:', gen_avg)
        # print('      > GENERATION BEST:', gen_best)
        # print('         > BEST SOLUTION:', gen_sol, '\n')

    # print('\n\n==================BEST SOLUTION==================')
    # print(best_solution_.tolist())
    # print('\n\n==================BEST FITNESS EACH GEN==================')
    # print(bset_fitness_all)
    return best_solution_



def novelty_search(set_A, set_B, n_novelty, n_neighbors):
    """
    :param n_novelty: number of replaced individuals
    :param n_neighbors:
    :return:
    """
    set_A = np.asarray(set_A)
    set_B = np.asarray(set_B)
    if set_A.shape[0] < n_neighbors:
        if set_A.shape[0] == 0:
            set_A = set_B
        set_A = np.append(set_A, set_B, axis=0)
        dis = pairwise_distances(set_B, set_A)
        sorted_indx = np.argsort(dis, axis=1)
        topk_indx = sorted_indx[:, :n_neighbors]
        x_indx = np.tile(np.arange(dis.shape[0]), (1, n_neighbors)).reshape((n_neighbors, dis.shape[0])).transpose()
        ij_indx = (x_indx, topk_indx)
        topk_dis = dis[ij_indx]
        avg_dis = np.mean(topk_dis, axis=1)
        replaced_indx = np.argsort(avg_dis)[-n_novelty:]
        return replaced_indx


# if __name__ == '__main__':
#
#     def f1(x):
#         x = np.asarray(x)
#         f = np.sum(x ** 2)
#         return f  # (np.sum(x**3 + 1))**2
#
#     def f2(x):
#         x = np.abs(np.asarray(x))
#         a = np.sum(x)
#         b = 1
#         for xi in x:
#             b *= xi
#         return a + b
#
#     def f3(x):
#         y = 0
#         for i in range(len(x)):
#             for j in range(i):
#                 y += np.sum(x[:i]) ** 2
#         return y
#
#     def f9(x):
#         x = np.asarray(x)
#         a = x**2 - 10 * np.cos(2 * np.pi * x) + 10
#         return np.sum(a)
#
#     def f4(x):
#         x = np.asarray(x)
#         y = np.max(np.abs(x))
#         return y
#
#     def f6(x):
#         x = np.asarray(x)
#         return np.sum(np.floor(x + 0.5) ** 2)
#
#     bounds = [(-100, 100)] * 30  # bounds [(x1_min, x1_max), (x2_min, x2_max),...]
#     # bounds = [(-5.12, 5.12)] * 30
#     popsize = 100  # population size, must be >= 4
#     maxiter = 200
#     n_neighbor = 15
#     n_novelty = 50
#     minimize(f6, bounds, popsize, maxiter=maxiter, n_novelty_solution=n_novelty, n_neighbors=n_neighbor)



