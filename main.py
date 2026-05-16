import time
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.stats import qmc

# --- GLOBALŪS KINTAMIEJI ---
global_eval_count = 0

# --- KONSTANTOS ---
BITS_PER_VAR = 16
MIN_VAL = -100
MAX_VAL = 100
HC_STEP = 0.05
HC_ITERATIONS = 5
TARGET_FITNESS = 69.998
D_FIXED = 20
POP_SIZE = 5000
MAX_GENS = 500
TRIALS_PER_RATE = 50


# --- PAGALBINĖS FUNKCIJOS ---

def real_to_binary_D(x, min_val=MIN_VAL, max_val=MAX_VAL, bits=BITS_PER_VAR):
    scaled = (x - min_val) / (max_val - min_val)
    int_val = int(scaled * (2 ** bits - 1))
    return bin(int_val)[2:].zfill(bits)


def binary_to_real_D(x, min_val=MIN_VAL, max_val=MAX_VAL, bits=BITS_PER_VAR):
    int_val = int(x, 2)
    scaled = int_val / (2 ** bits - 1)
    return scaled * (max_val - min_val) + min_val


def _get_real_variables(individual_binary, D, bits=BITS_PER_VAR):
    real_variables = []
    for i in range(D):
        start = i * bits
        end = (i + 1) * bits
        var_binary = individual_binary[start:end]
        real_variables.append(binary_to_real_D(var_binary))
    return real_variables

def create_initial_parents_D(population_size, D, min_val=MIN_VAL, max_val=MAX_VAL, bits=BITS_PER_VAR):
    parents = []
    for _ in range(population_size):
        individual_binary = ""
        for _ in range(D):
            val = np.random.uniform(min_val, max_val)
            individual_binary += real_to_binary_D(val, min_val, max_val, bits)
        parents.append(individual_binary)
    return parents


def trysKalniukai_D(points):
    global global_eval_count
    points = np.asarray(points)

    if points.ndim == 1:
        points = points[np.newaxis, :]

    global_eval_count += points.shape[0]

    A = np.array([40.0, 70.0, 34.0])
    s = np.array([2.0, 2.5, 2.0])
    centers = np.array([-70.0, 0.0, 40.0])


    diffs = points[:, :, np.newaxis] - centers
    dist_sq = np.sum(diffs ** 2, axis=1)  # (N, 3)

    gaussians = A * np.exp(-dist_sq / (2 * s ** 2))
    fitness_values = np.sum(gaussians, axis=1)

    return fitness_values if len(fitness_values) > 1 else fitness_values[0]

def fitness_D(individual_binary, D):
    vars = _get_real_variables(individual_binary, D)
    return trysKalniukai_D(vars)


def crossover_two_point_D(parents, population_size):
    children = []
    L = len(parents[0])
    while len(children) < population_size:
        p1, p2 = random.sample(parents, 2)
        cx = sorted(random.sample(range(1, L), 2))
        children.append(p1[:cx[0]] + p2[cx[0]:cx[1]] + p1[cx[1]:])
        if len(children) < population_size:
            children.append(p2[:cx[0]] + p1[cx[0]:cx[1]] + p2[cx[1]:])
    return children[:population_size]


def mutate(parents, mutation_rate):
    return ["".join([('1' if b == '0' else '0') if random.random() < mutation_rate else b for b in p]) for p in parents]


def hill_climbing_binary(individual_binary, D):
    x = np.array(_get_real_variables(individual_binary, D))
    curr_fit = trysKalniukai_D(x.tolist())
    for _ in range(HC_ITERATIONS):
        improved = False
        for i in range(D):
            for step in [HC_STEP, -HC_STEP]:
                xn = x.copy()
                xn[i] = np.clip(xn[i] + step, MIN_VAL, MAX_VAL)
                new_fit = trysKalniukai_D(xn.tolist())
                if new_fit > curr_fit:
                    x, curr_fit = xn, new_fit
                    improved = True
        if not improved: break
    return "".join(real_to_binary_D(val) for val in x)


# --- VIENO BANDYMO FUNKCIJA ---

def single_run(algo_type, mutation_rate):
    parents = ["".join(real_to_binary_D(np.random.uniform(MIN_VAL, MAX_VAL)) for _ in range(D_FIXED)) for _ in
               range(POP_SIZE)]
    best_fitness = -1.0
    best_binary = ""

    start_t = time.time()
    for i in range(1, MAX_GENS + 1):
        fitness_list = [(p, fitness_D(p, D_FIXED)) for p in parents]
        fitness_list.sort(key=lambda x: x[1], reverse=True)
        curr_best_p, curr_best_f = fitness_list[0]

        if curr_best_f > best_fitness:
            best_fitness = curr_best_f
            best_binary = curr_best_p

        if best_fitness >= TARGET_FITNESS:
            return i, time.time() - start_t, 1

        top_parents = [x[0] for x in fitness_list[:int(POP_SIZE * 0.2)]]
        new_parents = crossover_two_point_D(top_parents, POP_SIZE)
        new_parents = mutate(new_parents, mutation_rate)

        if algo_type == "MA":
            for j in range(int(POP_SIZE * 0.05)):
                new_parents[j] = hill_climbing_binary(new_parents[j], D_FIXED)

        new_parents[0] = best_binary
        parents = new_parents

    return MAX_GENS, time.time() - start_t, 0

def genetic_algorithm_D(initial_parents, fitness_function, mutation_rate, population_size, D, generations=5000):
    History = []
    parents = initial_parents.copy()

    L = D * BITS_PER_VAR
    current_mutation_rate = 1 / L

    # current_mutation_rate = mutation_rate

    top_parents_initial, initial_best_fitness, _ = _get_fittest_parents_D(parents, fitness_function, D, top_k=1)
    best_fitness = initial_best_fitness
    best_parent_binary_overall = top_parents_initial[0]

    for i in range(1, generations + 1):
        top_parents, curr_fitness, PFitness = _get_fittest_parents_D(parents, fitness_function, D,
                                                                     top_k=int(population_size * 0.2))
        new_parents = crossover_two_point_D(top_parents, population_size)
        new_parents = mutate(new_parents, current_mutation_rate)

        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_parent_binary_overall = top_parents[0]

        _, _, new_PFitness_sorted = _get_fittest_parents_D(new_parents, fitness_function, D, top_k=population_size)
        worst_new_parent_binary = new_PFitness_sorted[-1][0]

        try:
            worst_index = new_parents.index(worst_new_parent_binary)
            new_parents[worst_index] = best_parent_binary_overall
        except ValueError:
            new_parents[-1] = best_parent_binary_overall

        parents = new_parents
        History.append((i, best_fitness))

        if best_fitness >= 69.998:
            print(f'GA D={D} | Generacija {i} | Geriausias fitnesas: {best_fitness:.5e}')
            break

        if i % 10 == 0:
            print(f'GA D={D} | Generacija {i} | Geriausias fitnesas: {best_fitness:.5e}')
    return best_parent_binary_overall, best_fitness, History, i

def _get_fittest_parents_D(parents, fitness_function, D, top_k=40):
    _fitness_list = []
    for parent_binary in parents:
        _fitness_list.append(fitness_function(parent_binary, D))

    _fitness = np.array(_fitness_list)
    indices = np.argsort(_fitness)[::-1]

    top_parents = []
    for i in indices[:top_k]:
        top_parents.append(parents[i.item()])

    best_fitness = _fitness[indices[0]].item()

    PFitness_sorted = []
    for i in indices:
        PFitness_sorted.append((parents[i.item()], _fitness[i].item()))

    return top_parents, best_fitness, PFitness_sorted


def memetic_genetic_algorithm_D(initial_parents, fitness_function, mutation_rate, population_size, D, generations=5000):
    History = []
    parents = initial_parents.copy()
    LOCAL_SEARCH_RATE = 0.05

    L = D * BITS_PER_VAR
    current_mutation_rate = (1 / L)
    # current_mutation_rate = mutation_rate

    top_parents_initial, initial_best_fitness, _ = _get_fittest_parents_D(parents, fitness_function, D, top_k=1)
    best_fitness = initial_best_fitness
    best_parent_binary_overall = top_parents_initial[0]

    for i in range(1, generations + 1):
        top_parents, _, _ = _get_fittest_parents_D(parents, fitness_function, D, top_k=int(population_size * 0.2))
        new_parents = crossover_two_point_D(top_parents, population_size)
        new_parents = mutate(new_parents, current_mutation_rate)

        top_new_offspring, _, new_PFitness_sorted_full = _get_fittest_parents_D(
            new_parents, fitness_function, D, top_k=population_size)

        offspring_to_optimize = new_PFitness_sorted_full[:int(population_size * LOCAL_SEARCH_RATE)]

        for item in offspring_to_optimize:
            original_binary = item[0]
            original_fitness = item[1]
            optimized_binary = hill_climbing_binary(original_binary, D)
            optimized_fitness = fitness_function(optimized_binary, D)

            if optimized_fitness > original_fitness:
                try:
                    idx = new_parents.index(original_binary)
                    new_parents[idx] = optimized_binary
                except ValueError:
                    pass

        _, curr_fitness, new_PFitness_sorted = _get_fittest_parents_D(new_parents, fitness_function, D,
                                                                      top_k=population_size)

        if curr_fitness > best_fitness:
            best_fitness = curr_fitness
            best_parent_binary_overall = new_PFitness_sorted[0][0]

        if best_parent_binary_overall not in new_parents:
            new_parents[-1] = best_parent_binary_overall

        parents = new_parents
        History.append((i, best_fitness))

        if best_fitness >= 69.998:
            print(f'MA D={D} | Generacija {i} | Geriausias fitnesas: {best_fitness:.5e}')
            break

        if i % 10 == 0:
            print(f'MA D={D} | Generacija {i} | Geriausias fitnesas: {best_fitness:.5e}')

    return best_parent_binary_overall, best_fitness, History, i



def run_mutation_study(algo_type, filename):
    rates = np.arange(0.01, 1.0, 0.01)

    with open(filename, "w", encoding="utf-8") as f:
        # Pridėtas stulpelis 'Sekme'
        f.write("Rate Iteracijos Laikas Sekme\n")

        for r in rates:
            print(f"Analizuojama {algo_type}, Mutacijos Rate: {r:.2f}...")
            total_iters = 0
            total_time = 0
            success_count = 0

            for _ in range(TRIALS_PER_RATE):
                iters, duration, success = single_run(algo_type, r)
                total_iters += iters
                total_time += duration
                success_count += success

            avg_iters = total_iters / TRIALS_PER_RATE
            avg_time = total_time / TRIALS_PER_RATE
            success_rate = success_count / TRIALS_PER_RATE

            f.write(f"{r:.2f} {avg_iters:.2f} {avg_time:.4f} {success_rate:.2f}\n")
            f.flush()




#run_mutation_study("GA", "mutacija_GA_4D.txt")
#run_mutation_study("MA", "mutacija_MA_4D.txt")



#
# def nubraizyti_4_grafikus():
#     # Funkcija duomenų nuskaitymui
#     def gauti_duomenis(failas):
#         try:
#             data = np.loadtxt(failas, skiprows=1)
#             return data[:, 0], data[:, 1], data[:, 3]  # Rate, Iters, Success
#         except:
#             return None, None, None
#
#     # Nuskaitome abu failus
#     r_ga, i_ga, s_ga = gauti_duomenis("mutacija_GA_4D.txt")
#     r_ma, i_ma, s_ma = gauti_duomenis("mutacija_MA_4D.txt")
#
#     if r_ga is None or r_ma is None:
#         print("Klaida: Nepavyko rasti mutacija_GA.txt arba mutacija_MA.txt")
#         return
#
#     # Sukuriame 2x2 langą
#     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
#     fig.suptitle('Mutacijos įtaka algoritmų efektyvumui (D=4)', fontsize=16)
#
#     # --- MEMETINIS ALGORITMAS (MA) ---
#     # MA Iteracijos
#     axs[0, 0].plot(r_ma, i_ma, color='red', label='MA Iteracijos')
#     axs[0, 0].set_title('MA: Vidutinės iteracijos (D=4)')
#     axs[0, 0].set_ylabel('Iteracijos')
#     axs[0, 0].grid(True, alpha=0.3)
#
#     # MA Sėkmė
#     axs[0, 1].plot(r_ma, s_ma, color='darkorange', label='MA Sėkmė')
#     axs[0, 1].set_title('MA: Sėkmės tikimybė (D=4)')
#     axs[0, 1].set_ylabel('Sėkmė (0-1)')
#     axs[0, 1].set_ylim(-0.1, 1.1)
#     axs[0, 1].grid(True, alpha=0.3)
#
#     # --- GENETINIS ALGORITMAS (GA) ---
#     # GA Iteracijos
#     axs[1, 0].plot(r_ga, i_ga, color='blue', label='GA Iteracijos')
#     axs[1, 0].set_title('GA: Vidutinės iteracijos (D=4)')
#     axs[1, 0].set_xlabel('Mutacijos Rate')
#     axs[1, 0].set_ylabel('Iteracijos')
#     axs[1, 0].grid(True, alpha=0.3)
#
#     # GA Sėkmė
#     axs[1, 1].plot(r_ga, s_ga, color='green', label='GA Sėkmė')
#     axs[1, 1].set_title('GA: Sėkmės tikimybė (D=4)')
#     axs[1, 1].set_xlabel('Mutacijos Rate')
#     axs[1, 1].set_ylabel('Sėkmė (0-1)')
#     axs[1, 1].set_ylim(-0.1, 1.1)
#     axs[1, 1].grid(True, alpha=0.3)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#
#     # Išsaugome grafiką kaip paveikslėlį
#     plt.savefig('eksperimento_rezultatai.png', dpi=300)
#     plt.show()


#nubraizyti_4_grafikus()




# def trys_kalniukai_viz(x1, x2):
#     # Funkcijos parametrai
#     A1, A2, A3 = 40.0, 70.0, 34.0
#     s1, s2, s3 = 2, 2.5, 2  # Mažesnis plotis, kad kalnai nesiliestų
#
#     # Išdėstome centrus skirtingose vietose, kad matytųsi 3 atskiri kalnai
#     # c1 = [-5, -5], c2 = [0, 0], c3 = [5, 5]
#     dist1_sq = (x1 + 70) ** 2 + (x2 + 70) ** 2
#     dist2_sq = (x1) ** 2 + (x2) ** 2
#     dist3_sq = (x1 - 40) ** 2 + (x2 -  40) ** 2
#
#     return (A1 * np.exp(-dist1_sq / (2 * s1 ** 2)) +
#             A2 * np.exp(-dist2_sq / (2 * s2 ** 2)) +
#             A3 * np.exp(-dist3_sq / (2 * s3 ** 2)))
#
#
# # 1. Paruošiami duomenys
# x = np.linspace(-100, 100, 100)
# y = np.linspace(-100, 100, 100)
# X, Y = np.meshgrid(x, y)
# Z = trys_kalniukai_viz(X, Y)
#
# # 2. Sukuriamas grafikas
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # Paviršiaus braižymas
# surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                        linewidth=0, antialiased=True, alpha=0.9)
#
# # Pridedame spalvų skalę
# fig.colorbar(surf, shrink=0.5, aspect=10, label='Fitness vertė')
#
# # Pavadinimai ir instrukcija
# ax.set_title('Interaktyvus grafikas: Sukinėkite naudodami pelę')
# ax.set_xlabel('X ašis')
# ax.set_ylabel('Y ašis')
# ax.set_zlabel('Fitness')
#
# # Leidžiame interaktyvumą
# plt.ion()
# plt.show(block=True)
#
# print("\nEksperimentas baigtas. Failai paruošti analizei.")

def create_initial_parents_loop(target_pop_size, D, threshold=1e-300, batch_size=32768, time_limit=60):
    sampler = qmc.Sobol(d=D, scramble=True)
    start_time = time.time()
    found_parents = []

    print(f"--- Vektorizuota Shotgun paieška D={D} ---")

    while len(found_parents) < target_pop_size:

        sample = sampler.random(n=batch_size)
        scaled = qmc.scale(sample, [MIN_VAL] * D, [MAX_VAL] * D)

        all_fitness = trysKalniukai_D(scaled)

        success_indices = np.where(all_fitness > threshold)[0]

        for idx in success_indices:
            individual_real = scaled[idx]
            binary = "".join(real_to_binary_D(val) for val in individual_real)
            found_parents.append(binary)
            if len(found_parents) >= target_pop_size:
                break

        if time.time() - start_time > time_limit:
            print(f"   ! Laiko limitas baigtas. Rasta: {len(found_parents)}")
            needed = target_pop_size - len(found_parents)
            found_parents.extend(create_initial_parents_D(needed, D))
            break

    return found_parents



ma_best_fitness_rez = []
ma_iterations_rez = []
ma_time_rez = []
ma_evals_rez = []

dimensions_to_test = [4, 8, 12, 16, 20, 24]
rezultatu_failas = "eksperimento_rezultatai_D.txt"

with open(rezultatu_failas, "w", encoding="utf-8") as f:
    f.write("Dimensija Tinkamumo_verte Iteracijos Funkciju_kvietimai Vykdymo_laikas_s\n")

    for current_D in dimensions_to_test:
        global_eval_count = 0
        start_time = time.perf_counter()

        print(f"--- Atsitiktinė generacija D={current_D} ---")
        initial_pop = create_initial_parents_D(
            population_size=POP_SIZE,
            D=current_D
        )

        # 1. Standartinis Shotgun (Sobol)
        # initial_pop = create_initial_parents_loop(
        #     target_pop_size=POP_SIZE,
        #     D=current_D,
        #     threshold=1e-320,
        #     batch_size=32768,
        #     time_limit=30
        # )


        best_bin_ma, best_fit_ma, history_ma, iterations_ma  = memetic_genetic_algorithm_D(
            initial_parents=initial_pop,
            fitness_function=fitness_D,
            mutation_rate=0.05,
            population_size=POP_SIZE,
            D=current_D,
            generations=MAX_GENS
        )

        # best_bin_ma, best_fit_ma, history_ma, iterations_ma = genetic_algorithm_D(
        #     initial_parents=initial_pop,
        #     fitness_function=fitness_D,
        #     mutation_rate=0.05,
        #     population_size=POP_SIZE,
        #     D=current_D,
        #     generations=MAX_GENS
        # )

        duration = time.perf_counter() - start_time

        ma_best_fitness_rez.append(best_fit_ma)
        ma_iterations_rez.append(iterations_ma)
        ma_time_rez.append(duration)
        ma_evals_rez.append(global_eval_count)

        f.write(f"{current_D} {best_fit_ma:.5f} {iterations_ma} {global_eval_count} {duration:.4f}\n")
        f.flush()

        print(f"D={current_D} baigta. Fitnesas: {best_fit_ma:.4f}, Laikas: {duration:.2f}s, Funkcijų kvietimai: {global_eval_count}")

print("\n" + "=" * 55)
print(f"DIMENSIJOS:          {dimensions_to_test}")
print(f"FITNESAS:            {[f'{f:.4e}' for f in ma_best_fitness_rez]}")
print(f"ITERACIJOS:          {ma_iterations_rez}")
print(f"FUNKCIJŲ KVIETIMAI:  {ma_evals_rez}")
print(f"LAIKAS (s):          {[round(t, 2) for t in ma_time_rez]}")
print("=" * 55)
print(f"Rezultatai sėkmingai išsaugoti faile: {rezultatu_failas}")

