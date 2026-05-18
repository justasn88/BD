#include <iostream>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <curand_kernel.h>             
#include <thrust/device_ptr.h>        
#include <thrust/extrema.h>           
#include <thrust/execution_policy.h>

using namespace std;

// Konstantos
constexpr double MIN_VAL = -100.0;          
constexpr double MAX_VAL = 100.0;           
constexpr double TARGET_FITNESS = 69.998;           // Riba, kurią pasiekus skaičiavimai stabdomi
constexpr int D_FIXED = 12;                          // Optimizuojamų dimensijų skaičius
constexpr int POP_SIZE = 262144;                    // populiacija
constexpr int THREADS_PER_BLOCK = 256;              // Kiek gijų vykdys vienas CUDA blokas
constexpr int MAX_GENS = 500000000;                 // Maksimalus generacijų skaičius
constexpr double PA = 0.85;                         // Lizdų apleidimo tikimybė
constexpr double GA_MUTATION_STEP = 0.5;            // mutacijos tikimybė
constexpr double GA_RADIUS = 5.0;                   // Spindulys sukuriant pradinę GA populiaciją
constexpr double ABSOLUTE_GA_THRESHOLD = 1e-300;    //Riba su kuria perjungiama iš CS į GA

__constant__ double d_A_dbl[3] = {40.0, 70.0, 34.0};        // Kalnų aukščiai
__constant__ double d_s_dbl[3] = {2.0, 2.5, 2.0};           // Kalnų pločiai (sklaida)
__constant__ double d_centers_dbl[3] = {-70.0, 80.0, 50.0}; // Kalnų viršūnių koordinatės

__device__ unsigned long long d_eval_counter = 0;


// Inicializuoja atsitiktinių skaičių generatorių kiekvienai gijai
__global__ void setup_curand_kernel(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < POP_SIZE) curand_init(seed, id, 0, &state[id]);
}

// Tikslo (fitneso) funkcija.
__device__ double calculate_fitness(double* nest, int id) {
    atomicAdd(&d_eval_counter, 1ULL);

    double dist_sq_dbl[3] = {0.0, 0.0, 0.0};
    
    for(int d = 0; d < D_FIXED; ++d) {
        double val = nest[d * POP_SIZE + id];
        for(int i = 0; i < 3; ++i) {
            double diff = val - d_centers_dbl[i];
            dist_sq_dbl[i] += diff * diff;
        }
    }
    
    double sum_dbl = 0.0;
    for (int i = 0; i < 3; ++i) {
        sum_dbl += d_A_dbl[i] * exp(-dist_sq_dbl[i] / (2.0 * d_s_dbl[i] * d_s_dbl[i]));
    }
    return sum_dbl;
}

// Pradinės populiacijos sukūrimas 
__global__ void init_pop_kernel(double* nest, double* fitness, curandState *state) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < POP_SIZE) {
        curandState localState = state[id];
        for (int d = 0; d < D_FIXED; ++d) {
            nest[d * POP_SIZE + id] = MIN_VAL + curand_uniform_double(&localState) * (MAX_VAL - MIN_VAL);
        }
        fitness[id] = calculate_fitness(nest, id);
        state[id] = localState;
    }
}

// Gegutės paieškos algoritmas
__global__ void cs_fused_kernel(double* nest, double* fitness, int best_idx, curandState *state, double alpha, double* trial_nest) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= POP_SIZE) return;

    curandState localState = state[id];
    double sigma = 0.6965745;
    double current_fit = fitness[id];
    bool data_updated = false;

    // Lėvi skrydžiai
    for (int d = 0; d < D_FIXED; ++d) {
        double u = curand_normal_double(&localState) * sigma;
        double v = curand_normal_double(&localState);
        
        double v_abs = fmax(fabs(v), 1e-8); 
        double step = u / cbrt(v_abs * v_abs);
        
        double curr_gene = nest[d * POP_SIZE + id];
        double best_gene = nest[d * POP_SIZE + best_idx];
        
        double new_gene = curr_gene + alpha * step * (curr_gene - best_gene);
        
        if (new_gene < MIN_VAL) new_gene = MIN_VAL;
        if (new_gene > MAX_VAL) new_gene = MAX_VAL;
        trial_nest[d * POP_SIZE + id] = new_gene; 
    }
    
    double new_fit = calculate_fitness(trial_nest, id);
    if (new_fit > current_fit) {
        current_fit = new_fit;
        data_updated = true;
        for (int d = 0; d < D_FIXED; ++d) nest[d * POP_SIZE + id] = trial_nest[d * POP_SIZE + id];
    }

    // vetimų kiaušinių aptikimas (Lizdų apleidimas)
    if (curand_uniform_double(&localState) < PA) {
        int r1 = curand(&localState) % POP_SIZE;
        int r2 = curand(&localState) % POP_SIZE;
        double step_size = curand_uniform_double(&localState);

        for (int d = 0; d < D_FIXED; ++d) {
            double gene = nest[d * POP_SIZE + id] + step_size * (nest[d * POP_SIZE + r1] - nest[d * POP_SIZE + r2]);
            if (gene < MIN_VAL) gene = MIN_VAL;
            if (gene > MAX_VAL) gene = MAX_VAL;
            trial_nest[d * POP_SIZE + id] = gene;
        }
        
        current_fit = calculate_fitness(trial_nest, id);
        data_updated = true;
        for (int d = 0; d < D_FIXED; ++d) nest[d * POP_SIZE + id] = trial_nest[d * POP_SIZE + id];
    }

    if (data_updated) {
        fitness[id] = current_fit;
    }
    state[id] = localState;
}

// Perjungimas iš CS į GA.
__global__ void teleport_kernel(double* nest, double* fitness, int best_idx, curandState *state, double radius) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= POP_SIZE || id == best_idx) return; 

    curandState localState = state[id];

    for (int d = 0; d < D_FIXED; ++d) {
        double best_gene = nest[d * POP_SIZE + best_idx];
        double new_gene = best_gene + curand_normal_double(&localState) * radius;

        if (new_gene < MIN_VAL) new_gene = MIN_VAL;
        if (new_gene > MAX_VAL) new_gene = MAX_VAL;
        nest[d * POP_SIZE + id] = new_gene;
    }

    fitness[id] = calculate_fitness(nest, id);
    state[id] = localState;
}

// 6. Genetinis algoritmas
__global__ void ga_global_kernel(double* nest, double* fitness, double* trial_nest, int best_idx, curandState *state, double mut_rate) {
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= POP_SIZE || id == best_idx) return;

    curandState localState = state[id];

    int p1 = curand(&localState) % POP_SIZE;
    for(int t = 0; t < 2; ++t) {
        int cand = curand(&localState) % POP_SIZE;
        if(fitness[cand] > fitness[p1]) p1 = cand;
    }

    int p2 = curand(&localState) % POP_SIZE;
    for(int t = 0; t < 2; ++t) {
        int cand = curand(&localState) % POP_SIZE;
        if(fitness[cand] > fitness[p2]) p2 = cand;
    }

    // Kryžminimas ir Mutacija
    for (int d = 0; d < D_FIXED; ++d) {
        double weight = curand_uniform_double(&localState);
        double child_gene = nest[d * POP_SIZE + p1] * weight + nest[d * POP_SIZE + p2] * (1.0 - weight);
        
        if (curand_uniform_double(&localState) < mut_rate) {
            child_gene += curand_normal_double(&localState) * GA_MUTATION_STEP;
        }

        if (child_gene < MIN_VAL) child_gene = MIN_VAL;
        if (child_gene > MAX_VAL) child_gene = MAX_VAL;
        trial_nest[d * POP_SIZE + id] = child_gene;
    }

    fitness[id] = calculate_fitness(trial_nest, id);
    for (int d = 0; d < D_FIXED; ++d) nest[d * POP_SIZE + id] = trial_nest[d * POP_SIZE + id];

    state[id] = localState;
}

int main() {
    auto start_t = chrono::high_resolution_clock::now();

    // Rodyklės į GPU (Device) atmintį
    double *d_nest, *d_fitness, *d_trial_nest;
    curandState *d_state;
    
    // Alokuojama atmintis vaizdo plokštėje
    cudaMalloc(&d_nest, POP_SIZE * D_FIXED * sizeof(double));
    cudaMalloc(&d_trial_nest, POP_SIZE * D_FIXED * sizeof(double));
    cudaMalloc(&d_fitness, POP_SIZE * sizeof(double));
    cudaMalloc(&d_state, POP_SIZE * sizeof(curandState));

    int blocks = (POP_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    setup_curand_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_state, 1337ULL);
    
    init_pop_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_nest, d_fitness, d_state);
    
    thrust::device_ptr<double> d_ptr_fit(d_fitness);

    double initial_best = 0.0;
    auto max_iter_init = thrust::max_element(thrust::device, d_ptr_fit, d_ptr_fit + POP_SIZE);
    cudaMemcpy(&initial_best, d_fitness + (max_iter_init - d_ptr_fit), sizeof(double), cudaMemcpyDeviceToHost);
    
    if (initial_best == 0.0) initial_best = 1e-323; 

    cout << "Pradedamas GPU skaiciavimai\n";
    cout << "Pradinis atskaitos taskas: " << scientific << initial_best << endl;
    cout << "Slenkstis perejimui i GA: " << ABSOLUTE_GA_THRESHOLD << endl;
    cout << "--------------------------------------------------\n";

    double h_global_best = initial_best;
    bool ga_mode = false;
    double current_mutation_rate = 1.0 / (D_FIXED * 16.0); 
    double alpha_0 = 0.9; 
    double current_alpha = alpha_0; 

    for (int gen = 1; gen <= MAX_GENS; ++gen) {
        
        auto max_iter = thrust::max_element(thrust::device, d_ptr_fit, d_ptr_fit + POP_SIZE);
        int best_idx = max_iter - d_ptr_fit;
        
        if (!ga_mode) {
            // CS veikimas
            if (h_global_best < 1e-300) {
                current_alpha = alpha_0;
            } else {
                current_alpha = alpha_0 * (1.0 - (h_global_best / TARGET_FITNESS));
                if (current_alpha < 0.0) current_alpha = 0.0;
            }

            cs_fused_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_nest, d_fitness, best_idx, d_state, current_alpha, d_trial_nest);
        } else {
            // GA veikimas
            ga_global_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_nest, d_fitness, d_trial_nest, best_idx, d_state, current_mutation_rate);
        }
        
        cudaMemcpy(&h_global_best, &d_fitness[best_idx], sizeof(double), cudaMemcpyDeviceToHost);
        
        // Algoritmų perjungimas
        if (!ga_mode && h_global_best > ABSOLUTE_GA_THRESHOLD) {
            ga_mode = true;
            cout << "\n Perjungiama i GA. (Slenkstis: " << scientific << ABSOLUTE_GA_THRESHOLD << ") ---\n";
            teleport_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_nest, d_fitness, best_idx, d_state, GA_RADIUS);
        }

        if (h_global_best >= TARGET_FITNESS) {
            cout << "PASIEKTA! Generacija: " << gen << endl;
            break;
        }

        if (gen % 1000 == 0) {
            unsigned long long h_total_evals = 0;
            cudaMemcpyFromSymbol(&h_total_evals, d_eval_counter, sizeof(unsigned long long));

            cout << "Generacija: " << gen 
                 << " | Geriausias fitnesas: " << scientific << h_global_best 
                 << " | FE (Iskvietimai): " << h_total_evals
                 << " | Rezimas: " << (ga_mode ? "GA" : "CS (Alpha: " + to_string(current_alpha) + ")") << endl;
        }
    }

    auto end_t = chrono::high_resolution_clock::now();
    
    unsigned long long final_evals = 0;
    cudaMemcpyFromSymbol(&final_evals, d_eval_counter, sizeof(unsigned long long));

    cout << "\nGALUTINIS REZULTATAS: Hibridinis CS+GA D=" << D_FIXED 
         << " | Fitnesas: " << fixed << setprecision(5) << h_global_best 
         << "\nViso fitneso funkcijos iskvietimu (FE): " << final_evals
         << "\nLaikas (GPU): " << chrono::duration<double>(end_t - start_t).count() << "s\n";

    cudaFree(d_nest); cudaFree(d_trial_nest);
    cudaFree(d_fitness); cudaFree(d_state);

    return 0;
}