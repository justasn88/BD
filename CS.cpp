#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <fstream>

using namespace std;

// --- KONSTANTOS ---
constexpr double MIN_VAL = -100.0;
constexpr double MAX_VAL = 100.0;
constexpr double TARGET_FITNESS = 69.0;
constexpr int D_FIXED = 2;
constexpr int MAX_GENS = 50000;
constexpr int RUNS_PER_EXPERIMENT = 500;
constexpr double FIXED_PA = 0.8;

mt19937_64 rng;

inline double clip(double val, double min_v, double max_v) {
    return (val < min_v) ? min_v : ((val > max_v) ? max_v : val);
}

// Tikslo funkcija
double trysKalniukai_D(const vector<double>& point) {
    const double A[3] = {40.0, 70.0, 34.0};
    const double s[3] = {2.0, 2.5, 2.0};
    const double centers[3] = {-70.0, 80.0, 50.0};
    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        double dist_sq = 0.0;
        for (int d = 0; d < D_FIXED; ++d) {
            double diff = point[d] - centers[i];
            dist_sq += diff * diff;
        }
        sum += A[i] * exp(-dist_sq / (2.0 * s[i] * s[i]));
    }
    return sum;
}

void levy_flight_inplace(vector<double>& step, double Lambda, int D) {
    double pi = acos(-1.0);
    double num = tgamma(1.0 + Lambda) * sin(pi * Lambda / 2.0);
    double den = tgamma((1.0 + Lambda) / 2.0) * Lambda * pow(2.0, (Lambda - 1.0) / 2.0);
    double sigma = pow(num / den, 1.0 / Lambda);
    normal_distribution<double> norm_u(0.0, sigma);
    normal_distribution<double> norm_v(0.0, 1.0);
    for (int i = 0; i < D; ++i) {
        step[i] = norm_u(rng) / pow(abs(norm_v(rng)), 1.0 / Lambda);
    }
}

struct RunResult {
    double best_fitness;
    int iterations;
    long long evaluations;
    bool success;
};

RunResult run_cs(double current_pa, int current_pop_size) {
    vector<vector<double>> nest(current_pop_size, vector<double>(D_FIXED));
    vector<double> fitness(current_pop_size);
    vector<double> best_nest(D_FIXED);
    double best_fit = -1.0;
    long long eval_count = 0;
    uniform_real_distribution<double> unif_dist(MIN_VAL, MAX_VAL);
    uniform_real_distribution<double> prob_dist(0.0, 1.0);
    uniform_int_distribution<int> pop_dist(0, current_pop_size - 1);

    // Pradinė populiacija kūrimas
    for (int i = 0; i < current_pop_size; ++i) {
        for (int d = 0; d < D_FIXED; ++d) nest[i][d] = unif_dist(rng);
        fitness[i] = trysKalniukai_D(nest[i]);
        eval_count++;
        if (fitness[i] > best_fit) { best_fit = fitness[i]; best_nest = nest[i]; }
    }

    double alpha_0 = 0.01;
    vector<double> step(D_FIXED);
    vector<double> new_nest_temp(D_FIXED);

    int gen;
    for (gen = 1; gen <= MAX_GENS; ++gen) {
        // 1. Lévy skrydžiai
        for (int i = 0; i < current_pop_size; ++i) {
            levy_flight_inplace(step, 1.5, D_FIXED);
            for (int d = 0; d < D_FIXED; ++d) {
                new_nest_temp[d] = clip(nest[i][d] + alpha_0 * step[d] * (nest[i][d] - best_nest[d]), MIN_VAL, MAX_VAL);
            }
            double new_fit = trysKalniukai_D(new_nest_temp);
            eval_count++;

            int j = pop_dist(rng);
            if (new_fit > fitness[j]) { nest[j] = new_nest_temp; fitness[j] = new_fit; }
        }

        // 2. Svetimų kiaušinių aptikimas (PA)
        for (int k = 0; k < current_pop_size; ++k) {
            if (prob_dist(rng) < current_pa) {
                int r1 = pop_dist(rng), r2 = pop_dist(rng);
                double s_size = prob_dist(rng);
                for (int d = 0; d < D_FIXED; ++d) {
                    nest[k][d] = clip(nest[k][d] + s_size * (nest[r1][d] - nest[r2][d]), MIN_VAL, MAX_VAL);
                }
                fitness[k] = trysKalniukai_D(nest[k]);
                eval_count++;
            }
        }

        for (int i = 0; i < current_pop_size; ++i) {
            if (fitness[i] > best_fit) { best_fit = fitness[i]; best_nest = nest[i]; }
        }

        if (best_fit >= TARGET_FITNESS) break;
    }

    return {best_fit, (gen > MAX_GENS ? MAX_GENS : gen), eval_count, best_fit >= TARGET_FITNESS};
}

int main() {
    rng.seed(chrono::high_resolution_clock::now().time_since_epoch().count());

    ofstream dataFile("pop_size_eksperimentas2D.csv");
    // Pridėtas stulpelis CSV faile
    dataFile << "PopSize,VidutinisFitness,SekmesTikimybe,VidutinesIteracijos,VidutiniaiIvertinimai" << endl;

    cout << "Pradedamas Populiacijos dydžio (POP_SIZE) eksperimentas..." << endl;
    cout << "Fiksuotas PA: " << FIXED_PA << " | Dimensijos: " << D_FIXED << endl;
    cout << "----------------------------------------------------------------------" << endl;

    for (int current_pop = 10; current_pop <= 100; current_pop += 10) {
        double total_fitness = 0;
        long long total_iterations = 0;
        long long total_evaluations = 0;
        int successes = 0;

        auto start_pop = chrono::high_resolution_clock::now();

        for (int run = 0; run < RUNS_PER_EXPERIMENT; ++run) {
            RunResult res = run_cs(FIXED_PA, current_pop);
            total_fitness += res.best_fitness;
            total_iterations += res.iterations;
            total_evaluations += res.evaluations;
            if (res.success) successes++;
        }

        double avg_fitness = total_fitness / RUNS_PER_EXPERIMENT;
        double avg_iterations = (double)total_iterations / RUNS_PER_EXPERIMENT;
        double avg_evals = (double)total_evaluations / RUNS_PER_EXPERIMENT;
        double success_rate = (double)successes / RUNS_PER_EXPERIMENT;

        dataFile << current_pop << ","
                 << fixed << setprecision(8) << avg_fitness << ","
                 << setprecision(4) << success_rate << ","
                 << setprecision(2) << avg_iterations << ","
                 << setprecision(0) << avg_evals << endl;

        auto end_pop = chrono::high_resolution_clock::now();
        chrono::duration<double> duration = end_pop - start_pop;

        cout << "PopSize: " << setw(3) << current_pop
             << " | Sekme: " << setw(3) << setprecision(0) << success_rate * 100 << "%"
             << " | Vid. Iter: " << setw(6) << (int)avg_iterations
             << " | Vid. Ivert: " << setw(8) << (long long)avg_evals
             << " | Laikas: " << setprecision(2) << duration.count() << "s" << endl;
    }

    dataFile.close();


    return 0;
}