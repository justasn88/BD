#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>

using namespace std;

// --- KONSTANTOS ---
constexpr double MIN_VAL = -100.0;
constexpr double MAX_VAL = 100.0;
constexpr double TARGET_FITNESS = 69.998;   // Tikslas, kurį pasiekus algoritmas sustabdomas
constexpr int D_FIXED = 12;                 // Paieškos erdvės dimensijų skaičius
constexpr int POP_SIZE = 64;                // CS populiacijos dydis
constexpr int MAX_GENS = 1000000;           // Maksimalus bendras generacijų skaičius
constexpr double PA = 0.85;                 // Lizdų apleidimo tikimybė

// Slankstis: kai fitnesas viršija šią reikšmę, laikoma, kad rastas "šlaitas" ir pereinama į GA
constexpr double ABSOLUTE_GA_THRESHOLD = 1e-300;

// --- GA KONSTANTOS ---
constexpr int GA_POP_SIZE = 25;             // GA populiacijos dydis
constexpr double GA_RADIUS = 5.0;           // Spindulys aplink geriausią CS tašką, kuriame generuojama pradinė GA populiacija
constexpr double GA_MUTATION_STEP = 0.5;    // Maksimalus mutacijos žingsnio dydis
constexpr int BITS_PER_VAR = 16;            // Naudojama skaičiuojant dinaminę mutacijos tikimybę

unsigned long long function_calls = 0;      // Globalus kintamasis, skaičiuojantis tikslo funkcijos iškvietimus

mt19937_64 rng;                             // Atsitiktinių skaičių generatorius


inline double clip(double val, double min_v, double max_v) {
    if (val < min_v) return min_v;
    if (val > max_v) return max_v;
    return val;
}

// Tikslo funkcija (3 Gauso komponentų suma)
double trysKalniukai_D(const vector<double>& point) {
    function_calls++; // Fiksuojamas funkcijos iškvietimas

    const double A[3] = {40.0, 70.0, 34.0};        // Kalnų aukščiai (amplitudės)
    const double s[3] = {2.0, 2.5, 2.0};           // Kalnų sklaidos (pločiai)
    const double centers[3] = {-70.0, 80.0, 50.0}; // Kalnų centrų koordinatės

    double dist_sq[3] = {0.0, 0.0, 0.0};

    for (int d = 0; d < D_FIXED; ++d) {
        for (int i = 0; i < 3; ++i) {
            double diff = point[d] - centers[i];
            dist_sq[i] += diff * diff;
        }
    }

    double sum = 0.0;
    for (int i = 0; i < 3; ++i) {
        sum += A[i] * exp(-dist_sq[i] / (2.0 * s[i] * s[i]));
    }
    return sum;
}

// Sugeneruoja žingsnį pagal Lėvi skirstinį (Mantegna metodas)
void levy_flight_inplace(vector<double>& step, double Lambda, int D) {
    double pi = acos(-1.0);
    double num = tgamma(1.0 + Lambda) * sin(pi * Lambda / 2.0);
    double den = tgamma((1.0 + Lambda) / 2.0) * Lambda * pow(2.0, (Lambda - 1.0) / 2.0);
    double sigma = pow(num / den, 1.0 / Lambda);

    normal_distribution<double> norm_u(0.0, sigma);
    normal_distribution<double> norm_v(0.0, 1.0);

    for (int i = 0; i < D; ++i) {
        double u = norm_u(rng);
        double v = norm_v(rng);
        step[i] = u / pow(abs(v), 1.0 / Lambda);
    }
}

int main() {
    rng.seed(chrono::high_resolution_clock::now().time_since_epoch().count());
    auto start_t = chrono::high_resolution_clock::now(); // Pradedamas laiko matavimas

    // Apskaičiuojama mutacijos tikimybė GA etapui
    double current_mutation_rate = 1.0 / (D_FIXED * BITS_PER_VAR);

    cout << "Pradedamas Cuckoo Search..." << endl;

    // --- CS ALGORITMO INICIALIZACIJA ---
    vector<vector<double>> nest(POP_SIZE, vector<double>(D_FIXED));
    vector<double> fitness(POP_SIZE);
    vector<double> step(D_FIXED);
    vector<double> new_nest_temp(D_FIXED);

    uniform_real_distribution<double> unif_dist(MIN_VAL, MAX_VAL);
    uniform_real_distribution<double> prob_dist(0.0, 1.0);

    double best_fit = -1.0;
    vector<double> best_nest(D_FIXED);

    // Kuriama pradinė populiacija ir įvertinamas jos fitnesas
    for (int i = 0; i < POP_SIZE; ++i) {
        for (int d = 0; d < D_FIXED; ++d) {
            nest[i][d] = unif_dist(rng);
        }
        fitness[i] = trysKalniukai_D(nest[i]);
        if (fitness[i] > best_fit) {
            best_fit = fitness[i];
            best_nest = nest[i];
        }
    }

    double alpha_0 = 0.9; // Pradinis žingsnio daugiklis
    int gen = 0;
    bool ga_triggered = false;
    uniform_int_distribution<int> pop_dist(0, POP_SIZE - 1);

    // Gegutės paieška (CS)
    for (gen = 1; gen <= MAX_GENS; ++gen) {

        // Jei aptiktas "šlaitas" (pajudėta iš plokštumos), pereiname į GA etapą
        if (best_fit > ABSOLUTE_GA_THRESHOLD) {
            cout << "\n--- Perjungiama i GA. (Slenkstis: " << scientific << ABSOLUTE_GA_THRESHOLD << ") ---\n";
            ga_triggered = true;
            break;
        }

        double alpha;
        if (best_fit < 1e-300) {
            alpha = alpha_0;
        } else {
            alpha = alpha_0 * (1.0 - (best_fit / TARGET_FITNESS));
        }

        // Lėvi skrydžiai (Globali paieška) ---
        for (int i = 0; i < POP_SIZE; ++i) {
            levy_flight_inplace(step, 1.5, D_FIXED);
            for (int d = 0; d < D_FIXED; ++d) {
                new_nest_temp[d] = nest[i][d] + alpha * step[d] * (nest[i][d] - best_nest[d]);
                new_nest_temp[d] = clip(new_nest_temp[d], MIN_VAL, MAX_VAL);
            }

            double new_fit = trysKalniukai_D(new_nest_temp);
            int j = pop_dist(rng);

            if (new_fit > fitness[j]) {
                nest[j] = new_nest_temp;
                fitness[j] = new_fit;
                if (new_fit > best_fit) {
                    best_fit = new_fit;
                    best_nest = new_nest_temp;
                }
            }
        }

        if (best_fit > ABSOLUTE_GA_THRESHOLD) {
            cout << "\n--- SLAITAS RASTAS po Levy suolio! Perjungiama i GA. ---\n";
            ga_triggered = true;
            break;
        }

        // Svetimų kiaušinių aptikimas
        for (int k = 0; k < POP_SIZE; ++k) {
            if (prob_dist(rng) < PA) {
                int r1 = pop_dist(rng);
                int r2 = pop_dist(rng);
                while (r1 == r2) r2 = pop_dist(rng);

                double step_size = prob_dist(rng);
                for (int d = 0; d < D_FIXED; ++d) {
                    nest[k][d] = nest[k][d] + step_size * (nest[r1][d] - nest[r2][d]);
                    nest[k][d] = clip(nest[k][d], MIN_VAL, MAX_VAL);
                }
                fitness[k] = trysKalniukai_D(nest[k]);
                if (fitness[k] > best_fit) {
                    best_fit = fitness[k];
                    best_nest = nest[k];
                }
            }
        }

        if (gen % 10000 == 0) {
            cout << "CS | Gen " << gen << " | Fit: " << scientific << best_fit << " | Kvietimai: " << function_calls << endl;
        }
    }

    // GA ALGORITMO INICIALIZACIJA

    if (ga_triggered || best_fit > TARGET_FITNESS) {
        vector<vector<double>> ga_pop(GA_POP_SIZE, vector<double>(D_FIXED));
        vector<double> ga_fitness(GA_POP_SIZE);
        normal_distribution<double> radius_dist(0.0, GA_RADIUS);


        // Generuojama nauja populiacija aplink geriausią CS rastą tašką
        for (int i = 0; i < GA_POP_SIZE; ++i) {
            for (int d = 0; d < D_FIXED; ++d) {
                ga_pop[i][d] = clip(best_nest[d] + radius_dist(rng), MIN_VAL, MAX_VAL);
            }
            ga_fitness[i] = trysKalniukai_D(ga_pop[i]);
        }

        ga_pop[0] = best_nest;
        ga_fitness[0] = trysKalniukai_D(ga_pop[0]);

        uniform_int_distribution<int> ga_pop_dist(0, GA_POP_SIZE - 1);
        normal_distribution<double> mutation_dist(0.0, GA_MUTATION_STEP);

        // Pagrindinis GA ciklas
        while (gen < MAX_GENS) {
            gen++;
            vector<vector<double>> new_pop(GA_POP_SIZE, vector<double>(D_FIXED));

            int best_ga_idx = 0;
            for(int i = 1; i < GA_POP_SIZE; ++i){
                if(ga_fitness[i] > ga_fitness[best_ga_idx]) best_ga_idx = i;
            }

            best_fit = ga_fitness[best_ga_idx];
            new_pop[0] = ga_pop[best_ga_idx];

            if (best_fit >= TARGET_FITNESS) break;

            // Kryžminimas ir mutacija visiems kitiems populiacijos nariams
            for (int i = 1; i < GA_POP_SIZE; ++i) {
                int p1 = ga_pop_dist(rng);
                int p2 = ga_pop_dist(rng);

                for (int d = 0; d < D_FIXED; ++d) {
                    double weight = prob_dist(rng);
                    double child_gene = ga_pop[p1][d] * weight + ga_pop[p2][d] * (1.0 - weight);

                    // Mutacija
                    if (prob_dist(rng) < current_mutation_rate) child_gene += mutation_dist(rng);

                    new_pop[i][d] = clip(child_gene, MIN_VAL, MAX_VAL);
                }
            }

            ga_pop = new_pop;
            for(int i = 0; i < GA_POP_SIZE; ++i) {
                ga_fitness[i] = trysKalniukai_D(ga_pop[i]);
            }
            if (gen % 500 == 0) {
                cout << fixed << setprecision(5) << "GA | Gen " << gen << " | Fit: " << best_fit << " | Kvietimai: " << function_calls << endl;
            }
        }
    }

    auto end_t = chrono::high_resolution_clock::now();
    chrono::duration<double> t_diff = end_t - start_t; // Skaičiuojamas vykdymo laikas

    cout << "\n------------------------------------------" << endl;
    cout << "GALUTINIAI REZULTATAI:" << endl;
    cout << "Geriausias fitnesas: " << fixed << setprecision(6) << best_fit << endl;
    cout << "Funkcijos kvietimu skaicius: " << function_calls << endl;
    cout << "Skaiciavimo laikas: " << t_diff.count() << " s" << endl;
    cout << "Efektyvumas (kvietimai/sek): " << (long long)(function_calls / t_diff.count()) << endl;

    return 0;
}