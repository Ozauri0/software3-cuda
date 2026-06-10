// genetic_algorithm_cpu.cpp - Implementación del AG secuencial en CPU
#include "genetic_algorithm_cpu.hpp"
#include "common.hpp"
#include <algorithm>
#include <numeric>
#include <cstring>

// ============================================================================
// RNG simple (LCG) - same as CUDA's curand for reproducibility
// ============================================================================

static inline unsigned int nextRandom(unsigned int& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

static inline float nextRandomFloat(unsigned int& state) {
    return (float)(nextRandom(state) & 0x7FFFFFFF) / (float)0x7FFFFFFF;
}

// ============================================================================
// Generación de población inicial
// ============================================================================

void initializePopulation(std::vector<Chromosome>& population, const Instance& inst, unsigned int seed) {
    unsigned int rng = seed;
    int n = inst.n_items;
    
    for (auto& chrom : population) {
        chrom.genes.resize(n);
        chrom.fitness = 0;
        chrom.valor_total = 0;
        chrom.factible = false;
        
        for (int i = 0; i < n; i++) {
            chrom.genes[i] = (nextRandom(rng) % 2);
        }
    }
}

// ============================================================================
// Evaluación de fitness
// ============================================================================

int evaluateFitness(Chromosome& chrom, const Instance& inst, const GAConfig& config) {
    int n = inst.n_items;
    int valor_total = 0;
    int peso_total = 0;
    int volumen_total = 0;
    
    // Calcular totales
    for (int i = 0; i < n; i++) {
        if (chrom.genes[i]) {
            valor_total += inst.items[i].valor;
            peso_total += inst.items[i].peso;
            volumen_total += inst.items[i].volumen;
        }
    }
    
    // Calcular penalizaciones
    float penalizacion = 0.0f;
    
    // Exceso de peso (restricción dura)
    if (peso_total > inst.capacidad_peso) {
        penalizacion += config.alpha * (peso_total - inst.capacidad_peso);
    }
    
    // Exceso de volumen (restricción dura)
    if (volumen_total > inst.capacidad_volumen) {
        penalizacion += config.beta * (volumen_total - inst.capacidad_volumen);
    }
    
    // Violaciones de categoría (restricción blanda)
    for (const auto& rule : inst.category_rules) {
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (chrom.genes[i] && inst.items[i].categoria == rule.categoria) {
                count++;
            }
        }
        if (count < rule.minimo) {
            penalizacion += config.gamma * (rule.minimo - count);
        } else if (count > rule.maximo) {
            penalizacion += config.gamma * (count - rule.maximo);
        }
    }
    
    // Incompatibilidades (restricción dura)
    for (const auto& incomp : inst.incompatibilities) {
        if (chrom.genes[incomp.id_item_a] && chrom.genes[incomp.id_item_b]) {
            penalizacion += config.delta;
        }
    }
    
    // Dependencias incumplidas (restricción dura)
    for (const auto& dep : inst.dependencies) {
        if (chrom.genes[dep.id_item] && !chrom.genes[dep.id_requerido]) {
            penalizacion += config.epsilon;
        }
    }
    
    int fitness = valor_total - (int)penalizacion;
    
    // Actualizar campos del cromosoma
    chrom.valor_total = valor_total;
    chrom.fitness = fitness;
    
    // Verificar factibilidad (restricciones duras solamente)
    bool peso_ok = peso_total <= inst.capacidad_peso;
    bool volumen_ok = volumen_total <= inst.capacidad_volumen;
    bool incompat_ok = true;
    for (const auto& incomp : inst.incompatibilities) {
        if (chrom.genes[incomp.id_item_a] && chrom.genes[incomp.id_item_b]) {
            incompat_ok = false;
            break;
        }
    }
    bool deps_ok = true;
    for (const auto& dep : inst.dependencies) {
        if (chrom.genes[dep.id_item] && !chrom.genes[dep.id_requerido]) {
            deps_ok = false;
            break;
        }
    }
    
    chrom.factible = peso_ok && volumen_ok && incompat_ok && deps_ok;
    
    return fitness;
}

// ============================================================================
// Verificación de factibilidad
// ============================================================================

bool isFeasible(const Chromosome& chrom, const Instance& inst) {
    int peso_total = 0;
    int volumen_total = 0;
    int n = inst.n_items;
    
    for (int i = 0; i < n; i++) {
        if (chrom.genes[i]) {
            peso_total += inst.items[i].peso;
            volumen_total += inst.items[i].volumen;
        }
    }
    
    if (peso_total > inst.capacidad_peso) return false;
    if (volumen_total > inst.capacidad_volumen) return false;
    
    for (const auto& incomp : inst.incompatibilities) {
        if (chrom.genes[incomp.id_item_a] && chrom.genes[incomp.id_item_b]) return false;
    }
    
    for (const auto& dep : inst.dependencies) {
        if (chrom.genes[dep.id_item] && !chrom.genes[dep.id_requerido]) return false;
    }
    
    return true;
}

// ============================================================================
// Selección por torneo
// ============================================================================

int tournamentSelect(const std::vector<Chromosome>& population, int tournament_size, unsigned int& rng_state) {
    int best_idx = nextRandom(rng_state) % population.size();
    int best_fitness = population[best_idx].fitness;
    
    for (int i = 1; i < tournament_size; i++) {
        int idx = nextRandom(rng_state) % population.size();
        if (population[idx].fitness > best_fitness) {
            best_fitness = population[idx].fitness;
            best_idx = idx;
        }
    }
    
    return best_idx;
}

// ============================================================================
// Cruzamiento de un punto
// ============================================================================

void crossover(Chromosome& parent1, Chromosome& parent2, 
               Chromosome& child1, Chromosome& child2,
               float crossover_rate, unsigned int& rng_state) {
    int n = parent1.genes.size();
    
    child1.genes.resize(n);
    child2.genes.resize(n);
    
    if (nextRandomFloat(rng_state) < crossover_rate) {
        int point = nextRandom(rng_state) % (n - 1) + 1;
        
        for (int i = 0; i < n; i++) {
            if (i < point) {
                child1.genes[i] = parent1.genes[i];
                child2.genes[i] = parent2.genes[i];
            } else {
                child1.genes[i] = parent2.genes[i];
                child2.genes[i] = parent1.genes[i];
            }
        }
    } else {
        // Sin cruzamiento, copiar padres
        child1.genes = parent1.genes;
        child2.genes = parent2.genes;
    }
}

// ============================================================================
// Mutación por flip de bits
// ============================================================================

void mutate(Chromosome& chrom, float mutation_rate, unsigned int& rng_state) {
    for (int i = 0; i < (int)chrom.genes.size(); i++) {
        if (nextRandomFloat(rng_state) < mutation_rate) {
            chrom.genes[i] = 1 - chrom.genes[i]; // Flip
        }
    }
}

// ============================================================================
// Comparador para elitismo
// ============================================================================

static bool compareFitness(const Chromosome& a, const Chromosome& b) {
    return a.fitness > b.fitness;
}

// ============================================================================
// AG Secuencial CPU
// ============================================================================

GAResult runGA_CPU(const Instance& inst, const GAConfig& config) {
    GAResult result = {};
    CPUTimer timer;
    
    int pop_size = config.population_size;
    int n = inst.n_items;
    int elitism = config.elitism_count;
    
    timer.startTimer();
    
    // ---- Inicializar población ----
    std::vector<Chromosome> population(pop_size);
    initializePopulation(population, inst, config.seed);
    
    // ---- Evaluar fitness inicial ----
    for (auto& chrom : population) {
        evaluateFitness(chrom, inst, config);
    }
    
    // ---- Evolución ----
    unsigned int rng_state = config.seed + 12345; // Different seed for operators
    
    for (int gen = 0; gen < config.generations; gen++) {
        // Crear nueva población
        std::vector<Chromosome> new_population(pop_size);
        
        // Elitismo: copiar los mejores
        std::vector<Chromosome> sorted_pop = population;
        std::sort(sorted_pop.begin(), sorted_pop.end(), compareFitness);
        
        for (int i = 0; i < elitism && i < pop_size; i++) {
            new_population[i] = sorted_pop[i];
            new_population[i].fitness = evaluateFitness(new_population[i], inst, config);
        }
        
        // Generar resto de la población
        for (int i = elitism; i < pop_size; i += 2) {
            // Selección por torneo
            int p1 = tournamentSelect(population, config.tournament_size, rng_state);
            int p2 = tournamentSelect(population, config.tournament_size, rng_state);
            
            Chromosome child1, child2;
            crossover(population[p1], population[p2], child1, child2, 
                     config.crossover_rate, rng_state);
            
            mutate(child1, config.mutation_rate, rng_state);
            mutate(child2, config.mutation_rate, rng_state);
            
            child1.fitness = evaluateFitness(child1, inst, config);
            child2.fitness = evaluateFitness(child2, inst, config);
            
            new_population[i] = child1;
            if (i + 1 < pop_size) {
                new_population[i + 1] = child2;
            }
        }
        
        population = new_population;
    }
    
    // ---- Encontrar mejor solución ----
    int best_fitness = INT_MIN;
    int best_value = 0;
    int best_idx = -1;
    int feasible_count = 0;
    
    for (int i = 0; i < pop_size; i++) {
        if (population[i].factible && population[i].valor_total > best_value) {
            best_value = population[i].valor_total;
            best_fitness = population[i].fitness;
            best_idx = i;
        }
        if (population[i].factible) feasible_count++;
    }
    
    // Si no hay factible, tomar el de mayor fitness
    if (best_idx == -1) {
        for (int i = 0; i < pop_size; i++) {
            if (population[i].fitness > best_fitness) {
                best_fitness = population[i].fitness;
                best_value = population[i].valor_total;
                best_idx = i;
            }
        }
    }
    
    timer.stopTimer();
    
    // ---- Llenar resultado ----
    result.time_total_ms = timer.elapsedMs();
    result.time_fitness_ms = result.time_total_ms * 0.6; // Estimación
    result.time_kernel_total_ms = 0; // CPU, no hay kernels
    result.time_transfer_h2d_ms = 0;
    result.time_transfer_d2h_ms = 0;
    result.best_value = best_value;
    result.best_fitness = best_fitness;
    result.feasible_percentage = (float)feasible_count / pop_size;
    result.best_solution_index = best_idx;
    
    if (best_idx >= 0) {
        result.best_chromosome = population[best_idx].genes;
    }
    
    return result;
}
