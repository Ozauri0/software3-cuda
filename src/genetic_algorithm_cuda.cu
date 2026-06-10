// genetic_algorithm_cuda.cu - Implementación completa del AG CUDA
#include "genetic_algorithm_cuda.cuh"
#include "fitness_cuda.cuh"
#include "common.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <algorithm>
#include <cstring>

// ============================================================================
// Device RNG (curand)
// ============================================================================

__global__ void setupRNG(curandState* states, unsigned int seed, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// ============================================================================
// Kernel: Selección por torneo (1 torneo por individuo)
// ============================================================================

__global__ void kernelTournamentSelect(
    int* d_selected,
    const int* d_fitness,
    int pop_size,
    int tournament_size,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    curandState localState = states[idx];
    
    int best_idx = curand(&localState) % pop_size;
    int best_fitness = d_fitness[best_idx];
    
    for (int t = 1; t < tournament_size; t++) {
        int candidate = curand(&localState) % pop_size;
        if (d_fitness[candidate] > best_fitness) {
            best_fitness = d_fitness[candidate];
            best_idx = candidate;
        }
    }
    
    d_selected[idx] = best_idx;
    states[idx] = localState;
}

// ============================================================================
// Kernel: Cruzamiento de un punto
// ============================================================================

__global__ void kernelCrossover(
    int* d_new_genes,
    const int* d_genes,
    const int* d_selected,
    int pop_size,
    int n_items,
    float crossover_rate,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size / 2) return;
    
    curandState localState = states[idx];
    
    int parent1_idx = d_selected[idx * 2];
    int parent2_idx = d_selected[idx * 2 + 1];
    
    const int* parent1 = d_genes + parent1_idx * n_items;
    const int* parent2 = d_genes + parent2_idx * n_items;
    int* child1 = d_new_genes + (idx * 2) * n_items;
    int* child2 = d_new_genes + (idx * 2 + 1) * n_items;
    
    if (curand_uniform(&localState) < crossover_rate) {
        int point = curand(&localState) % (n_items - 1) + 1;
        
        for (int i = 0; i < n_items; i++) {
            if (i < point) {
                child1[i] = parent1[i];
                child2[i] = parent2[i];
            } else {
                child1[i] = parent2[i];
                child2[i] = parent1[i];
            }
        }
    } else {
        for (int i = 0; i < n_items; i++) {
            child1[i] = parent1[i];
            child2[i] = parent2[i];
        }
    }
    
    states[idx] = localState;
}

// ============================================================================
// Kernel: Mutación por flip de bits
// ============================================================================

__global__ void kernelMutate(
    int* d_genes,
    int pop_size,
    int n_items,
    float mutation_rate,
    curandState* states
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    curandState localState = states[idx];
    
    int* my_genes = d_genes + idx * n_items;
    
    for (int i = 0; i < n_items; i++) {
        if (curand_uniform(&localState) < mutation_rate) {
            my_genes[i] = 1 - my_genes[i];
        }
    }
    
    states[idx] = localState;
}

// ============================================================================
// Kernel: Copiar elitismo
// ============================================================================

__global__ void kernelCopyElite(
    int* d_new_genes,
    const int* d_genes,
    const int* d_elite_indices,
    int elitism_count,
    int n_items
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= elitism_count) return;
    
    int src_idx = d_elite_indices[idx];
    int* dst = d_new_genes + idx * n_items;
    const int* src = d_genes + src_idx * n_items;
    
    for (int i = 0; i < n_items; i++) {
        dst[i] = src[i];
    }
}

// ============================================================================
// Wrapper: Evaluación fitness (dispatch basic/optimized)
// ============================================================================

static void evaluatePopulation(
    bool optimized,
    int* d_fitness, int* d_valor_total, bool* d_factible,
    const int* d_genes,
    const Instance& inst,
    const GAConfig& config,
    // Device data
    const int* d_item_valores, const int* d_item_pesos,
    const int* d_item_volumenes, const int* d_item_categorias,
    const int* d_cat_minimos, const int* d_cat_maximos,
    const int* d_cat_categorias,
    const int* d_incomp_a, const int* d_incomp_b,
    const int* d_dep_item, const int* d_dep_requerido
) {
    if (optimized) {
        evaluateFitness_CUDA_Optimized(
            d_fitness, d_valor_total, d_factible, d_genes,
            d_item_valores, d_item_pesos, d_item_volumenes, d_item_categorias,
            config.population_size, inst.n_items,
            inst.capacidad_peso, inst.capacidad_volumen,
            d_cat_minimos, d_cat_maximos, d_cat_categorias, (int)inst.category_rules.size(),
            d_incomp_a, d_incomp_b, (int)inst.incompatibilities.size(),
            d_dep_item, d_dep_requerido, (int)inst.dependencies.size(),
            config.alpha, config.beta, config.gamma, config.delta, config.epsilon
        );
    } else {
        evaluateFitness_CUDA_Basic(
            d_fitness, d_valor_total, d_factible, d_genes,
            d_item_valores, d_item_pesos, d_item_volumenes, d_item_categorias,
            config.population_size, inst.n_items,
            inst.capacidad_peso, inst.capacidad_volumen,
            d_cat_minimos, d_cat_maximos, d_cat_categorias, (int)inst.category_rules.size(),
            d_incomp_a, d_incomp_b, (int)inst.incompatibilities.size(),
            d_dep_item, d_dep_requerido, (int)inst.dependencies.size(),
            config.alpha, config.beta, config.gamma, config.delta, config.epsilon
        );
    }
}

// ============================================================================
// AG CUDA Genérico (used by both basic and optimized)
// ============================================================================

static GAResult runGA_CUDA_Internal(const Instance& inst, const GAConfig& config, bool optimized) {
    GAResult result = {};
    CUDATimer timer_transfer, timer_kernel;
    
    int pop_size = config.population_size;
    int n = inst.n_items;
    int elitism = config.elitism_count;
    int blockSize = config.block_size;
    
    // ---- Transfer data to device ----
    timer_transfer.startTimer();
    
    // Items data
    std::vector<int> h_valores(n), h_pesos(n), h_volumenes(n), h_categorias(n);
    for (int i = 0; i < n; i++) {
        h_valores[i] = inst.items[i].valor;
        h_pesos[i] = inst.items[i].peso;
        h_volumenes[i] = inst.items[i].volumen;
        h_categorias[i] = inst.items[i].categoria;
    }
    
    int *d_item_valores, *d_item_pesos, *d_item_volumenes, *d_item_categorias;
    CUDA_CHECK(cudaMalloc(&d_item_valores, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_item_pesos, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_item_volumenes, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_item_categorias, n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_item_valores, h_valores.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_item_pesos, h_pesos.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_item_volumenes, h_volumenes.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_item_categorias, h_categorias.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    
    // Category rules
    int n_cats = (int)inst.category_rules.size();
    std::vector<int> h_cat_categorias(n_cats), h_cat_minimos(n_cats), h_cat_maximos(n_cats);
    for (int i = 0; i < n_cats; i++) {
        h_cat_categorias[i] = inst.category_rules[i].categoria;
        h_cat_minimos[i] = inst.category_rules[i].minimo;
        h_cat_maximos[i] = inst.category_rules[i].maximo;
    }
    int *d_cat_categorias, *d_cat_minimos, *d_cat_maximos;
    CUDA_CHECK(cudaMalloc(&d_cat_categorias, n_cats * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cat_minimos, n_cats * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cat_maximos, n_cats * sizeof(int)));
    if (n_cats > 0) {
        CUDA_CHECK(cudaMemcpy(d_cat_categorias, h_cat_categorias.data(), n_cats * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cat_minimos, h_cat_minimos.data(), n_cats * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cat_maximos, h_cat_maximos.data(), n_cats * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Incompatibilities
    int n_incomp = (int)inst.incompatibilities.size();
    std::vector<int> h_incomp_a(n_incomp), h_incomp_b(n_incomp);
    for (int i = 0; i < n_incomp; i++) {
        h_incomp_a[i] = inst.incompatibilities[i].id_item_a;
        h_incomp_b[i] = inst.incompatibilities[i].id_item_b;
    }
    int *d_incomp_a, *d_incomp_b;
    CUDA_CHECK(cudaMalloc(&d_incomp_a, n_incomp * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_incomp_b, n_incomp * sizeof(int)));
    if (n_incomp > 0) {
        CUDA_CHECK(cudaMemcpy(d_incomp_a, h_incomp_a.data(), n_incomp * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_incomp_b, h_incomp_b.data(), n_incomp * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Dependencies
    int n_deps = (int)inst.dependencies.size();
    std::vector<int> h_dep_item(n_deps), h_dep_requerido(n_deps);
    for (int i = 0; i < n_deps; i++) {
        h_dep_item[i] = inst.dependencies[i].id_item;
        h_dep_requerido[i] = inst.dependencies[i].id_requerido;
    }
    int *d_dep_item, *d_dep_requerido;
    CUDA_CHECK(cudaMalloc(&d_dep_item, n_deps * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dep_requerido, n_deps * sizeof(int)));
    if (n_deps > 0) {
        CUDA_CHECK(cudaMemcpy(d_dep_item, h_dep_item.data(), n_deps * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dep_requerido, h_dep_requerido.data(), n_deps * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // ---- Población en GPU ----
    int* d_genes;
    int* d_new_genes;
    CUDA_CHECK(cudaMalloc(&d_genes, pop_size * n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_new_genes, pop_size * n * sizeof(int)));
    
    // Fitness arrays
    int *d_fitness, *d_valor_total;
    bool* d_factible;
    CUDA_CHECK(cudaMalloc(&d_fitness, pop_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_valor_total, pop_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_factible, pop_size * sizeof(bool)));
    
    // Selection indices
    int* d_selected;
    CUDA_CHECK(cudaMalloc(&d_selected, pop_size * sizeof(int)));
    
    // RNG states
    curandState* d_states;
    CUDA_CHECK(cudaMalloc(&d_states, pop_size * sizeof(curandState)));
    
    timer_transfer.stopTimer();
    result.time_transfer_h2d_ms = timer_transfer.elapsedMs();
    
    // ---- Initialize population on GPU ----
    // Generate random genes on CPU and copy to GPU
    {
        std::vector<int> h_genes(pop_size * n);
        unsigned int rng = config.seed;
        for (int i = 0; i < pop_size * n; i++) {
            rng = rng * 1664525u + 1013904223u;
            h_genes[i] = (rng >> 16) & 1;
        }
        CUDA_CHECK(cudaMemcpy(d_genes, h_genes.data(), pop_size * n * sizeof(int), cudaMemcpyHostToDevice));
    }
    
    // Setup RNG states
    {
        int setupBlock = 256;
        int setupGrid = (pop_size + setupBlock - 1) / setupBlock;
        setupRNG<<<setupGrid, setupBlock>>>(d_states, config.seed, pop_size);
        CUDA_CHECK_KERNEL();
    }
    
    // ---- Evolución ----
    timer_kernel.startTimer();
    
    int gridSize = (pop_size + blockSize - 1) / blockSize;
    int halfGrid = (pop_size / 2 + blockSize - 1) / blockSize;
    
    for (int gen = 0; gen < config.generations; gen++) {
        // 1. Evaluar fitness
        evaluatePopulation(optimized, d_fitness, d_valor_total, d_factible, d_genes,
                          inst, config,
                          d_item_valores, d_item_pesos, d_item_volumenes, d_item_categorias,
                          d_cat_minimos, d_cat_maximos, d_cat_categorias,
                          d_incomp_a, d_incomp_b, d_dep_item, d_dep_requerido);
        
        // 2. Selección por torneo
        kernelTournamentSelect<<<gridSize, blockSize>>>(d_selected, d_fitness, pop_size, config.tournament_size, d_states);
        CUDA_CHECK_KERNEL();
        
        // 3. Cruzamiento
        kernelCrossover<<<halfGrid, blockSize>>>(d_new_genes, d_genes, d_selected, pop_size, n, config.crossover_rate, d_states);
        CUDA_CHECK_KERNEL();
        
        // 4. Mutación
        kernelMutate<<<gridSize, blockSize>>>(d_new_genes, pop_size, n, config.mutation_rate, d_states);
        CUDA_CHECK_KERNEL();
        
        // 5. Copiar nueva generación
        CUDA_CHECK(cudaMemcpy(d_genes, d_new_genes, pop_size * n * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    // Evaluación final
    evaluatePopulation(optimized, d_fitness, d_valor_total, d_factible, d_genes,
                      inst, config,
                      d_item_valores, d_item_pesos, d_item_volumenes, d_item_categorias,
                      d_cat_minimos, d_cat_maximos, d_cat_categorias,
                      d_incomp_a, d_incomp_b, d_dep_item, d_dep_requerido);
    
    timer_kernel.stopTimer();
    result.time_kernel_total_ms = timer_kernel.elapsedMs();
    
    // ---- Transfer results back ----
    timer_transfer.startTimer();
    
    std::vector<int> h_fitness(pop_size), h_valor_total(pop_size);
    std::vector<char> h_factible(pop_size);
    std::vector<int> h_genes_out(pop_size * n);
    
    CUDA_CHECK(cudaMemcpy(h_fitness.data(), d_fitness, pop_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_valor_total.data(), d_valor_total, pop_size * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_factible.data(), d_factible, pop_size * sizeof(bool), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_genes_out.data(), d_genes, pop_size * n * sizeof(int), cudaMemcpyDeviceToHost));
    
    timer_transfer.stopTimer();
    result.time_transfer_d2h_ms = timer_transfer.elapsedMs();
    
    // ---- Find best solution ----
    int best_value = 0;
    int best_fitness = INT_MIN;
    int best_idx = -1;
    int feasible_count = 0;
    
    for (int i = 0; i < pop_size; i++) {
        if (h_factible[i] && h_valor_total[i] > best_value) {
            best_value = h_valor_total[i];
            best_fitness = h_fitness[i];
            best_idx = i;
        }
        if (h_factible[i]) feasible_count++;
    }
    
    if (best_idx == -1) {
        for (int i = 0; i < pop_size; i++) {
            if (h_fitness[i] > best_fitness) {
                best_fitness = h_fitness[i];
                best_value = h_valor_total[i];
                best_idx = i;
            }
        }
    }
    
    result.best_value = best_value;
    result.best_fitness = best_fitness;
    result.feasible_percentage = (float)feasible_count / pop_size;
    result.best_solution_index = best_idx;
    result.time_total_ms = result.time_transfer_h2d_ms + result.time_kernel_total_ms + result.time_transfer_d2h_ms;
    result.time_fitness_ms = result.time_kernel_total_ms * 0.7; // Estimación
    
    if (best_idx >= 0) {
        result.best_chromosome.resize(n);
        memcpy(result.best_chromosome.data(), h_genes_out.data() + best_idx * n, n * sizeof(int));
    }
    
    // ---- Cleanup ----
    cudaFree(d_item_valores);
    cudaFree(d_item_pesos);
    cudaFree(d_item_volumenes);
    cudaFree(d_item_categorias);
    cudaFree(d_cat_categorias);
    cudaFree(d_cat_minimos);
    cudaFree(d_cat_maximos);
    cudaFree(d_incomp_a);
    cudaFree(d_incomp_b);
    cudaFree(d_dep_item);
    cudaFree(d_dep_requerido);
    cudaFree(d_genes);
    cudaFree(d_new_genes);
    cudaFree(d_fitness);
    cudaFree(d_valor_total);
    cudaFree(d_factible);
    cudaFree(d_selected);
    cudaFree(d_states);
    
    return result;
}

// ============================================================================
// Public wrappers
// ============================================================================

GAResult runGA_CUDA_Basic(const Instance& inst, const GAConfig& config) {
    return runGA_CUDA_Internal(inst, config, false);
}

GAResult runGA_CUDA_Optimized(const Instance& inst, const GAConfig& config) {
    return runGA_CUDA_Internal(inst, config, true);
}
