// fitness_cuda.cu - Kernels CUDA para evaluación de fitness
#include "fitness_cuda.cuh"
#include "common.hpp"

// ============================================================================
// Kernel básico: 1 hilo por individuo, acceso directo a memoria global
// ============================================================================

__global__ void kernelFitnessBasic(
    int* d_fitness,
    int* d_valor_total,
    bool* d_factible,
    const int* d_genes,
    const int* d_item_valores,
    const int* d_item_pesos,
    const int* d_item_volumenes,
    const int* d_item_categorias,
    int pop_size,
    int n_items,
    int capacidad_peso,
    int capacidad_volumen,
    const int* d_cat_minimos,
    const int* d_cat_maximos,
    const int* d_cat_categorias,
    int n_cat_rules,
    const int* d_incomp_a,
    const int* d_incomp_b,
    int n_incompat,
    const int* d_dep_item,
    const int* d_dep_requerido,
    int n_deps,
    float alpha, float beta, float gamma, float delta, float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    // Calcular valor total, peso y volumen
    int valor_total = 0;
    int peso_total = 0;
    int volumen_total = 0;
    
    const int* my_genes = d_genes + idx * n_items;
    
    for (int i = 0; i < n_items; i++) {
        if (my_genes[i]) {
            valor_total += d_item_valores[i];
            peso_total += d_item_pesos[i];
            volumen_total += d_item_volumenes[i];
        }
    }
    
    // Penalizaciones
    float penalizacion = 0.0f;
    
    // Peso
    if (peso_total > capacidad_peso) {
        penalizacion += alpha * (peso_total - capacidad_peso);
    }
    
    // Volumen
    if (volumen_total > capacidad_volumen) {
        penalizacion += beta * (volumen_total - capacidad_volumen);
    }
    
    // Categorías
    for (int r = 0; r < n_cat_rules; r++) {
        int cat = d_cat_categorias[r];
        int count = 0;
        for (int i = 0; i < n_items; i++) {
            if (my_genes[i] && d_item_categorias[i] == cat) {
                count++;
            }
        }
        if (count < d_cat_minimos[r]) {
            penalizacion += gamma * (d_cat_minimos[r] - count);
        } else if (count > d_cat_maximos[r]) {
            penalizacion += gamma * (count - d_cat_maximos[r]);
        }
    }
    
    // Incompatibilidades
    for (int i = 0; i < n_incompat; i++) {
        if (my_genes[d_incomp_a[i]] && my_genes[d_incomp_b[i]]) {
            penalizacion += delta;
        }
    }
    
    // Dependencias
    for (int i = 0; i < n_deps; i++) {
        if (my_genes[d_dep_item[i]] && !my_genes[d_dep_requerido[i]]) {
            penalizacion += epsilon;
        }
    }
    
    d_fitness[idx] = valor_total - (int)penalizacion;
    d_valor_total[idx] = valor_total;
    
    // Verificar factibilidad (solo restricciones duras)
    bool peso_ok = peso_total <= capacidad_peso;
    bool volumen_ok = volumen_total <= capacidad_volumen;
    
    bool incompat_ok = true;
    for (int i = 0; i < n_incompat && incompat_ok; i++) {
        if (my_genes[d_incomp_a[i]] && my_genes[d_incomp_b[i]]) {
            incompat_ok = false;
        }
    }
    
    bool deps_ok = true;
    for (int i = 0; i < n_deps && deps_ok; i++) {
        if (my_genes[d_dep_item[i]] && !my_genes[d_dep_requerido[i]]) {
            deps_ok = false;
        }
    }
    
    d_factible[idx] = peso_ok && volumen_ok && incompat_ok && deps_ok;
}

// ============================================================================
// Kernel optimizado: shared memory para datos de items, reducción por bloque
// ============================================================================

#define BLOCK_SIZE_OPT 256

__global__ void kernelFitnessOptimized(
    int* d_fitness,
    int* d_valor_total,
    bool* d_factible,
    const int* d_genes,
    const int* d_item_valores,
    const int* d_item_pesos,
    const int* d_item_volumenes,
    const int* d_item_categorias,
    int pop_size,
    int n_items,
    int capacidad_peso,
    int capacidad_volumen,
    const int* d_cat_minimos,
    const int* d_cat_maximos,
    const int* d_cat_categorias,
    int n_cat_rules,
    const int* d_incomp_a,
    const int* d_incomp_b,
    int n_incompat,
    const int* d_dep_item,
    const int* d_dep_requerido,
    int n_deps,
    float alpha, float beta, float gamma, float delta, float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    // Shared memory para datos de items (reutilización dentro del bloque)
    extern __shared__ int shared_mem[];
    // Layout: [valores(n_items)] [pesos(n_items)] [volumenes(n_items)] [categorias(n_items)]
    
    int* s_valores = shared_mem;
    int* s_pesos = shared_mem + n_items;
    int* s_volumenes = shared_mem + 2 * n_items;
    int* s_categorias = shared_mem + 3 * n_items;
    
    // Cargar datos de items a shared memory (coalesced load)
    for (int i = threadIdx.x; i < n_items; i += blockDim.x) {
        s_valores[i] = d_item_valores[i];
        s_pesos[i] = d_item_pesos[i];
        s_volumenes[i] = d_item_volumenes[i];
        s_categorias[i] = d_item_categorias[i];
    }
    __syncthreads();
    
    // Calcular fitness usando shared memory
    int valor_total = 0;
    int peso_total = 0;
    int volumen_total = 0;
    
    const int* my_genes = d_genes + idx * n_items;
    
    for (int i = 0; i < n_items; i++) {
        if (my_genes[i]) {
            valor_total += s_valores[i];
            peso_total += s_pesos[i];
            volumen_total += s_volumenes[i];
        }
    }
    
    // Penalizaciones
    float penalizacion = 0.0f;
    
    if (peso_total > capacidad_peso) {
        penalizacion += alpha * (peso_total - capacidad_peso);
    }
    
    if (volumen_total > capacidad_volumen) {
        penalizacion += beta * (volumen_total - capacidad_volumen);
    }
    
    // Categorías (acceso a shared memory)
    for (int r = 0; r < n_cat_rules; r++) {
        int cat = d_cat_categorias[r];
        int count = 0;
        for (int i = 0; i < n_items; i++) {
            if (my_genes[i] && s_categorias[i] == cat) {
                count++;
            }
        }
        if (count < d_cat_minimos[r]) {
            penalizacion += gamma * (d_cat_minimos[r] - count);
        } else if (count > d_cat_maximos[r]) {
            penalizacion += gamma * (count - d_cat_maximos[r]);
        }
    }
    
    // Incompatibilidades
    for (int i = 0; i < n_incompat; i++) {
        if (my_genes[d_incomp_a[i]] && my_genes[d_incomp_b[i]]) {
            penalizacion += delta;
        }
    }
    
    // Dependencias
    for (int i = 0; i < n_deps; i++) {
        if (my_genes[d_dep_item[i]] && !my_genes[d_dep_requerido[i]]) {
            penalizacion += epsilon;
        }
    }
    
    d_fitness[idx] = valor_total - (int)penalizacion;
    d_valor_total[idx] = valor_total;
    
    // Verificar factibilidad
    bool peso_ok = peso_total <= capacidad_peso;
    bool volumen_ok = volumen_total <= capacidad_volumen;
    
    bool incompat_ok = true;
    for (int i = 0; i < n_incompat && incompat_ok; i++) {
        if (my_genes[d_incomp_a[i]] && my_genes[d_incomp_b[i]]) {
            incompat_ok = false;
        }
    }
    
    bool deps_ok = true;
    for (int i = 0; i < n_deps && deps_ok; i++) {
        if (my_genes[d_dep_item[i]] && !my_genes[d_dep_requerido[i]]) {
            deps_ok = false;
        }
    }
    
    d_factible[idx] = peso_ok && volumen_ok && incompat_ok && deps_ok;
}

// ============================================================================
// Wrapper functions
// ============================================================================

void evaluateFitness_CUDA_Basic(
    int* d_fitness, int* d_valor_total, bool* d_factible,
    const int* d_genes,
    const int* d_item_valores, const int* d_item_pesos,
    const int* d_item_volumenes, const int* d_item_categorias,
    int pop_size, int n_items,
    int capacidad_peso, int capacidad_volumen,
    const int* d_cat_minimos, const int* d_cat_maximos,
    const int* d_cat_categorias, int n_cat_rules,
    const int* d_incomp_a, const int* d_incomp_b, int n_incompat,
    const int* d_dep_item, const int* d_dep_requerido, int n_deps,
    float alpha, float beta, float gamma, float delta, float epsilon
) {
    int blockSize = 256;
    int gridSize = (pop_size + blockSize - 1) / blockSize;
    
    kernelFitnessBasic<<<gridSize, blockSize>>>(
        d_fitness, d_valor_total, d_factible, d_genes,
        d_item_valores, d_item_pesos, d_item_volumenes, d_item_categorias,
        pop_size, n_items, capacidad_peso, capacidad_volumen,
        d_cat_minimos, d_cat_maximos, d_cat_categorias, n_cat_rules,
        d_incomp_a, d_incomp_b, n_incompat,
        d_dep_item, d_dep_requerido, n_deps,
        alpha, beta, gamma, delta, epsilon
    );
    CUDA_CHECK_KERNEL();
}

void evaluateFitness_CUDA_Optimized(
    int* d_fitness, int* d_valor_total, bool* d_factible,
    const int* d_genes,
    const int* d_item_valores, const int* d_item_pesos,
    const int* d_item_volumenes, const int* d_item_categorias,
    int pop_size, int n_items,
    int capacidad_peso, int capacidad_volumen,
    const int* d_cat_minimos, const int* d_cat_maximos,
    const int* d_cat_categorias, int n_cat_rules,
    const int* d_incomp_a, const int* d_incomp_b, int n_incompat,
    const int* d_dep_item, const int* d_dep_requerido, int n_deps,
    float alpha, float beta, float gamma, float delta, float epsilon
) {
    int blockSize = BLOCK_SIZE_OPT;
    int gridSize = (pop_size + blockSize - 1) / blockSize;
    int sharedMemSize = 4 * n_items * sizeof(int); // 4 arrays de int por item
    
    kernelFitnessOptimized<<<gridSize, blockSize, sharedMemSize>>>(
        d_fitness, d_valor_total, d_factible, d_genes,
        d_item_valores, d_item_pesos, d_item_volumenes, d_item_categorias,
        pop_size, n_items, capacidad_peso, capacidad_volumen,
        d_cat_minimos, d_cat_maximos, d_cat_categorias, n_cat_rules,
        d_incomp_a, d_incomp_b, n_incompat,
        d_dep_item, d_dep_requerido, n_deps,
        alpha, beta, gamma, delta, epsilon
    );
    CUDA_CHECK_KERNEL();
}
