// fitness_cuda.cuh - Kernels CUDA para evaluación de fitness
#ifndef FITNESS_CUDA_CUH
#define FITNESS_CUDA_CUH

#include "types.hpp"

// ============================================================================
// Kernel: Evaluar fitness de toda la población
// 1 hilo por individuo
// ============================================================================

// Versión básica: sin optimizaciones de memoria
void evaluateFitness_CUDA_Basic(
    int* d_fitness,         // [out] Fitness de cada individuo
    int* d_valor_total,     // [out] Valor total de cada individuo  
    bool* d_factible,       // [out] ¿Es factible?
    const int* d_genes,     // [in] Población aplanada (pop_size * n_items)
    const int* d_item_valores,
    const int* d_item_pesos,
    const int* d_item_volumenes,
    const int* d_item_categorias,
    int pop_size,
    int n_items,
    int capacidad_peso,
    int capacidad_volumen,
    // Penalizaciones
    const int* d_cat_minimos,   // [in] Mínimos por categoría
    const int* d_cat_maximos,   // [in] Máximos por categoría
    const int* d_cat_categorias, // [in] IDs de categoría
    int n_cat_rules,
    // Incompatibilidades
    const int* d_incomp_a,
    const int* d_incomp_b,
    int n_incompat,
    // Dependencias
    const int* d_dep_item,
    const int* d_dep_requerido,
    int n_deps,
    // Pesos de penalización
    float alpha, float beta, float gamma, float delta, float epsilon
);

// Versión optimizada: shared memory, reducciones
void evaluateFitness_CUDA_Optimized(
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
);

#endif // FITNESS_CUDA_CUH
