// common.hpp - Utilidades compartidas CPU/GPU
#ifndef COMMON_HPP
#define COMMON_HPP

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================================
// CUDA error checking
// ============================================================================

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUDA_CHECK_KERNEL() do { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA kernel error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ============================================================================
// Timing utilities (CPU)
// ============================================================================

#ifdef _WIN32
#include <windows.h>

struct CPUTimer {
    LARGE_INTEGER start, end, freq;
    
    void startTimer() { QueryPerformanceCounter(&start); }
    void stopTimer() { QueryPerformanceCounter(&end); QueryPerformanceFrequency(&freq); }
    double elapsedMs() { return (double)(end.QuadPart - start.QuadPart) * 1000.0 / freq.QuadPart; }
};

#else
#include <sys/time.h>

struct CPUTimer {
    struct timeval start, end;
    
    void startTimer() { gettimeofday(&start, NULL); }
    void stopTimer() { gettimeofday(&end, NULL); }
    double elapsedMs() { 
        return (end.tv_sec - start.tv_sec) * 1000.0 + 
               (end.tv_usec - start.tv_usec) / 1000.0; 
    }
};

#endif

// ============================================================================
// CUDA Timing events
// ============================================================================

struct CUDATimer {
    cudaEvent_t start, stop;
    
    CUDATimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CUDATimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void startTimer() { CUDA_CHECK(cudaEventRecord(start)); }
    void stopTimer() { CUDA_CHECK(cudaEventRecord(stop)); CUDA_CHECK(cudaEventSynchronize(stop)); }
    
    float elapsedMs() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// ============================================================================
// Default GA config
// ============================================================================

inline GAConfig defaultConfig() {
    GAConfig cfg;
    cfg.population_size = 4096;
    cfg.generations = 500;
    cfg.block_size = 256;
    cfg.seed = 42;
    cfg.crossover_rate = 0.85f;
    cfg.mutation_rate = 0.01f;
    // Penalizaciones (valores justificados: deben ser > max_valor_posible
    // para que la violación siempre sea peor que cualquier ganancia)
    // Con 100 items y valor medio ~50, max valor posible ~5000
    // Penalizaciones deben ser al menos 10x eso por unidad de violación
    cfg.alpha = 100.0f;   // Penalización por unidad de exceso de peso
    cfg.beta = 100.0f;    // Penalización por unidad de exceso de volumen
    cfg.gamma = 50.0f;    // Penalización por violación de categoría
    cfg.delta = 500.0f;   // Penalización por incompatibilidad (fuerte, es dura)
    cfg.epsilon = 500.0f; // Penalización por dependencia incumplida (fuerte, es dura)
    cfg.tournament_size = 5;  // Mayor presión hacia soluciones factibles
    cfg.elitism_count = 4;  // Preservar los 4 mejores individuos
    cfg.run_cpu = true;
    cfg.run_cuda_basic = true;
    cfg.run_cuda_optimized = true;
    return cfg;
}

// ============================================================================
// Utility functions
// ============================================================================

inline void printInstanceSummary(const Instance& inst) {
    printf("  Items: %d\n", inst.n_items);
    printf("  Capacidad peso: %d / %d (%.1f%%)\n", 
           inst.capacidad_peso, inst.suma_total_pesos,
           100.0f * inst.capacidad_peso / inst.suma_total_pesos);
    printf("  Capacidad volumen: %d / %d (%.1f%%)\n", 
           inst.capacidad_volumen, inst.suma_total_volumenes,
           100.0f * inst.capacidad_volumen / inst.suma_total_volumenes);
    printf("  Categorias: %zu reglas\n", inst.category_rules.size());
    printf("  Incompatibilidades: %zu pares\n", inst.incompatibilities.size());
    printf("  Dependencias: %zu\n", inst.dependencies.size());
}

inline void printGAResult(const GAResult& result, const char* variant_name) {
    printf("=== %s ===\n", variant_name);
    printf("  Tiempo total: %.2f ms\n", result.time_total_ms);
    printf("  Tiempo fitness: %.2f ms\n", result.time_fitness_ms);
    printf("  Mejor valor factible: %d\n", result.best_value);
    printf("  Mejor fitness: %d\n", result.best_fitness);
    printf("  Soluciones factibles: %.1f%%\n", result.feasible_percentage * 100.0f);
    printf("  Tiempo kernel CUDA: %.2f ms\n", result.time_kernel_total_ms);
    printf("  Transfer H→D: %.2f ms\n", result.time_transfer_h2d_ms);
    printf("  Transfer D→H: %.2f ms\n", result.time_transfer_d2h_ms);
}

#endif // COMMON_HPP
