// genetic_algorithm_cuda.cuh - Algoritmo genético CUDA
#ifndef GA_CUDA_CUH
#define GA_CUDA_CUH

#include "types.hpp"

// ============================================================================
// Variante CUDA Básico
// ============================================================================

GAResult runGA_CUDA_Basic(const Instance& inst, const GAConfig& config);

// ============================================================================
// Variante CUDA Optimizado
// ============================================================================

GAResult runGA_CUDA_Optimized(const Instance& inst, const GAConfig& config);

#endif // GA_CUDA_CUH
