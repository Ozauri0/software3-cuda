// main.cu - Punto de entrada: CLI, carga datos, ejecuta variantes, genera CSV
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "types.hpp"
#include "common.hpp"
#include "instance_loader.hpp"
#include "genetic_algorithm_cpu.hpp"
#include "genetic_algorithm_cuda.cuh"

// ============================================================================
// Print usage
// ============================================================================

void printUsage(const char* prog) {
    printf("Uso: %s [opciones]\n\n", prog);
    printf("Opciones:\n");
    printf("  --instance <small|medium|large>  Instancia a usar (default: small)\n");
    printf("  --variant <cpu|cuda|cuda_opt|all> Variante a ejecutar (default: all)\n");
    printf("  --population <N>                 Tamano de poblacion (default: 4096)\n");
    printf("  --generations <N>                Numero de generaciones (default: 500)\n");
    printf("  --block <N>                      Tamano de bloque CUDA (default: 256)\n");
    printf("  --seed <N>                       Semilla RNG (default: 42)\n");
    printf("  --help                           Mostrar esta ayuda\n\n");
    printf("Ejemplo:\n");
    printf("  %s --instance small --variant all --population 4096 --generations 500 --block 256 --seed 42\n", prog);
}

// ============================================================================
// Parse CLI arguments
// ============================================================================

struct CLIArgs {
    std::string instance = "small";
    std::string variant = "all";
    int population = 4096;
    int generations = 500;
    int block_size = 256;
    unsigned int seed = 42;
    bool help = false;
};

CLIArgs parseArgs(int argc, char* argv[]) {
    CLIArgs args;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--instance") == 0 && i + 1 < argc) {
            args.instance = argv[++i];
        } else if (strcmp(argv[i], "--variant") == 0 && i + 1 < argc) {
            args.variant = argv[++i];
        } else if (strcmp(argv[i], "--population") == 0 && i + 1 < argc) {
            args.population = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--generations") == 0 && i + 1 < argc) {
            args.generations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            args.block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args.seed = (unsigned int)atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            args.help = true;
        }
    }
    
    return args;
}

// ============================================================================
// Print GPU info
// ============================================================================

void printGPUInfo() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("ERROR: No CUDA-capable device found\n");
        return;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== GPU: %s ===\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %.1f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("  Shared memory per block: %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("\n");
}

// ============================================================================
// Run and print results
// ============================================================================

void runAndReport(const char* name, GAResult (*func)(const Instance&, const GAConfig&),
                  const Instance& inst, const GAConfig& config) {
    printf("\n>>> Ejecutando %s...\n", name);
    GAResult result = func(inst, config);
    printGAResult(result, name);
    
    // Print best chromosome summary
    if (!result.best_chromosome.empty()) {
        int selected = 0;
        for (int g : result.best_chromosome) selected += g;
        printf("  Items seleccionados: %d / %d\n", selected, inst.n_items);
    }
    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    CLIArgs args = parseArgs(argc, argv);
    
    if (args.help) {
        printUsage(argv[0]);
        return 0;
    }
    
    printf("=============================================================\n");
    printf("  Mochila GA CUDA — Actividad 3 INFO1195\n");
    printf("=============================================================\n\n");
    
    // GPU info
    printGPUInfo();
    
    // Load instance
    std::string data_dir = "data/" + args.instance;
    Instance inst = loadInstance(data_dir);
    printInstanceSummary(inst);
    
    // Configure GA
    GAConfig config = defaultConfig();
    config.population_size = args.population;
    config.generations = args.generations;
    config.block_size = args.block_size;
    config.seed = args.seed;
    
    // Scale penalties based on instance size
    // Key insight: penalty per hard-constraint violation must be >> max possible value
    // so that ANY feasible solution is preferred over ANY infeasible one
    float scale = (float)inst.n_items / 100.0f;
    config.alpha = 100.0f * scale;     // Per unit weight excess
    config.beta = 100.0f * scale;      // Per unit volume excess
    config.gamma = 50.0f * scale;      // Per category violation (soft)
    // Hard constraints: penalty per violation must be so large that
    // ANY feasible solution (even value=0) beats ANY infeasible one
    // This is the standard "death penalty" approach for hard constraints
    config.delta = 1000000.0f;  // Per incompatibility (HARD - effectively infinite)
    config.epsilon = 1000000.0f; // Per dependency (HARD - effectively infinite)
    
    printf("\n=== Configuracion AG ===\n");
    printf("  Poblacion: %d\n", config.population_size);
    printf("  Generaciones: %d\n", config.generations);
    printf("  Block size: %d\n", config.block_size);
    printf("  Semilla: %u\n", config.seed);
    printf("  Cruzamiento: %.2f\n", config.crossover_rate);
    printf("  Mutacion: %.4f\n", config.mutation_rate);
    printf("  Elitismo: %d\n", config.elitism_count);
    printf("  Torneo: %d\n", config.tournament_size);
    printf("  Penalizaciones: a=%.1f b=%.1f c=%.1f d=%.1f e=%.1f\n",
           config.alpha, config.beta, config.gamma, config.delta, config.epsilon);
    
    // ---- CPU Variant ----
    if (args.variant == "cpu" || args.variant == "all") {
        runAndReport("CPU Secuencial", runGA_CPU, inst, config);
    }
    
    // ---- CUDA Basic Variant ----
    if (args.variant == "cuda" || args.variant == "all") {
        runAndReport("CUDA Basico", runGA_CUDA_Basic, inst, config);
    }
    
    // ---- CUDA Optimized Variant ----
    if (args.variant == "cuda_opt" || args.variant == "all") {
        runAndReport("CUDA Optimizado", runGA_CUDA_Optimized, inst, config);
    }
    
    printf("=============================================================\n");
    printf("  Ejecucion completada\n");
    printf("=============================================================\n");
    
    return 0;
}
