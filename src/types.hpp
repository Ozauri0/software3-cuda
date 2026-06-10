// types.hpp - Definiciones de tipos compartidos CPU/GPU
#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>
#include <vector>
#include <string>

// ============================================================================
// Item (ítem de la mochila)
// ============================================================================

struct Item {
    int id;
    int valor;
    int peso;
    int volumen;
    int categoria;
};

// ============================================================================
// Reglas de categorías
// ============================================================================

struct CategoryRule {
    int categoria;
    int minimo;
    int maximo;
};

// ============================================================================
// Incompatibilidad
// ============================================================================

struct Incompatibility {
    int id_item_a;
    int id_item_b;
};

// ============================================================================
// Dependencia
// ============================================================================

struct Dependency {
    int id_item;
    int id_requerido;
};

// ============================================================================
// Instancia del problema
// ============================================================================

struct Instance {
    std::vector<Item> items;
    std::vector<CategoryRule> category_rules;
    std::vector<Incompatibility> incompatibilities;
    std::vector<Dependency> dependencies;
    
    int capacidad_peso;      // W
    int capacidad_volumen;   // V
    int n_items;
    
    // Suma total para cálculo de capacidades
    int suma_total_pesos;
    int suma_total_volumenes;
};

// ============================================================================
// Cromosoma (solución)
// ============================================================================

// Representación: vector binario de largo n_items
// genes[i] = 1 si el ítem i está seleccionado, 0 si no
struct Chromosome {
    int fitness;         // Valor de aptitud (con penalizaciones)
    int valor_total;     // Valor puro sin penalizar
    bool factible;       // ¿Cumple todas las restricciones duras?
    std::vector<int> genes;  // 0 o 1 por cada ítem
};

// ============================================================================
// Configuración del AG
// ============================================================================

struct GAConfig {
    int population_size;    // Tamaño de población
    int generations;        // Número de generaciones
    int block_size;         // Tamaño de bloque CUDA
    unsigned int seed;      // Semilla para RNG
    
    // Probabilidades de operadores genéticos
    float crossover_rate;   // Probabilidad de cruzamiento (0.0 - 1.0)
    float mutation_rate;    // Probabilidad de mutación por gen (0.0 - 1.0)
    
    // Elitismo
    int elitism_count;      // Número de individuos elite a preservar
    
    // Penalizaciones (valores a justificar en el informe)
    float alpha;            // Penalización por exceso de peso
    float beta;             // Penalización por exceso de volumen
    float gamma;            // Penalización por violación de categoría
    float delta;            // Penalización por incompatibilidades
    float epsilon;          // Penalización por dependencias incumplidas
    
    // Selección por torneo
    int tournament_size;    // Tamaño del torneo
    
    // Variantes
    bool run_cpu;           // Ejecutar variante CPU
    bool run_cuda_basic;    // Ejecutar variante CUDA básico
    bool run_cuda_optimized;// Ejecutar variante CUDA optimizado
};

// ============================================================================
// Resultados de una ejecución
// ============================================================================

struct GAResult {
    // Tiempos
    double time_total_ms;          // Tiempo total de ejecución
    double time_fitness_ms;        // Tiempo de evaluación fitness
    double time_selection_ms;      // Tiempo de selección
    double time_crossover_ms;      // Tiempo de cruzamiento
    double time_mutation_ms;       // Tiempo de mutación
    double time_kernel_total_ms;   // Solo kernels CUDA
    double time_transfer_h2d_ms;   // Transferencia host→device
    double time_transfer_d2h_ms;   // Transferencia device→host
    
    // Solución
    int best_value;                // Mejor valor factible encontrado
    int best_fitness;              // Mejor fitness obtenido
    float feasible_percentage;     // % de soluciones factibles
    int best_solution_index;       // Índice del mejor individuo factible
    
    // Información adicional
    std::vector<int> best_chromosome;  // Mejor cromosoma factible
};

// ============================================================================
// Variantes de ejecución
// ============================================================================

enum class Variant {
    CPU_SEQUENTIAL,
    CUDA_BASIC,
    CUDA_OPTIMIZED
};

#endif // TYPES_HPP
