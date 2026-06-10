// genetic_algorithm_cpu.hpp - Algoritmo genético secuencial en CPU
#ifndef GA_CPU_HPP
#define GA_CPU_HPP

#include "types.hpp"

// Ejecuta el AG secuencial en CPU
// Retorna el resultado con métricas de rendimiento
GAResult runGA_CPU(const Instance& inst, const GAConfig& config);

// Evaluación de fitness para un cromosoma individual
int evaluateFitness(Chromosome& chrom, const Instance& inst, const GAConfig& config);

// Verifica si un cromosoma es factible (cumple restricciones duras)
bool isFeasible(const Chromosome& chrom, const Instance& inst);

// Genera población inicial aleatoria
void initializePopulation(std::vector<Chromosome>& population, const Instance& inst, unsigned int seed);

// Selección por torneo
int tournamentSelect(const std::vector<Chromosome>& population, int tournament_size, unsigned int& rng_state);

// Cruzamiento de un punto
void crossover(Chromosome& parent1, Chromosome& parent2, Chromosome& child1, Chromosome& child2, 
               float crossover_rate, unsigned int& rng_state);

// Mutación por flip de bits
void mutate(Chromosome& chrom, float mutation_rate, unsigned int& rng_state);

#endif // GA_CPU_HPP
