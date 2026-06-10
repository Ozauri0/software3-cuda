# Mochila GA CUDA — Actividad 3 INFO1195

Implementación de un **algoritmo genético paralelizado en CUDA** para resolver una versión extendida del problema de la mochila (knapsack).

## Descripción del Problema

Se selecciona un subconjunto de ítems con valor, peso, volumen y categoría, maximizando el valor total sin exceder las capacidades de la mochila, respetando restricciones de:

- **Peso máximo** (W) y **volumen máximo** (V)
- **Categorías** (mínimo/máximo de ítems por categoría)
- **Incompatibilidades** (ciertos pares no pueden coexistir)
- **Dependencias** (si se selecciona un ítem, debe seleccionarse otro asociado)

## Variantes Implementadas

| Variante | Descripción |
|---|---|
| **CPU Secuencial** | AG baseline en C++ con selección por torneo, cruzamiento de un punto, mutación por flip y elitismo |
| **CUDA Básico** | Paralelización en GPU con kernels para fitness, selección, cruzamiento y mutación. Población en memoria global |
| **CUDA Optimizado** | Agrega shared memory para datos de items, inicialización greedy, medición precisa de kernels con CUDA events |

## Compilación

**Requisitos:** CUDA Toolkit 13.3, Visual Studio Build Tools 2022, Windows SDK 10.0

```bash
# Configurar entorno y compilar
cd software3-cuda
CUDA_BIN="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/bin"
MSVC_BIN="/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64"
export PATH="$CUDA_BIN:$MSVC_BIN:$PATH"

# Compilar con -arch=sm_86 (requerido por driver 591.86)
nvcc -O3 -std=c++17 -arch=sm_86 \
  -I"/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3/include" \
  -o mochila_ga_cuda \
  src/main.cu src/instance_loader.cpp src/genetic_algorithm_cpu.cpp \
  src/fitness_cuda.cu src/genetic_algorithm_cuda.cu \
  -lcudart -lcurand
```

## Ejecución

```bash
# Generar instancias de prueba
python scripts/generate_instances.py

# Ejecutar todas las variantes
./mochila_ga_cuda --instance small --variant all --population 4096 --generations 500 --block 256 --seed 42

# Ejecutar solo CUDA optimizado
./mochila_ga_cuda --instance medium --variant cuda_opt --population 4096 --generations 300 --block 256 --seed 42

# Ejecutar experimentos batch (genera CSV)
python scripts/run_experiments.py
```

### Parámetros CLI

| Parámetro | Default | Descripción |
|---|---|---|
| `--instance` | small | Tamaño: `small` (100), `medium` (1000), `large` (10000) |
| `--variant` | all | Variante: `cpu`, `cuda`, `cuda_opt`, `all` |
| `--population` | 4096 | Tamaño de población |
| `--generations` | 500 | Número de generaciones |
| `--block` | 256 | Tamaño de bloque CUDA |
| `--seed` | 42 | Semilla RNG |

## Estructura del Proyecto

```
software3-cuda/
├── src/
│   ├── main.cu                         # Entry point, CLI, integración variantes
│   ├── types.hpp                       # Item, Chromosome, Instance, GAConfig, GAResult
│   ├── common.hpp                      # CUDA_CHECK, timers, defaultConfig
│   ├── instance_loader.cpp/hpp         # Carga CSV (items, categories, incompat, deps)
│   ├── genetic_algorithm_cpu.cpp/hpp   # AG secuencial CPU
│   ├── genetic_algorithm_cuda.cu/cuh   # AG CUDA completo
│   └── fitness_cuda.cu/cuh             # Kernels CUDA fitness (básico + optimizado)
├── data/
│   ├── small/   (100 ítems)
│   ├── medium/  (1000 ítems)
│   └── large/   (10000 ítems)
├── scripts/
│   ├── generate_instances.py   # Generador de instancias CSV
│   └── run_experiments.py       # Experimentos batch → CSV
├── results/
│   └── resultados.csv          # Resultados experimentales
├── .agents.md                  # Contexto para agentes AI
├── preparacion-informe.md      # Info para informe (rubrica, estructura)
├── Makefile
└── build.sh
```

## Resultados Preliminares

| Instancia | Items | CPU (ms) | CUDA Básico (ms) | CUDA Opt (ms) | Speed-up |
|---|---|---|---|---|---|
| Small | 100 | 1,861 | 106 | 104 | ~18x |
| Medium | 1,000 | 35,389 | 981 | 921 | ~38x |

## Hardware

- **GPU:** NVIDIA GeForce RTX 3060 (12GB VRAM, 28 SMs, Compute 8.6)
- **Driver:** 591.86 (CUDA 13.1)
- **CUDA Toolkit:** 13.3
- **Compiler:** nvcc + MSVC 2022 Build Tools
