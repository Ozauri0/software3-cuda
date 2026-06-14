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
│   ├── small/   (100 ítems, 99 incompatibilidades)
│   ├── medium/  (1000 ítems, 99 incompatibilidades)
│   └── large/   (10000 ítems, 499 incompatibilidades)
├── scripts/
│   ├── generate_instances.py   # Generador de instancias CSV
│   └── run_experiments.py       # Experimentos batch → CSV
├── results/
│   └── resultados.csv          # Resultados experimentales (270 ejecuciones)
├── .agents.md                  # Contexto para agentes AI
├── preparacion-informe.md      # Info para informe (rubrica, estructura)
├── Makefile
└── build.sh
```

## Resultados Experimentales

270 ejecuciones completas: 3 instancias × 3 variantes × 3 poblaciones × 10 repeticiones.
Cada configuración se ejecutó 10 veces con semillas diferentes (42-51) para calcular promedios.

### Speed-up por instancia y población

| Instancia | Items | Población | CPU (ms) | CUDA (ms) | CUDA Opt (ms) | Speed-up |
|---|---|---|---|---|---|---|
| Small | 100 | 1,024 | 606 | 105 | 106 | **~6x** |
| Small | 100 | 4,096 | 2,686 | 113 | 120 | **~24x** |
| Small | 100 | 16,384 | 11,448 | 400 | 398 | **~29x** |
| Medium | 1,000 | 1,024 | 7,808 | 535 | 531 | **~15x** |
| Medium | 1,000 | 4,096 | 33,094 | 708 | 668 | **~47x** |
| Medium | 1,000 | 16,384 | 132,018 | 3,173 | 2,980 | **~44x** |
| Large | 10,000 | 1,024 | 27,569 | 2,320 | 2,327 | **~12x** |
| Large | 10,000 | 4,096 | 113,614 | 3,397 | 3,390 | **~33x** |
| Large | 10,000 | 16,384 | 518,126 | 14,273 | 14,262 | **~36x** |

### Calidad de solución (valor promedio)

| Instancia | CPU | CUDA | CUDA Opt |
|---|---|---|---|
| Small | 3,142 | 3,152 | 3,152 |
| Medium | 136,412 | 135,595 | 135,595 |
| Large | 1,607,068 | 1,614,894 | 1,614,894 |

### Porcentaje de soluciones factibles

| Instancia | Small | Medium | Large |
|---|---|---|---|
| Promedio | ~56% | ~36% | ~0.3% |

### Efecto del tamaño de bloque (Medium, pop=4096)

| Block Size | Tiempo (ms) | Speed-up vs CPU |
|---|---|---|
| 128 | 645 | **~51x** |
| 256 | 668 | **~49x** |
| 512 | 905 | **~37x** |

## Conclusiones

1. **Speed-up crece con el tamaño de la instancia**: de ~6x (small, 100 ítems) hasta ~47x (medium, 1000 ítems). A más ítems, más paralelismo aprovecha la GPU.

2. **CUDA Optimizado es 5-10% más rápido** que CUDA básico en medium (668ms vs 708ms) y produce exactamente los mismos valores de solución — la optimización acelera sin sacrificar calidad.

3. **La calidad de solución es consistente** entre CPU y CUDA — el paralelismo no degrada la búsqueda.

4. **Factibilidad disminuye con el tamaño**: small 56% → medium 36% → large 0.3%. Con más ítems y restricciones, es más difícil encontrar soluciones válidas.

5. **Block size 128 es el más rápido** en medium (~645ms), seguido de 256 (~668ms) y 512 (~905ms).

6. **CPU con 16,384 individuos en large toma ~8.6 minutos** — CUDA lo hace en ~14 segundos (~36x speedup).

## Hardware

- **GPU:** NVIDIA GeForce RTX 3060 (12GB VRAM, 28 SMs, Compute 8.6)
- **Driver:** 591.86 (CUDA 13.1)
- **CUDA Toolkit:** 13.3
- **Compiler:** nvcc + MSVC 2022 Build Tools
