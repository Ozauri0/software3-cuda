# Preparación del Informe Técnico

## Información General

- **Curso:** INFO1195 — Semana 12
- **Actividad 3:** Optimización paralela en GPU del problema extendido de la mochila mediante CUDA
- **Peso:** 20% de la asignatura
- **Componentes evaluados:**
  - Informe técnico: **30%**
  - Presentación + defensa + ejecución: **70%**

---

## Estructura del Informe (sugerida por la pauta)

### 1. Introducción, contexto y formulación extendida (20%)

Incluir:
- Descripción del problema de la mochila clásico
- Complejidad computacional (NP-hard)
- Aplicaciones reales (inversión, corte de materiales, carga de contenedores, etc.)
- Formulación matemática clásica
- Extensión a la versión con peso, volumen, categorías, incompatibilidades, dependencias
- Restricciones duras vs blandas (definir cuáles son cuáles)
- Función de aptitud y penalizaciones

### 2. Metodología y diseño experimental (20%)

Documentar:
- Parámetros del AG (tamaño población, generaciones, probabilidad cruzamiento/mutación, elitismo)
- 3 variantes: CPU, CUDA básica, CUDA optimizada
- Tamaños de instancia (100 / 1,000 / 10,000 ítems)
- Configuraciones de población (1024, 4096, 16384)
- Repeticiones: ≥10 por configuración
- Semillas registradas
- Hardware: CPU, GPU (RTX 3060), memoria, versión CUDA, versión driver
- Capacidades W y V calculadas como % del total

### 3. Descripción del AG y variantes (15%)

Explicar:
- Representación de la población en memoria
- Generación inicial
- Función de aptitud completa
- Selección por torneo
- Cruzamiento (tipo elegido y por qué)
- Mutación (tipo elegido y por qué)
- Elitismo
- Criterio de término
- Diferencias entre CPU / CUDA básica / CUDA optimizada

### 4. Resultados, tablas y gráficos (20%)

Presentar:
- Tabla de tiempos promedio por variante, instancia y población
- Tabla de speed-up (CPU vs CUDA básica, CPU vs CUDA optimizada)
- Tabla de mejor valor factible y fitness
- Tabla de % soluciones factibles
- Gráficos: speed-up vs tamaño de población
- Gráficos: tiempo vs tamaño de instancia
- Gráficos: efecto del block size
- Efecto de transferencias host↔device
- Desviación estándar de tiempos

### 5. Análisis técnico y conclusiones (20%)

Analizar:
- Por qué CUDA es más rápido (paralelismo de la evaluación fitness)
- Impacto real de cada optimización (shared memory, const memory, reducciones, etc.)
- Cuáles optimizaciones mejoraron y cuáles no
- Límites observados (divergencia warps, overhead transferencias, etc.)
- Relación calidad-tiempo (¿más población = mejor solución?)
- Limitaciones del enfoque
- Aprendizajes técnicos

### 6. Redacción, orden y trazabilidad (5%)

---

## Datos que necesitamos del programa (generar después de ejecutar)

Los siguientes datos saldrán del programa y se incluirán en el informe:

- [ ] Tiempo promedio total por configuración
- [ ] Desviación estándar de tiempos
- [ ] Tiempo de kernels CUDA (medido con eventos CUDA)
- [ ] Tiempo de transferencias host↔device
- [ ] Mejor valor factible encontrado
- [ ] Mejor fitness obtenido
- [ ] Porcentaje de soluciones factibles
- [ ] Speed-up por configuración
- [ ] Hardware reportado (CPU, GPU, RAM, CUDA version, driver)

## Notas para la Presentación (PPTX — 70%)

La presentación debe incluir:
- Introducción breve al problema
- Al menos 1 ejecución reproducible mostrada en vivo
- Resultados generados por el programa (consola o CSV)
- Justificación de restricciones, fitness, selección, cruzamiento, mutación, elitismo
- Diagramas de qué parte corre en CPU vs GPU
- Qué datos se transfieren host↔device
- Fragmentos breves de kernels si se desea mostrar código
- NO leer código completo — explicar decisiones de diseño

## Preguntas Técnicas Probables en la Defensa

1. ¿Por qué parallelizar la evaluación fitness y no la selección?
2. ¿Cómo se maneja la aleatoriedad en GPU?
3. ¿Qué datos están en memoria global vs shared?
4. ¿Por qué se usa memoria constante para ciertos datos?
5. ¿Cómo se calculan W y V (por qué 40%)?
6. ¿Qué restricciones son duras y por qué?
7. ¿Cómo se verifica que la solución final es factible?
8. ¿Qué hace el elitismo y por qué es necesario?
9. ¿Cómo se miden los tiempos de kernels vs transferencias?
10. ¿Qué optimizaciones no funcionaron y por qué?
