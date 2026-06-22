# Guía de Defensa — Mochila GA CUDA

## Instrucciones para el Presentador

Esta guía sigue el orden exacto de la presentación HTML (12 slides). Cada sección indica qué decir en voz alta, qué mostrar en pantalla y posibles preguntas del profesor.

---

## Slide 1 — Título

**Qué decir:**
> "Buenos días. Somos el grupo 2 de la sección [X]. Nuestra actividad 3 consistió en implementar y paralelizar mediante CUDA un algoritmo genético para resolver una versión extendida del problema de la mochila, considerando restricciones de peso, volumen, categorías, incompatibilidades y dependencias. El foco principal no era solo obtener una buena solución, sino demostrar comprensión sobre qué partes del algoritmo genético pueden ejecutarse eficientemente en GPU y cómo medir el impacto real de la paralelización."

**Datos clave en pantalla:**
- 161x speed-up máximo (RTX 4090, medium, pop=16,384)
- 2 GPUs evaluadas: RTX 3060 y RTX 4090
- 570 experimentos totales (285 por GPU × 2)

---

## Slide 2 — El Problema de la Mochila Extendido

**Qué decir:**
> "El problema clásico de la mochila es NP-hard: dado un conjunto de ítems con valor y peso, seleccionar un subconjunto que maximice el valor sin exceder la capacidad. Nosotros trabajamos con una versión extendida que agrega 4 restricciones adicionales: volumen máximo, categorías con mínimo/máximo, incompatibilidades entre pares de ítems y dependencias obligatorias."

> "La función de aptitud combina el valor total con penalizaciones. Para las restricciones duras (peso, volumen, incompatibilidades, dependencias) usamos un 'death penalty' de 1,000,000, lo que garantiza que cualquier solución factible sea siempre preferida sobre cualquier solución infeasible. Las categorías son tratadas como restricciones blandas."

**Preguntas probables:**
- *¿Por qué NP-hard?* → Porque el espacio de búsqueda crece exponencialmente: 2^n combinaciones posibles.
- *¿Por qué death penalty y no eliminar individuos inválidos?* → Porque permite explorar soluciones cercanas a la factibilidad que, mediante cruzamiento o mutación, puedan transformarse en soluciones válidas de buena calidad.
- *¿Qué pasaría si las penalizaciones fueran muy bajas?* → Las soluciones infeasibles dominarían la población porque ignorarían las restricciones.

---

## Slide 3 — 3 Variantes del Algoritmo Genético

**Qué decir:**
> "Implementamos tres variantes. La CPU secuencial es nuestra línea base: un AG clásico en C++ con selección por torneo k=5, cruzamiento de un punto con 85% de probabilidad, mutación por flip de bits al 1% y elitismo preservando los 4 mejores individuos."

> "La variante CUDA básico paraleliza la evaluación de fitness: cada hilo de la GPU evalúa un individuo completo en paralelo. La población permanece en memoria global durante toda la evolución, minimizando transferencias host↔device."

> "La variante CUDA optimizado agrega shared memory para los datos de ítems, lo que permite reutilizar los valores, pesos y volúmenes dentro de cada bloque. Cuando la instancia tiene más de 10,000 ítems, el tamaño de shared memory necesaria excede los 48KB disponibles, así que el sistema hace un fallback automático al kernel básico."

**Preguntas probables:**
- *¿Por qué paralelizar la evaluación de fitness y no la selección?* → Porque la evaluación es el cuello de botella: cada individuo recorre todos los ítems de forma independiente. La selección depende de los fitness calculados, así que primero se paraleliza la evaluación.
- *¿Qué es shared memory?* → Es una memoria de alta velocidad compartida entre hilos dentro de un mismo bloque. Es más rápida que la memoria global pero está limitada a 48KB por bloque.
- *¿Qué hace el fallback?* → Cuando n_items × 4 arrays × 4 bytes > 48KB, el kernel optimizado no puede cargar todos los datos a shared memory, así que se usa el kernel básico que accede directamente a memoria global.

---

## Slide 4 — Instancias de Prueba y Hardware

**Qué decir:**
> "Generamos tres instancias: small con 100 ítems para verificación funcional, medium con 1,000 ítems para comparación básica, y large con 10,000 ítems para evaluar el paralelismo real de la GPU. Las capacidades W y V se calcularon como el 40% de la suma total de pesos y volúmenes."

> "Las pruebas se ejecutaron en dos equipos: el servidor de la universidad con una RTX 4090 y un i9-14900KF, y nuestro PC local con una RTX 3060 y un Ryzen 5 5600X. Cada configuración se ejecutó 10 veces con semillas de la 42 a la 51 para calcular promedios y desviación estándar."

**Preguntas probables:**
- *¿Por qué 40% para W y V?* → Es un valor estándar que garantiza que existan soluciones factibles pero también que existan soluciones inválidas. Si fuera 80%, casi todos los ítems cabrían y el problema sería trivial.
- *¿Por qué 10 repeticiones?* → Para calcular desviación estándar y asegurar que los resultados son reproducibles, no producto de una semilla favorable.
- *¿Qué significan las semillas?* → Controlan la inicialización de la población y los operadores genéticos. Usar las mismas semillas garantiza reproducibilidad.

---

## Slide 5 — Resultados RTX 4090

**Qué decir:**
> "En la RTX 4090 del servidor, los resultados son impresionantes. Para la instancia medium con población 16,384, el CPU tarda 62,574 milisegundos mientras que CUDA optimizado tarda solo 388 milisegundos, logrando un speed-up de 161.1x. Esto demuestra que la GPU aprovecha al máximo sus 16,384 CUDA cores para evaluar toda la población en paralelo."

> "Para la instancia large con 10,000 ítems, incluso con población de 16,384, el speed-up alcanza 70.1x. La desviación estándar de CUDA es casi nula (±0-2ms), lo que indica que la GPU es mucho más consistente que la CPU."

> "Las soluciones factibles son idénticas entre CPU y CUDA: 55.8% en small, 36.3% en medium y 0.4% en large. Esto valida que el paralelismo no degrada la calidad de la búsqueda."

**Preguntas probables:**
- *¿Por qué la factibilidad baja en large?* → Con 499 incompatibilidades y 10,000 ítems, la probabilidad de generar un individuo que respete todas las restricciones simultáneamente es astronómicamente baja. El espacio de búsqueda es 2^10000.
- *¿Por qué CUDA es tan consistente?* → Porque todos los hilos ejecutan exactamente la misma secuencia de instrucciones en la misma arquitectura, sin variaciones de cache o branch prediction como en CPU.
- *¿Qué significa 161x?* → Que la GPU completa en 388ms lo que la CPU tarda 62,574ms.

---

## Slide 6 — Resultados RTX 3060

**Qué decir:**
> "En la RTX 3060 local, los speed-ups son menores pero仍然 impresionantes. Para medium con pop=4,096 logramos 49.6x, y para large con pop=16,384 llegamos a 36.3x. La diferencia con la 4090 se debe a que la 3060 tiene solo 3,584 CUDA cores versus los 16,384 de la 4090."

> "Un dato importante: la CPU del Ryzen 5 5600X es más lenta que la del i9-14900KF (33,094ms vs 15,186ms en medium pop=4,096), lo que también contribuye a que el speed-up sea menor en la 3060."

**Preguntas probables:**
- *¿Por qué el speed-up de la 3060 es menor que el de la 4090?* → La 3060 tiene 4.6x menos cores, así que con poblaciones grandes no puede saturar toda la GPU. Además, la CPU del Ryzen es más lenta, lo que reduce el denominator del speed-up.

---

## Slide 7 — Comparativa Directa

**Qué decir:**
> "Aquí vemos lado a lado los resultados de ambas GPUs. Los tiempos de CUDA optimizado son consistentemente menores en la 4090. Para pop=16,384 en medium, la 4090 es 7.7x más rápida que la 3060 (388ms vs 2,980ms). El CPU de la 4090 también es 2.2x más rápido que el de la 3060."

> "Esto demuestra que el beneficio de la paralelización escala con la potencia de la GPU: a más cores, más individuos se evalúan simultáneamente, y el overhead de transferencia se distribuye entre más trabajo."

**Preguntas probables:**
- *¿Por qué la 4090 es 7.7x más rápida y no 4.6x (proporción de cores)?* → Porque la 4090 también tiene mayor ancho de banda de memoria (1 TB/s vs 360 GB/s), más shared memory por SM, y mejor occupancy.
- *¿Qué es occupancy?* → Es la proporción de warps activos versus el máximo posible. A mayor occupancy, mejor se aprovechan los cores de la GPU.

---

## Slide 8 — Tendencia del Speed-up

**Qué decir:**
> "Este gráfico muestra claramente la tendencia: a mayor población, mayor speed-up. Con pop=1,024 el speed-up es modesto (~6-14x) porque el overhead de transferencia host↔device domina. Pero con pop=16,384, la GPU puede evaluar todos los individuos casi en un solo ciclo masivo."

> "La RTX 4090 escala exponencialmente con la población, alcanzando 161x. La RTX 3060 se estabiliza alrededor de 44x porque sus 3,584 cores se saturen antes."

**Preguntas probables:**
- *¿Por qué con pop=1,024 el speed-up es bajo?* → Porque el tiempo de transferencia de datos entre CPU y GPU es fijo (~5ms), y con pocos individuos el tiempo de cómputo es pequeño en comparación.
- *¿Qué es Ley de Amdahl?* → Establece que el speed-up máximo de un sistema paralelo está limitado por la fracción secuencial del algoritmo. Aquí, la transferencia de datos es la fracción secuencial.

---

## Slide 9 — Calidad y Factibilidad

**Qué decir:**
> "Un resultado clave es que la calidad de solución es prácticamente idéntica entre CPU y CUDA. Para medium, el valor promedio es 136,073 tanto en CPU como en CUDA. Esto confirma que el paralelismo no degrada la búsqueda."

> "La factibilidad disminuye con el tamaño: 55.8% en small, 36.3% en medium, 0.4% en large. Esto es esperado porque con más ítems e incompatibilidades, es más difícil encontrar soluciones que respeten todas las restricciones simultáneamente."

> "El efecto del block size muestra que 128 es óptimo para la RTX 3060 con esta carga: 645ms vs 905ms con block size 512. Bloques más grandes reducen el occupancy."

**Preguntas probables:**
- *¿Por qué la factibilidad es idéntica entre CPU y CUDA?* → Porque usamos las mismas semillas, los mismos operadores genéticos y las mismas penalizaciones. El algoritmo es determinista dado el mismo estado inicial.
- *¿Qué es occupancy y por qué 128 es mejor?* → Con block size 128, cada SM puede alojar más bloques simultáneamente, aprovechando mejor los recursos. Con 512, cada bloque consume más shared memory y registros, reduciendo el número de bloques concurrentes.
- *¿Por qué las transferencias son solo 0.7%?* → Porque la población se mantiene en GPU durante toda la evolución. Solo se transfieren los datos al inicio y los resultados al final.

---

## Slide 10 — Fragmentos de Código

**Qué decir:**
> "El kernel básico recibe la población aplanada en memoria global. Cada hilo calcula su índice global, y si está dentro del rango, recorre todos los ítems acumulando valor, peso y volumen. Luego aplica las penalizaciones y guarda el fitness."

> "El kernel optimizado carga los datos de ítems a shared memory usando una carga coalescente: cada hilo carga varios elementos (stride = blockDim.x). Después usa __syncthreads() para sincronizar, y todos los hilos del bloque leen desde shared memory, que es ~100x más rápida que la memoria global."

**Preguntas probables:**
- *¿Qué es coalescente?* → Significa que hilos consecutivos acceden a direcciones de memoria consecutivas, lo que permite que el hardware de memoriacombine las solicitudes en pocas transacciones grandes.
- *¿Qué hace __syncthreads()?* → Sincroniza todos los hilos dentro de un bloque. Ningún hilo puede pasar esta línea hasta que todos los demás hilos del bloque la alcancen. Es necesario después de cargar datos a shared memory.
- *¿Por qué se usa int en vez de float para fitness?* → Para evitar errores de precisión en las comparaciones de selección por torneo. Con enteros, los empates se resuelven de forma consistente.

---

## Slide 11 — Análisis Técnico

**Qué decir:**
> "Las optimizaciones que funcionaron mejor fueron: shared memory (5-10% de mejora), la inicialización greedy que mejora la factibilidad en ~5%, y el block size de 128 que da mejor occupancy."

> "Las limitaciones principales son: shared memory no alcanza para large (160KB > 48KB), el block size 512 es 35% más lento, y la factibilidad en large es casi nativa. El elitismo requiere transferir datos de GPU a CPU para encontrar los mejores individuos."

> "La razón fundamental de por qué CUDA es más rápido es que la evaluación de fitness es el cuello de botella del AG, y cada individuo puede evaluarse completamente en paralelo sin dependencias. Con 16,384 cores en la 4090, se evalúan 16,384 individuos simultáneamente."

**Preguntas probables:**
- *¿Qué es curand y por qué es inferior a RNG en CPU?* → curand es la librería de CUDA para generación de números pseudoaleatorios. Tiene calidad estadística inferior a algoritmos como Mersenne Twister, pero es muy rápido y paralelizable.
- *¿Cómo se miden los tiempos de kernel vs transferencia?* → Usando CUDA events: cudaEventRecord antes y después de cada operación, y cudaEventElapsedTime para calcular la diferencia.
- *¿Qué pasaría si quisiéramos mejorar la factibilidad en large?* → Necesitaríamos operadores de reparación que modifiquen individuos inválidos para hacerlos factibles, o una representación alternativa que evite generar violaciones.

---

## Slide 12 — Conclusiones

**Qué decir:**
> "Como conclusiones principales: logramos speed-ups de hasta 161x en la RTX 4090 y 49x en la RTX 3060. Implementamos 3 variantes funcionales validadas en 2 GPUs diferentes, con 570 experimentos totales. La calidad de solución es idéntica entre CPU y CUDA, lo que valida la correctitud de la implementación paralela."

> "Los aprendizajes clave son: paralelizar la evaluación de fitness es la optimización más impactante, shared memory es efectiva solo con menos de 3,000 ítems, el speed-up escala con el tamaño de población, y la 'death penalty' es necesaria para manejar restricciones duras."

> "Como trabajo futuro, se podrían implementar operadores de reparación para mejorar la factibilidad en instancias grandes, usar memoria constante para parámetros, y explorar CUDA streams para solapar kernels con transferencias."

---

## Preguntas y Respuestas para Estudiar

### Preguntas sobre el Problema

**P: ¿Qué es el problema de la mochila extendido y cómo difiere del clásico?**
R: El clásico solo considera peso y valor. La versión extendida agrega volumen, categorías con mínimo/máximo, incompatibilidades entre pares de ítems y dependencias obligatorias. Esto convierte un problema NP-hard en uno aún más complejo, donde la factibilidad depende de múltiples restricciones cruzadas.

**P: ¿Por qué se usa un algoritmo genético y no un algoritmo exacto?**
R: Porque el problema es NP-hard. Con 10,000 ítems, el espacio de búsqueda es 2^10000, lo que hace inviable cualquier algoritmo exacto. Los AG exploran una fracción del espacio y encuentran soluciones de buena calidad en tiempo razonable.

**P: ¿Qué representa la variable xi en la formulación?**
R: Es una variable binaria: xi=1 si el ítem i está seleccionado, xi=0 si no. El cromosoma X = (x1, x2, ..., xn) representa una solución completa.

### Preguntas sobre el Algoritmo Genético

**P: ¿Cómo funciona la selección por torneo?**
R: Se eligen k individuos al azar de la población y el de mayor fitness gana el torneo. Con k=5, hay alta presión selectiva pero se mantiene diversidad. Es simple de paralelizar porque cada torneo es independiente.

**P: ¿Por qué cruzamiento de un punto y no de dos puntos?**
R: El cruzamiento de un punto es más simple de implementar y paralelizar. Con 10,000 ítems, la diferencia entre un punto y dos puntos es marginal en calidad de solución.

**P: ¿Qué hace el elitismo y por qué es importante?**
R: Preserva los 4 mejores individuos de cada generación sin pasar por selección/cruzamiento/mutación. Esto garantiza que la mejor solución encontrada nunca se pierda, acelerando la convergencia.

**P: ¿Por qué la tasa de mutación es tan baja (1%)?**
R: Una mutación alta destruiría las soluciones buenas encontradas por el cruzamiento. El 1% permite exploración sin destruir el progreso de la evolución.

### Preguntas sobre CUDA

**P: ¿Qué es un kernel en CUDA?**
R: Es una función que se ejecuta en paralelo en la GPU. Cada hilo ejecuta el mismo código pero con un índice diferente, procesando un dato diferente.

**P: ¿Qué es la memoria global vs shared memory?**
R: La memoria global es grande (12GB en la 3060) pero lenta (~360 GB/s). La shared memory es pequeña (48KB por bloque) pero muy rápida (~19 TB/s). Shared memory se usa para datos reutilizados dentro de un bloque.

**P: ¿Qué es un warp?**
R: Un grupo de 32 hilos que ejecutan las mismas instrucciones en lock-step. Si hilos de un warp toman caminos diferentes (divergencia de warps), se ejecutan secuencialmente, reduciendo el rendimiento.

**P: ¿Qué es coalescente?**
R: Cuando hilos consecutivos acceden a direcciones de memoria consecutivas. El hardware de memoria puede combinar estas solicitudes en pocas transacciones grandes, maximizando el ancho de banda.

**P: ¿Por qué se usa -arch=sm_86?**
R: Porque nuestro driver (591.86) soporta CUDA 13.1 pero el toolkit es 13.3. El flag genera código SASS (ensamblador GPU) directamente para la arquitectura 8.6, evitando PTX que sería incompatible.

### Preguntas sobre Resultados

**P: ¿Por qué el speed-up crece con la población?**
R: Con más individuos, hay más trabajo paralelizable. El overhead de transferencia es fijo (~5ms), así que con 16,384 individuos el cómputo domina, mientras que con 1,024 la transferencia representa una fracción significativa.

**P: ¿Por qué la 4090 es 7.7x más rápida que la 3060 y no 4.6x (proporción de cores)?**
R: Porque la 4090 también tiene mayor ancho de banda de memoria (1 TB/s vs 360 GB/s), más shared memory por SM, mejor occupancy, y el i9-14900KF es 2.2x más rápido que el Ryzen 5 5600X.

**P: ¿Por qué la factibilidad es idéntica entre CPU y CUDA?**
R: Porque usamos las mismas semillas, los mismos operadores y las mismas penalizaciones. El algoritmo es determinista dado el mismo estado inicial, sin importar dónde se ejecute.

**P: ¿Qué significa "death penalty" para las restricciones duras?**
R: Que una sola violación de restricción dura genera una penalización tan alta (1,000,000) que el fitness se vuelve muy negativo, haciendo que la solución sea siempre peor que cualquier solución factible.

**P: ¿Por qué la factibilidad baja en large?**
R: Con 499 incompatibilidades y 10,000 ítems, el espacio de búsqueda es 2^10000. La probabilidad de generar un individuo que respete todas las restricciones simultáneamente es astronómicamente baja. Sin operadores de reparación, el GA no puede encontrar soluciones factibles.

**P: ¿Qué es el block size y por qué 128 es óptimo?**
R: Es el número de hilos por bloque. Con 128, cada SM puede alojar más bloques, mejorando el occupancy. Con 512, cada bloque consume más recursos, reduciendo bloques concurrentes y aumentando el tiempo.

**P: ¿Qué mide la desviación estándar en los resultados?**
R: La variabilidad entre las 10 ejecuciones con diferentes semillas. CUDA tiene desviación casi nula (±0-2ms) porque es más consistente que CPU (±10-7,000ms).

### Preguntas sobre Diseño Experimental

**P: ¿Por qué se usaron 10 repeticiones?**
R: Para calcular estadísticas confiables (promedio y desviación estándar) y demostrar que los resultados son reproducibles, no producto de una semilla favorable.

**P: ¿Por qué las capacidades W y V son 40% del total?**
R: Para garantizar que existan tanto soluciones factibles como inválidas. Con 80%, casi todos los ítems cabrían (trivial). Con 20%, sería casi imposible encontrar soluciones factibles.

**P: ¿Cómo se genera la población inicial?**
R: Híbrida: 40% greedy (items ordenados por valor/peso), 33% semi-aleatorio (20% de probabilidad por gen), 27% totalmente aleatorio (50% por gen). El greedy sesga hacia soluciones con menos items, mejorando la factibilidad.

**P: ¿Por qué se usaron dos GPUs diferentes?**
R: Para evaluar la escalabilidad del algoritmo en hardware de diferentes generaciones y capacidades. La 3060 (Turing, 3,584 cores) y la 4090 (Ada Lovelace, 16,384 cores) representan rangos distintos de potencia GPU.

### Preguntas sobre Entregables

**P: ¿Qué contiene el CSV de resultados?**
R: 285 filas por GPU (3 instancias × 3 variantes × 3 poblaciones × 10 repeticiones), con columnas: tiempo total, tiempo fitness, tiempo kernel, transferencias H→D y D→H, mejor valor, fitness, % factibilidad, ítems seleccionados, y desviación estándar.

**P: ¿Cómo se compila el programa?**
R: Con nvcc usando -gencode para múltiples arquitecturas (sm_75 a sm_90), lo que genera un binario compatible con cualquier GPU NVIDIA moderna (GTX 1660 en adelante).

**P: ¿El programa funciona en cualquier GPU NVIDIA?**
R: Sí, siempre que tenga compute capability ≥ 7.5 (Turing o posterior). La compilación con -gencode genera código para 5 arquitecturas simultáneamente.
