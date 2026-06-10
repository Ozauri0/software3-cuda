#!/usr/bin/env python3
"""
run_experiments.py - Ejecuta todas las configuraciones experimentales y genera CSV.
Requiere que el binario mochila_ga_cuda esté compilado.
"""

import subprocess
import csv
import os
import sys
import time

BINARY = "./mochila_ga_cuda"
if sys.platform == "win32":
    BINARY = "./mochila_ga_cuda.exe"

RESULTS_DIR = "results"
CSV_FILE = os.path.join(RESULTS_DIR, "resultados.csv")

# Configuraciones experimentales
INSTANCES = ["small", "medium", "large"]
VARIANTS = ["cpu", "cuda", "cuda_opt"]
POPULATIONS = [1024, 4096, 16384]
GENERATIONS = {"small": 500, "medium": 300, "large": 100}
BLOCK_SIZES = [128, 256, 512]
REPETITIONS = 10
BASE_SEED = 42


def run_single(instance, variant, population, generations, block_size, seed):
    """Ejecuta una configuración individual y retorna los resultados parseados."""
    cmd = [
        BINARY,
        "--instance", instance,
        "--variant", variant,
        "--population", str(population),
        "--generations", str(generations),
        "--block", str(block_size),
        "--seed", str(seed),
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        # Parse output
        data = {}
        for line in output.split("\n"):
            line = line.strip()
            if "Tiempo total:" in line:
                data["time_total_ms"] = float(line.split(":")[-1].strip().replace(" ms", ""))
            elif "Tiempo fitness:" in line:
                data["time_fitness_ms"] = float(line.split(":")[-1].strip().replace(" ms", ""))
            elif "Mejor valor factible:" in line:
                data["best_value"] = int(line.split(":")[-1].strip())
            elif "Mejor fitness:" in line:
                data["best_fitness"] = int(line.split(":")[-1].strip())
            elif "Soluciones factibles:" in line:
                data["feasible_pct"] = float(line.split(":")[-1].strip().replace("%", ""))
            elif "Tiempo kernel CUDA:" in line:
                data["time_kernel_ms"] = float(line.split(":")[-1].strip().replace(" ms", ""))
            elif "Transfer H" in line and "D:" in line:
                data["time_h2d_ms"] = float(line.split(":")[-1].strip().replace(" ms", ""))
            elif "Transfer D" in line and "H:" in line:
                data["time_d2h_ms"] = float(line.split(":")[-1].strip().replace(" ms", ""))
            elif "Ítems seleccionados:" in line:
                # Extract "X / Y"
                parts = line.split(":")[-1].strip().split("/")
                data["items_selected"] = int(parts[0].strip())
                data["items_total"] = int(parts[1].strip())
        
        return data
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after 600s")
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # CSV columns
    fieldnames = [
        "instance", "variant", "population", "generations", "block_size", "seed",
        "time_total_ms", "time_fitness_ms", "time_kernel_ms", "time_h2d_ms", "time_d2h_ms",
        "best_value", "best_fitness", "feasible_pct", "items_selected", "items_total"
    ]
    
    with open(CSV_FILE, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_runs = len(INSTANCES) * len(VARIANTS) * len(POPULATIONS) * REPETITIONS
        run_count = 0
        
        for instance in INSTANCES:
            gens = GENERATIONS[instance]
            
            for variant in VARIANTS:
                for pop in POPULATIONS:
                    for rep in range(REPETITIONS):
                        seed = BASE_SEED + rep
                        run_count += 1
                        
                        print(f"[{run_count}/{total_runs}] {instance} | {variant} | pop={pop} | seed={seed}")
                        
                        data = run_single(instance, variant, pop, gens, 256, seed)
                        
                        if data:
                            data["instance"] = instance
                            data["variant"] = variant
                            data["population"] = pop
                            data["generations"] = gens
                            data["block_size"] = 256
                            data["seed"] = seed
                            writer.writerow(data)
                            csvfile.flush()
                            
                            print(f"  -> {data.get('time_total_ms', 0):.1f}ms | "
                                  f"val={data.get('best_value', 0)} | "
                                  f"fact={data.get('feasible_pct', 0):.1f}%")
                        else:
                            print(f"  -> FAILED")
        
        # Block size experiment (medium only, pop=4096)
        print(f"\n--- Block size experiment ---")
        for bs in BLOCK_SIZES:
            for rep in range(5):  # 5 repeticiones
                seed = BASE_SEED + rep
                run_count += 1
                print(f"[{run_count}] medium | cuda_opt | pop=4096 | block={bs} | seed={seed}")
                
                data = run_single("medium", "cuda_opt", 4096, 300, bs, seed)
                if data:
                    data["instance"] = "medium"
                    data["variant"] = f"cuda_opt_b{bs}"
                    data["population"] = 4096
                    data["generations"] = 300
                    data["block_size"] = bs
                    data["seed"] = seed
                    writer.writerow(data)
                    csvfile.flush()
                    
                    print(f"  -> {data.get('time_total_ms', 0):.1f}ms | "
                          f"val={data.get('best_value', 0)} | "
                          f"fact={data.get('feasible_pct', 0):.1f}%")
    
    print(f"\n=== Experimentos completados ===")
    print(f"Resultados guardados en {CSV_FILE}")


if __name__ == "__main__":
    main()
