#!/usr/bin/env python3
"""
postprocess_csv.py - Calcula desviación estándar y corrige CSV de resultados.
NO reejecuta el algoritmo. Lee el CSV existente y agrega columnas faltantes.
"""

import csv
import os
from collections import defaultdict

RESULTS_DIR = "results"
INPUT_CSV = os.path.join("resultados.csv")
OUTPUT_CSV = os.path.join("resultados_procesado.csv")


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"ERROR: No se encontro {INPUT_CSV}")
        return

    # Read all rows
    rows = []
    with open(INPUT_CSV, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        for row in reader:
            rows.append(row)

    print(f"Leidas {len(rows)} filas de {INPUT_CSV}")

    # Group by (instance, variant, population) for std dev calculation
    groups = defaultdict(list)
    for row in rows:
        key = (row["instance"], row["variant"], row["population"])
        groups[key].append(row)

    # Calculate std dev for each group
    import math
    stats = {}
    for key, group_rows in groups.items():
        times = [float(r["time_total_ms"]) for r in group_rows if r["time_total_ms"]]
        values = [float(r["best_value"]) for r in group_rows if r["best_value"]]
        feasibles = [float(r["feasible_pct"]) for r in group_rows if r["feasible_pct"]]

        def std_dev(vals):
            if len(vals) < 2:
                return 0.0
            mean = sum(vals) / len(vals)
            variance = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            return math.sqrt(variance)

        stats[key] = {
            "time_std_ms": std_dev(times),
            "value_std": std_dev(values),
            "feasible_std": std_dev(feasibles),
            "time_mean": sum(times) / len(times) if times else 0,
            "value_mean": sum(values) / len(values) if values else 0,
            "feasible_mean": sum(feasibles) / len(feasibles) if feasibles else 0,
        }

    # Add new columns
    new_fields = fieldnames + ["time_std_ms", "value_std", "feasible_std"]
    if "items_selected" not in fieldnames:
        new_fields.append("items_selected")
    if "items_total" not in fieldnames:
        new_fields.append("items_total")

    # Write processed CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=new_fields)
        writer.writeheader()
        for row in rows:
            key = (row["instance"], row["variant"], row["population"])
            s = stats[key]
            row["time_std_ms"] = round(s["time_std_ms"], 2)
            row["value_std"] = round(s["value_std"], 2)
            row["feasible_std"] = round(s["feasible_std"], 2)
            # Fill missing items_selected/total with empty if not present
            if "items_selected" not in row:
                row["items_selected"] = ""
            if "items_total" not in row:
                row["items_total"] = ""
            writer.writerow(row)

    # Print summary table
    print(f"\nCSV procesado guardado en {OUTPUT_CSV}")
    print(f"\nResumen por configuracion:")
    print(f"{'Instancia':<8} {'Variante':<12} {'Pop':<6} {'Tiempo (ms)':<14} {'Std (ms)':<10} {'Valor':<10} {'Fact%':<8}")
    print("-" * 70)

    seen = set()
    for key in sorted(stats.keys()):
        if key not in seen:
            seen.add(key)
            s = stats[key]
            print(f"{key[0]:<8} {key[1]:<12} {key[2]:<6} {s['time_mean']:<14.1f} {s['time_std_ms']:<10.1f} {s['value_mean']:<10.0f} {s['feasible_mean']:<8.1f}")


if __name__ == "__main__":
    main()
