#!/usr/bin/env python3
"""
generate_instances.py - Generador de instancias para el problema de la mochila extendida.

Genera archivos CSV en data/small/, data/medium/, data/large/.
Cada directorio contiene: items.csv, category_rules.csv, incompatibilities.csv, dependencies.csv
"""

import os
import random
import argparse

def generate_instance(n_items, n_categories, incompatibility_ratio, dependency_ratio, 
                      weight_range, volume_range, value_range, seed, output_dir):
    """Genera una instancia completa del problema."""
    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generando instancia con {n_items} ítems, semilla={seed}...")
    
    # ---- Generate items ----
    items = []
    for i in range(n_items):
        cat = random.randint(0, n_categories - 1)
        peso = random.randint(weight_range[0], weight_range[1])
        volumen = random.randint(volume_range[0], volume_range[1])
        valor = random.randint(value_range[0], value_range[1])
        items.append((i, valor, peso, volumen, cat))
    
    # Write items.csv
    with open(os.path.join(output_dir, "items.csv"), "w") as f:
        f.write("id,valor,peso,volumen,categoria\n")
        for item in items:
            f.write(f"{item[0]},{item[1]},{item[2]},{item[3]},{item[4]}\n")
    
    # ---- Category rules ----
    cat_rules = []
    for c in range(n_categories):
        minimo = random.randint(0, max(1, n_items // (n_categories * 4)))
        maximo = random.randint(minimo + 1, max(minimo + 2, n_items // n_categories + 5))
        cat_rules.append((c, minimo, maximo))
    
    with open(os.path.join(output_dir, "category_rules.csv"), "w") as f:
        f.write("categoria,minimo,maximo\n")
        for rule in cat_rules:
            f.write(f"{rule[0]},{rule[1]},{rule[2]}\n")
    
    # ---- Incompatibilities ----
    n_incompat = int(n_items * incompatibility_ratio * (n_items - 1) / 2)
    n_incompat = min(n_incompat, n_items * 3)  # Cap
    incompatibilities = set()
    while len(incompatibilities) < n_incompat:
        a = random.randint(0, n_items - 1)
        b = random.randint(0, n_items - 1)
        if a != b:
            pair = (min(a, b), max(a, b))
            if pair not in incompatibilities:
                incompatibilities.add(pair)
    
    with open(os.path.join(output_dir, "incompatibilities.csv"), "w") as f:
        f.write("id_item_a,id_item_b\n")
        for pair in sorted(incompatibilities):
            f.write(f"{pair[0]},{pair[1]}\n")
    
    # ---- Dependencies ----
    n_deps = int(n_items * dependency_ratio)
    dependencies = set()
    while len(dependencies) < n_deps:
        item = random.randint(0, n_items - 1)
        req = random.randint(0, n_items - 1)
        if item != req and (item, req) not in dependencies:
            dependencies.add((item, req))
    
    with open(os.path.join(output_dir, "dependencies.csv"), "w") as f:
        f.write("id_item,id_requerido\n")
        for dep in sorted(dependencies):
            f.write(f"{dep[0]},{dep[1]}\n")
    
    print(f"  -> {n_items} ítems, {len(cat_rules)} reglas, "
          f"{len(incompatibilities)} incompatibilidades, {len(dependencies)} dependencias")
    print(f"  -> Guardado en {output_dir}/")
    
    # Summary
    total_peso = sum(item[2] for item in items)
    total_volumen = sum(item[3] for item in items)
    total_valor = sum(item[1] for item in items)
    print(f"  -> Suma pesos: {total_peso}, Suma volúmenes: {total_volumen}, Suma valores: {total_valor}")
    print(f"  -> W=40%: {int(0.4*total_peso)}, V=40%: {int(0.4*total_volumen)}")


def main():
    parser = argparse.ArgumentParser(description="Generador de instancias para mochila extendida")
    parser.add_argument("--seed", type=int, default=42, help="Semilla RNG")
    parser.add_argument("--data-dir", type=str, default="data", help="Directorio base de datos")
    args = parser.parse_args()
    
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.data_dir)
    
    # Small: 100 items
    generate_instance(
        n_items=100, n_categories=5,
        incompatibility_ratio=0.02, dependency_ratio=0.03,
        weight_range=(1, 50), volume_range=(1, 30), value_range=(5, 100),
        seed=args.seed, output_dir=os.path.join(base_dir, "small")
    )
    
    print()
    
    # Medium: 1000 items
    generate_instance(
        n_items=1000, n_categories=8,
        incompatibility_ratio=0.0002, dependency_ratio=0.002,
        weight_range=(1, 100), volume_range=(1, 80), value_range=(10, 500),
        seed=args.seed + 1, output_dir=os.path.join(base_dir, "medium")
    )
    
    print()
    
    # Large: 10000 items
    generate_instance(
        n_items=10000, n_categories=12,
        incompatibility_ratio=0.00001, dependency_ratio=0.0003,
        weight_range=(1, 200), volume_range=(1, 150), value_range=(5, 1000),
        seed=args.seed + 2, output_dir=os.path.join(base_dir, "large")
    )
    
    print("\nTodas las instancias generadas.")


if __name__ == "__main__":
    main()
