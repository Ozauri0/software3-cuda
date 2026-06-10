// instance_loader.hpp - Carga de instancias CSV
#ifndef INSTANCE_LOADER_HPP
#define INSTANCE_LOADER_HPP

#include "types.hpp"
#include <string>

// Carga una instancia completa desde un directorio
// El directorio debe contener: items.csv, category_rules.csv,
// incompatibilities.csv, dependencies.csv
Instance loadInstance(const std::string& data_dir);

// Carga solo items
std::vector<Item> loadItems(const std::string& filepath);

// Carga reglas de categoría
std::vector<CategoryRule> loadCategoryRules(const std::string& filepath);

// Carga incompatibilidades
std::vector<Incompatibility> loadIncompatibilities(const std::string& filepath);

// Carga dependencias
std::vector<Dependency> loadDependencies(const std::string& filepath);

// Calcula capacidades W y V como porcentaje del total
void calculateCapacities(Instance& inst, float weight_pct = 0.40f, float volume_pct = 0.40f);

#endif // INSTANCE_LOADER_HPP
