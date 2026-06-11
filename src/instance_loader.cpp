// instance_loader.cpp - Implementacion del cargador de instancias
#include "instance_loader.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

// ============================================================================
// Parse CSV helper
// ============================================================================

static std::vector<std::string> splitCSV(const std::string& line, char delimiter = ',') {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        tokens.push_back(token);
    }
    return tokens;
}

// ============================================================================
// Load Items
// ============================================================================

std::vector<Item> loadItems(const std::string& filepath) {
    std::vector<Item> items;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        fprintf(stderr, "Error: No se pudo abrir %s\n", filepath.c_str());
        return items;
    }
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto tokens = splitCSV(line);
        if (tokens.size() < 5) continue;
        
        Item item;
        item.id = std::stoi(tokens[0]);
        item.valor = std::stoi(tokens[1]);
        item.peso = std::stoi(tokens[2]);
        item.volumen = std::stoi(tokens[3]);
        item.categoria = std::stoi(tokens[4]);
        items.push_back(item);
    }
    
    printf("  Cargados %zu items desde %s\n", items.size(), filepath.c_str());
    return items;
}

// ============================================================================
// Load Category Rules
// ============================================================================

std::vector<CategoryRule> loadCategoryRules(const std::string& filepath) {
    std::vector<CategoryRule> rules;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        printf("  Advertencia: No se encontro %s (sin reglas de categoria)\n", filepath.c_str());
        return rules;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto tokens = splitCSV(line);
        if (tokens.size() < 3) continue;
        
        CategoryRule rule;
        rule.categoria = std::stoi(tokens[0]);
        rule.minimo = std::stoi(tokens[1]);
        rule.maximo = std::stoi(tokens[2]);
        rules.push_back(rule);
    }
    
    printf("  Cargadas %zu reglas de categoria\n", rules.size());
    return rules;
}

// ============================================================================
// Load Incompatibilities
// ============================================================================

std::vector<Incompatibility> loadIncompatibilities(const std::string& filepath) {
    std::vector<Incompatibility> incompat;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        printf("  Advertencia: No se encontro %s (sin incompatibilidades)\n", filepath.c_str());
        return incompat;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto tokens = splitCSV(line);
        if (tokens.size() < 2) continue;
        
        Incompatibility incomp;
        incomp.id_item_a = std::stoi(tokens[0]);
        incomp.id_item_b = std::stoi(tokens[1]);
        incompat.push_back(incomp);
    }
    
    printf("  Cargadas %zu incompatibilidades\n", incompat.size());
    return incompat;
}

// ============================================================================
// Load Dependencies
// ============================================================================

std::vector<Dependency> loadDependencies(const std::string& filepath) {
    std::vector<Dependency> deps;
    std::ifstream file(filepath);
    
    if (!file.is_open()) {
        printf("  Advertencia: No se encontro %s (sin dependencias)\n", filepath.c_str());
        return deps;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto tokens = splitCSV(line);
        if (tokens.size() < 2) continue;
        
        Dependency dep;
        dep.id_item = std::stoi(tokens[0]);
        dep.id_requerido = std::stoi(tokens[1]);
        deps.push_back(dep);
    }
    
    printf("  Cargadas %zu dependencias\n", deps.size());
    return deps;
}

// ============================================================================
// Calculate Capacities
// ============================================================================

void calculateCapacities(Instance& inst, float weight_pct, float volume_pct) {
    inst.suma_total_pesos = 0;
    inst.suma_total_volumenes = 0;
    
    for (const auto& item : inst.items) {
        inst.suma_total_pesos += item.peso;
        inst.suma_total_volumenes += item.volumen;
    }
    
    inst.capacidad_peso = (int)(weight_pct * inst.suma_total_pesos);
    inst.capacidad_volumen = (int)(volume_pct * inst.suma_total_volumenes);
    
    printf("  Capacidades calculadas: peso=%d (%.0f%%), volumen=%d (%.0f%%)\n",
           inst.capacidad_peso, weight_pct * 100.0f,
           inst.capacidad_volumen, volume_pct * 100.0f);
}

// ============================================================================
// Load Full Instance
// ============================================================================

Instance loadInstance(const std::string& data_dir) {
    Instance inst;
    
    printf("Cargando instancia desde %s:\n", data_dir.c_str());
    
    inst.items = loadItems(data_dir + "/items.csv");
    inst.n_items = (int)inst.items.size();
    
    inst.category_rules = loadCategoryRules(data_dir + "/category_rules.csv");
    inst.incompatibilities = loadIncompatibilities(data_dir + "/incompatibilities.csv");
    inst.dependencies = loadDependencies(data_dir + "/dependencies.csv");
    
    calculateCapacities(inst);
    
    return inst;
}
