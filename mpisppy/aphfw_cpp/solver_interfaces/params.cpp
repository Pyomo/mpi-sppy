///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
#include "params.hpp"

// Function to convert JSON to VariantType
// (I got it from chatgpt, and I don't understand j, but it works)
VariantType jsonToVariant(const json& j) {
    if (j.is_string()) {
        return j.get<std::string>();
    } else if (j.is_number_integer()) {
        return j.get<int>();
    } else if (j.is_number_float()) {
        return j.get<double>();
    } else if (j.is_boolean()) {
        return j.get<bool>();
    } else if (j.is_array()) {
        std::vector<std::string> vec;
        for (const auto& item : j) {
            if (item.is_string()) {
                vec.push_back(item.get<std::string>());
            } else {
                throw std::runtime_error("Only string arrays are supported");
            }
        }
        return vec;
    }
    throw std::runtime_error("Unsupported JSON type");
}

// Function to convert JSON to map (from chatgpt, but it makes sense)
std::map<std::string, VariantType> jsonToMap(const json& j) {
    std::map<std::string, VariantType> result;
    for (auto& [key, value] : j.items()) {
        result[key] = jsonToVariant(value);
    }
    return result;
}

ParamsBase::ParamsBase(const std::string jsonFileName):
  jsonFileName(jsonFileName)
{
  // Open the JSON file
  std::ifstream file(jsonFileName);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: "+jsonFileName);
  }
  json jsonData;
  file >> jsonData;

  for (auto& [key, value] : jsonData.items()) {
    paramMap[key] = jsonToVariant(value);
  }
}

void ParamsBase::printParams() {
  // simple apart from the need to handle a list
  for (const auto& [key, value] : paramMap) {
    std::cout << key << ": ";
    std::visit([](const auto& v) {
      using T = std::decay_t<decltype(v)>;
      if constexpr (std::is_same_v<T, std::vector<std::string>>) {
        std::cout << "[ ";
        for (const auto& item : v) {
          std::cout << item << " ";
        }
        std::cout << "]";
      } else {
        std::cout << v;
      }
      std::cout << std::endl;
    }, value);
  }
}

void ParamsBase::checkFor(const std::set<std::string> mustHave,
                          const std::string msg) {
  // throw an error if something is missing
  std::vector<std::string> missing;
  std::set<std::string> mapKeys;
  for (const auto& [key, _] : paramMap) {
    mapKeys.insert(key);
  }
  for (const std::string& key : mustHave) {
    if (mapKeys.find(key) == mapKeys.end()) {
      missing.push_back(key);
    }
  }
  if (missing.size() > 0) {
    std::cerr << "Missing keys list: " << std::endl;
    for (std::string m : missing) {
      std::cerr << "   " << m << std::endl;
    }
    throw std::runtime_error(msg + " Parameters shown above are missing");
  }
}
    
void ParamsBase::extrasWarn(const std::set<std::string> canHave,
                          const std::string msg) {
  std::set<std::string> vectorSet(canHave.begin(), canHave.end());  
  std::vector<std::string> extras;

  for (const auto& [key, _] : paramMap) {
    if (vectorSet.find(key) == vectorSet.end()) {
      extras.push_back(key);
    }
  }
  if (extras.size() > 0) {
    std::cerr << "Warning: " << msg << "the following parameters are extra:"
                          << std::endl;
    for (std::string x : extras) {
      std::cerr << "   " << x << std::endl;
    }
  }
}
