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
