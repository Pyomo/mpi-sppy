#ifndef CONFIG_HPP
#define CONFIG_HPP

/*
To get started (and maybe forever) we are going to get parameters, such
as solver parameters, from json files
To get started type checking will be a little lax.
Note: we will take anything in the json file, but we can check for things.
 */

#include <iostream>
#include <fstream>
#include <map>
#include <variant>
#include <string>
#include <vector>
#include <set>
#include "json.hpp"

// Using nlohmann::json
using json = nlohmann::json;

// Define the variant type to hold multiple types
using VariantType = std::variant<int, double, bool,
                                 std::string, std::vector<std::string>>;

// Map for parameter dictionaries
using ParamMapType = std::map<std::string, VariantType>;

class ParamsBase {
public:
  ParamsBase(const std::string jsonFileName);
  ~ParamsBase() { };
  std::string const getJsonFileName() { return jsonFileName; }
  virtual ParamMapType& getParamMap() { return paramMap; }
  void checkFor(const std::set<std::string> mustHave, const std::string msg);
  void extrasWarn(const std::set<std::string> canHave,
                  const std::string msg);
  void printParams();
protected:
  std::string jsonFileName;
  ParamMapType paramMap;
};

#endif  // CONFIG_HPP
