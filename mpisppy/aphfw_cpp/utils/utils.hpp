#pragma once

//--------- Utilities --------
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <filesystem>
#include <vector>
#include <string>
#include <iomanip>

namespace fs = std::filesystem;

inline std::size_t countLPFiles(const std::string& directoryPath) {
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        throw std::runtime_error("Directory does not exist: " + directoryPath);
    }

    std::size_t count = 0;
    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (entry.is_regular_file() && entry.path().extension() == ".lp") {
            ++count;
        }
    }
    return count;
}


inline std::vector<std::string> getLPFilesWithoutExtension(const std::string& directoryPath) {
    std::vector<std::string> lpFileNames;
    
    fs::path dirPath(directoryPath);
    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        throw std::runtime_error("Directory does not exist: " + directoryPath);
    }

    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            if (entry.path().extension() == ".lp") {
                fs::path p = entry.path();
                p.replace_extension();  // remove the extension
                lpFileNames.push_back(p.string());
            }
        }
    }
    return lpFileNames;
}

inline void ensure_fresh_output_dir(const std::string& csvOutputDirName) {
    fs::path p(csvOutputDirName);

    std::error_code ec;
    bool created = fs::create_directories(p, ec);

    if (ec) {
        // Creation failed for some reason (permissions, path component is a file, etc.)
        throw std::runtime_error("Failed to create output directory '" + p.string() +
                                 "': " + ec.message());
    }
    if (!created) {
        // Path already exists (directory or file)
        throw std::runtime_error("Output directory already exists: '" + p.string() + "'");
    }
}

template<typename T>
inline void getConfigOptionOrThrow(const std::string& optionName,
                                 T& outputVar,
                                 const auto& cfg) {
    if (auto inVal = cfg.template getOption<T>(optionName)) {
        outputVar = *inVal;
    } else {
        throw std::runtime_error("The cfg object must supply " + optionName);
    }
}

// for code in the FWAPH directory, use the macro om utilities.hpp
inline bool _vb(const auto& cfg, int level=1) {
  // tbd: only output on rank 0
  return (cfg.template getOption<int>("verbosity").value_or(0) >= level);
}


// for code in the FWAPH directory, use the macro om utilities.hpp
inline void _vb(const auto& cfg, const std::string& msg, int level=1) {
  if (_vb(cfg, level)) {
    std::cout << msg << std::endl;
  }
}

template <typename T>
void printListMap(const std::map<std::string, std::vector<T>>& myMap) {
    for (const auto& [key, valueList] : myMap) {
        std::cout << "Key: " << key << " => [";
        for (const auto& item : valueList) {
            std::cout << item << ", ";
        }
        if (!valueList.empty()) {
            std::cout << "\b\b"; // Remove the trailing ", " for non-empty lists
        }
        std::cout << "]" << std::endl;
    }
}


// Call to this function should be deleted after use.
template <typename T>
inline void debug(const std::string& msg, const T& it) {
    std::cout << "DEBUG: " << msg << ": " << it << std::endl;
}

// Function to read a CSV file and validate headers
inline std::map<std::string, double> readCsvToMapWithHeaders(
    const std::string& filename,
    const std::string& expectedHeader1,
    const std::string& expectedHeader2) {

    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Could not open file: " + filename);
    }

    std::string line;

    // Read the header line
    std::string header1, header2;
    if (std::getline(file, line)) {
        std::istringstream headerStream(line);
        if (!std::getline(headerStream, header1, ',')\
            || !std::getline(headerStream, header2, ',')) {
            throw std::runtime_error("Error: Invalid header format in file: "\
                                     + filename);
        }

        // Validate headers
        if (header1 != expectedHeader1 || header2 != expectedHeader2) {
            throw std::runtime_error(
                "Error: Header mismatch in file: " + filename +
                ". Expected: '" + expectedHeader1 + "," + expectedHeader2 +
                "' but got: '" + header1 + "," + header2 + "'.");
        }
    } else {
        throw std::runtime_error("Error: Empty file or missing header in file: "\
                                 + filename);
    }

    // Read the rest of the file into a map<string, double>
    std::map<std::string, double> data;
    while (std::getline(file, line)) {
        std::istringstream lineStream(line);
        std::string key;
        double value;

        if (std::getline(lineStream, key, ',') && lineStream >> value) {
            data[key] = value;
        } else {
            std::cerr << "Warning: Skipping invalid line in file: "\
                      << filename << ": " << line << std::endl;
        }
    }

    file.close();
    return data;
}

inline std::vector<double>
vectorFromMapWithOrder(const std::map<std::string, double> iMap,
                       const std::vector<std::string> orderVector) {
  // get the data from iMap and the order from orderVector
  if (iMap.size() != orderVector.size()) {
    std::cerr << iMap.size() << " != " << orderVector.size() << std::endl;
    throw std::invalid_argument("vectorFromMapWithOrder got different sizes ");
  }
  std::vector<double> retval;
  for (std::string i : orderVector) {
    if (!iMap.contains(i)) {
      throw std::invalid_argument("vectorFromMapWithOrder: missing map key: "+i);
    }
    retval.push_back(iMap.at(i));
  }
  return retval;
}

inline void write_vector_to_csv(const std::vector<double>& vec, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i < vec.size() - 1) {
                file << ",";
            }
        }
        file << std::endl;
        file.close();
        std::cout << "Vector successfully written to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

template <typename T>
inline void vecToOpenCSVFile(const T& vec, std::ofstream&file) {
  if (file.is_open()) {       
    for (size_t i = 0; i < vec.size(); ++i) {
      file << vec[i];
      if (i < vec.size() - 1) {
        file << ",";
      }
    }
    file << std::endl;
  } else {
    throw std::invalid_argument("file handle passed to vecToOpenFile is not open");
  }
}

// return true if they have the same values
inline bool sameVectorValues(const std::vector<double>& a,
                             const std::vector<double>& b,
                             double epsilon = 1e-8) {
    if (a.size() != b.size()) {
      std::cerr << "Vector size mismatch: "
                  << a.size() << " vs " << b.size() << std::endl;
      throw std::invalid_argument("vector size mismatch in call to sameVectorValues");
    }

    bool anyDiff = false;
    for (size_t i = 0; i < a.size(); ++i) {
        double diff = std::fabs(a[i] - b[i]);
        if (diff > epsilon) {
            if (!anyDiff) {
                std::cout << "Differences > " << epsilon << ":\n";
                anyDiff = true;
            }
            std::cout << "  idx " << std::setw(4) << i
                      << "  a=" << std::setw(12) << a[i]
                      << "  b=" << std::setw(12) << b[i]
                      << "  diff=" << diff << '\n';
        }
    }

    if (!anyDiff)
        std::cout << "No differences larger than " << epsilon << '\n';
    return !anyDiff;
}


// Utility to print a labeled vector on separate lines
inline void printVector(const std::string& name,
                        const std::vector<double>& v,
                        int precision = 6)
{
    std::cout << name << " (size=" << v.size() << "):" << std::endl;
    std::cout << std::fixed << std::setprecision(precision);
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << "  [" << i << "] " << v[i] << '\n';
    }
}


// Mainly a helper -  writes the vector representation to any ostream
inline void writeVectorInline(std::ostream& os,
                              const std::string& name,
                              const std::vector<double>& v,
                              int precision = 6)
{
    os << name << " = [ ";
    os << std::fixed << std::setprecision(precision);

    for (size_t i = 0; i < v.size(); ++i)
        os << v[i] << (i + 1 < v.size() ? ", " : " ");

    os << "]";
}

// Public version that prints to std::cout
inline void printVectorInline(const std::string& name,
                              const std::vector<double>& v,
                              int precision = 6)
{
    writeVectorInline(std::cout, name, v, precision);
    std::cout << "\n";
}


inline std::string vectorInlineToString(const std::string& name,
                                        const std::vector<double>& v,
                                        int precision = 6)
{
    std::ostringstream oss;
    writeVectorInline(oss, name, v, precision);
    return oss.str();
}


template <typename T>
inline std::vector<T> read_single_row_csv(const std::string& filename) {
    std::ifstream file(filename);
    if (!file) {
        throw std::runtime_error("Could not open csv file: " + filename);
    }

    std::string line;
    if (!std::getline(file, line)) {
        return {}; // empty file
    }

    std::vector<T> values;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        if constexpr (std::is_same_v<T, std::string>) {
            values.push_back(cell);
        } else {
            values.push_back(static_cast<T>(std::stod(cell)));
        }
    }

    return values;
}
