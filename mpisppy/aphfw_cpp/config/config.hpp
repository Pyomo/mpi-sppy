#pragma once
#include "CLI11.hpp"

#include <string>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <variant>
#include <optional>
#include <type_traits>
#include <stdexcept>
#include <sstream>

class Config {
private:
  CLI::App* app;

  using OptionType = std::variant<std::string, double, int, bool,
                                  std::vector<std::string>, std::vector<double>>;

  // Current values (CLI-parsed or programmatically set). CLI-registered entries start as T{}.
  std::map<std::string, OptionType> optionValues;

  // Declared defaults for CLI options.
  std::map<std::string, OptionType> optionDefaults;

  // Track CLI-registered options + their handles for count().
  std::set<std::string> cliRegisteredOptions;
  std::map<std::string, CLI::Option*> optionHandles;

  // Names that were programmatically overridden via setOption (for CLI options).
  std::set<std::string> programmaticOverrides;

  // Normalize a flag/option name: "r" -> "-r", "rho" -> "--rho", "-x"/"--x" unchanged
  static std::string normalize_opt_name(const std::string& s) {
    if (s.empty()) return "";
    if (s.rfind("--", 0) == 0 || s.rfind("-", 0) == 0) return s;
    return (s.size() == 1 ? "-" : "--") + s;
  }

  // Build the CLI11 name spec, e.g. "-r,--rho" or "--rho" if no alias
  static std::string build_opt_spec(const std::string& primary, const std::string& alias) {
    const std::string p = normalize_opt_name(primary);
    const std::string a = normalize_opt_name(alias);
    if (a.empty() || a == p) return p;
    // Put short alias first if provided (nice in help), then the long name
    return a + "," + p;
  }

  // Helper to stringify defaults (used in description)
  template <typename U>
  static std::string defaultToString(const U& v) {
    if constexpr (std::is_same_v<U, std::string>) {
      return "\"" + v + "\"";
    } else if constexpr (std::is_same_v<U, bool>) {
      return v ? "true" : "false";
    } else if constexpr (std::is_same_v<U, std::vector<std::string>>) {
      std::ostringstream oss;
      oss << "[";
      for (size_t i = 0; i < v.size(); ++i) { if (i) oss << ", "; oss << '"' << v[i] << '"'; }
      oss << "]";
      return oss.str();
    } else if constexpr (std::is_same_v<U, std::vector<double>>) {
      std::ostringstream oss;
      oss << "[";
      for (size_t i = 0; i < v.size(); ++i) { if (i) oss << ", "; oss << v[i]; }
      oss << "]";
      return oss.str();
    } else {
      std::ostringstream oss; oss << v; return oss.str(); // int, double, etc.
    }
  }

public:
  explicit Config(CLI::App& cliApp) : app(&cliApp) {}

  // Register a CLI11 option:
  // - Bind to zero-initialized storage (T{}), not the default.
  // - Keep the default separately in optionDefaults.
  // - Store the CLI::Option* so we can check count() in getOption().
  template <typename T>
  void addOption(const std::string& name,
		 T default_value,
		 const std::string& description,
		 const std::string& alias = "") {
    optionDefaults[name] = default_value;
    optionValues[name]   = T{};                // do NOT store default here
    cliRegisteredOptions.insert(name);

    const std::string desc_with_default =
      description + " (default " + defaultToString<T>(default_value) + ")";

    const std::string spec = build_opt_spec(name, alias);

    CLI::Option* handle = nullptr;
    if constexpr (std::is_same_v<T, std::string>) {
      handle = app->add_option(spec, std::get<std::string>(optionValues[name]), desc_with_default);
    } else if constexpr (std::is_same_v<T, double>) {
      handle = app->add_option(spec, std::get<double>(optionValues[name]), desc_with_default)
	->check(CLI::PositiveNumber);
    } else if constexpr (std::is_same_v<T, int>) {
      handle = app->add_option(spec, std::get<int>(optionValues[name]), desc_with_default);
    } else if constexpr (std::is_same_v<T, bool>) {
      handle = app->add_flag(spec, std::get<bool>(optionValues[name]), desc_with_default);
    } else if constexpr (std::is_same_v<T, std::vector<std::string>>) {
      handle = app->add_option(spec, std::get<std::vector<std::string>>(optionValues[name]),
			       desc_with_default)->expected(-1);
    } else if constexpr (std::is_same_v<T, std::vector<double>>) {
      handle = app->add_option(spec, std::get<std::vector<double>>(optionValues[name]),
			       desc_with_default)->expected(-1)->check(CLI::PositiveNumber);
    } else {
      static_assert(!sizeof(T), "Unsupported option type for addOption.");
    }

    optionHandles[name] = handle;
  }

  
  // Internal (non-CLI) options still go straight into optionValues
  template <typename T>
  void addInternalOption(const std::string& name, T value) {
    optionValues[name] = value;
  }

  
  // Retrieve option with precedence:
  //   CLI option: (user-specified) -> value
  //               else if programmatically set -> programmatic value
  //               else -> default (if available)
  //   Non-CLI option: stored value (if available)
  template <typename T>
  std::optional<T> getOption(const std::string& name) const {
    const bool is_cli = (cliRegisteredOptions.find(name) != cliRegisteredOptions.end());

    if (is_cli) {
      // 1) Did user supply it on CLI?
      if (auto h = optionHandles.find(name); h != optionHandles.end() && h->second) {
        if (h->second->count() > 0) {
          if (auto it = optionValues.find(name); it != optionValues.end()) {
            if (auto v = std::get_if<T>(&it->second)) return *v;
          }
          return std::nullopt; // type mismatch (registration inconsistency)
        }
      }
      // 2) Programmatic override (when CLI did not supply)
      if (programmaticOverrides.count(name)) {
        if (auto it = optionValues.find(name); it != optionValues.end()) {
          if (auto v = std::get_if<T>(&it->second)) return *v;
        }
        return std::nullopt;
      }
      // 3) Declared default
      if (auto dit = optionDefaults.find(name); dit != optionDefaults.end()) {
        if (auto dv = std::get_if<T>(&dit->second)) return *dv;
      }
      return std::nullopt;
    }

    // Non-CLI/internal
    if (auto it = optionValues.find(name); it != optionValues.end()) {
      if (auto v = std::get_if<T>(&it->second)) return *v;
    }
    return std::nullopt;
  }

  // get a required string option assuming default is ""
  std::string getRequiredString(std::string oName) {
    auto inVal = getOption<std::string>(oName);
    if (!inVal) throw std::runtime_error("The cfg object must supply " + oName);
    if (inVal->empty()) throw std::runtime_error("You must supply a string for " + oName);
    return *inVal;
  }

  // Register a non-negative double option (allows 0).
  // Same semantics as addOption: value storage starts at T{} (0.0), default stored separately.
  template <typename T>
  void addOptionNonNegative(const std::string& name,
                            T default_value,
                            const std::string& description,
                            const std::string& alias = "") {
    static_assert(std::is_same_v<T, double>, "addOptionNonNegative currently supports only double.");

    optionDefaults[name] = default_value;
    optionValues[name]   = T{};                // do NOT store default here
    cliRegisteredOptions.insert(name);

    const std::string desc_with_default =
      description + " (default " + defaultToString<T>(default_value) + ")";

    const std::string spec = build_opt_spec(name, alias);

    // IMPORTANT: allow 0 by using NonNegativeNumber (NOT PositiveNumber).
    CLI::Option* handle =
      app->add_option(spec, std::get<double>(optionValues[name]), desc_with_default)
         ->check(CLI::NonNegativeNumber);

    optionHandles[name] = handle;
  }

  template <typename T>
  T getOptionValue(const std::string& name) const {
    if (auto v = getOption<T>(name)) {
      return *v;  // effective value (CLI/programmatic/default) (* is overloaded)
    }

    // No effective value available:
    // - For CLI-registered options this would typically mean a type mismatch
    //   (since addOption always provides a default of the right type).
    // - For non-CLI options it means nothing was set internally.
    const bool is_cli = (cliRegisteredOptions.find(name) != cliRegisteredOptions.end());
    if (is_cli) {
      throw std::runtime_error(
			       "getOptionValue<requested T>: option \"" + name +
			       "\" is a CLI option but no value of the requested type is available. "
			       "Check that the template type matches the type used in addOption.");
    } else {
      throw std::runtime_error(
			       "getOptionValue<requested T>: internal option \"" + name +
			       "\" is not set.");
    }
  }

  
  // Delete an option by its canonical name (the `name` you passed to addOption).
  // This removes the CLI11 option (and any alias bound to it) and clears
  // internal/default/override state. Returns true if something was removed.
  bool deleteOption(const std::string& name) {
    bool touched = false;

    // If this was a CLI-registered option, remove it from CLI11 first.
    if (auto h = optionHandles.find(name); h != optionHandles.end()) {
      if (h->second != nullptr) {
	// CLI11 App has remove_option(Option*)
	try {
	  app->remove_option(h->second);
	} catch (...) {
	  // Ignore removal errors; we'll still clean local state.
	}
      }
      optionHandles.erase(h);
      touched = true;
    }

    // Remove from internal state
    touched = (optionValues.erase(name) > 0) || touched;
    touched = (optionDefaults.erase(name) > 0) || touched;
    touched = (programmaticOverrides.erase(name) > 0) || touched;
    touched = (cliRegisteredOptions.erase(name) > 0) || touched;

    return touched;
  }

 
  bool explicitlyGiven(const std::string& optionName) const {
    auto it = optionHandles.find(optionName);
    if (it == optionHandles.end() || it->second == nullptr) return false;  // not a CLI option
    return it->second->count() > 0;  // >=1 appearances means explicitly given
  }


  // allow programmitic option setting
  template <typename T>
  void setOption(const std::string& name, T value) {
    optionValues[name] = value;
    // Mark this as an override so getOption prefers it over the default
    // even if the user didn't provide it on the CLI.
    if (cliRegisteredOptions.count(name)) {
      programmaticOverrides.insert(name);
    }
  }
  // Pretty-print all options and their effective values (with source tags) to a stream
  void pprint(std::ostream& out) const {
    out << "Configuration Options (effective values):\n";
    out << std::boolalpha;

    // Build the union of keys from optionValues and optionDefaults
    std::set<std::string> seen;
    std::vector<std::string> keys;
    keys.reserve(optionValues.size() + optionDefaults.size());

    for (const auto& kv : optionValues) {
      if (seen.insert(kv.first).second) keys.push_back(kv.first);
    }
    for (const auto& kv : optionDefaults) {
      if (seen.insert(kv.first).second) keys.push_back(kv.first);
    }

    for (const auto& key : keys) {
      // Determine the variant prototype to deduce type T for getOption<T>(key)
      const OptionType* proto = nullptr;
      if (auto it = optionValues.find(key); it != optionValues.end()) {
	proto = &it->second;
      } else if (auto dit = optionDefaults.find(key); dit != optionDefaults.end()) {
	proto = &dit->second;
      } else {
	continue; // defensive; shouldn't happen
      }

      // Work out the source tag consistent with getOption precedence
      std::string source = "internal";
      if (cliRegisteredOptions.count(key)) {
	if (auto h = optionHandles.find(key); h != optionHandles.end() && h->second) {
	  if (h->second->count() > 0) {
	    source = "cli";
	  } else if (programmaticOverrides.count(key)) {
	    source = "override";
	  } else if (optionDefaults.count(key)) {
	    source = "default";
	  } else {
	    source = "cli"; // defensive
	  }
	}
      }

      // Print the effective value for this key
      std::visit([&](const auto& val_tag) {
	using T = std::decay_t<decltype(val_tag)>;
	auto eff = getOption<T>(key);

	out << "  " << key << ": ";
	if (!eff) {
	  out << "<unset>";
	} else {
	  if constexpr (std::is_same_v<T, std::vector<std::string>>) {
	    out << "[";
	    for (size_t i = 0; i < eff->size(); ++i) {
	      if (i) out << ", ";
	      out << (*eff)[i];
	    }
	    out << "]";
	  } else if constexpr (std::is_same_v<T, std::vector<double>>) {
	    out << "[";
	    for (size_t i = 0; i < eff->size(); ++i) {
	      if (i) out << ", ";
	      out << (*eff)[i];
	    }
	    out << "]";
	  } else {
	    out << *eff;
	  }
	}
	out << " (" << source << ")\n";
      }, *proto);
    }
  }
};



