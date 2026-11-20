///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// src/FWAPH/WTrace.hpp
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <mutex>
#include <iostream>

#include "json.hpp" // nlohmann::json

using json = nlohmann::json;

struct WRecord {
    double time;                // wall-clock time since run start
    std::string scenario;       // scenario name (e.g. SCEN0)
    std::vector<double> W;      // W vector at this time
};

class WTrace {
public:
    // Constructor: filename where this rank's trace will be written
    explicit WTrace(const std::string& filename)
      : filename_(filename)
    {}

    // Non-copyable, movable
    WTrace(const WTrace&) = delete;
    WTrace& operator=(const WTrace&) = delete;
    WTrace(WTrace&&) = default;
    WTrace& operator=(WTrace&&) = default;

    // Append one snapshot of W for a given scenario at given time
    void append(double time, const std::string& scenarioName, const std::vector<double>& W)
    {
        WRecord rec;
        rec.time = time;
        rec.scenario = scenarioName;
        rec.W = W;

        std::lock_guard<std::mutex> lock(mutex_);
        records_.push_back(std::move(rec));
    }

    // Save all records to JSON (overwrites filename_)
    void save() const
    {
        // Build JSON array
        json j = json::array();

        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& rec : records_) {
                json entry;
                entry["time"] = rec.time;
                entry["scenario"] = rec.scenario;
                entry["W"] = rec.W;
                j.push_back(std::move(entry));
            }
        }

        std::ofstream ofs(filename_, std::ios::out | std::ios::trunc);
        if (!ofs) {
            std::cerr << "WTrace::save: Unable to open file for writing: " << filename_ << "\n";
            return;
        }

        // Pretty-print with 2-space indent
        ofs << j.dump(2) << "\n";
        if (!ofs) {
            std::cerr << "WTrace::save: Write error for file: " << filename_ << "\n";
        }
    }

  // Destructor
  ~WTrace() = default;

private:
    std::string filename_;
    mutable std::mutex mutex_;
    std::vector<WRecord> records_;
};
