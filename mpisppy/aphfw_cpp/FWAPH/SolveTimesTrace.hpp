// src/FWAPH/SolveTimesTrace.hpp
#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>

struct SolveTimesRecord {
    double      wall_time;
    int         dispatch_cycle;
    std::string scenario;
    double      fw_total_time;
    double      nl_time;
    double      milp_time;
    int         milp_solves;
};

class SolveTimesTrace {
public:
    explicit SolveTimesTrace(const std::string& filename)
        : filename_(filename)
    {}

    SolveTimesTrace(const SolveTimesTrace&) = delete;
    SolveTimesTrace& operator=(const SolveTimesTrace&) = delete;
    SolveTimesTrace(SolveTimesTrace&&) = default;
    SolveTimesTrace& operator=(SolveTimesTrace&&) = default;

    void append(double wall_time, int dispatch_cycle, const std::string& scenario,
                double fw_total_time, double nl_time, double milp_time, int milp_solves)
    {
        records_.push_back({wall_time, dispatch_cycle, scenario,
                            fw_total_time, nl_time, milp_time, milp_solves});
    }

    void save() const
    {
        std::ofstream ofs(filename_, std::ios::out | std::ios::trunc);
        if (!ofs) {
            std::cerr << "SolveTimesTrace::save: Unable to open file: " << filename_ << "\n";
            return;
        }
        ofs << "wall_time,dispatch_cycle,scenario,fw_total_time,nl_time,milp_time,milp_solves\n";
        for (const auto& r : records_) {
            ofs << r.wall_time      << ","
                << r.dispatch_cycle << ","
                << r.scenario       << ","
                << r.fw_total_time  << ","
                << r.nl_time        << ","
                << r.milp_time      << ","
                << r.milp_solves    << "\n";
        }
        if (!ofs) {
            std::cerr << "SolveTimesTrace::save: Write error for file: " << filename_ << "\n";
        }
    }

    ~SolveTimesTrace() = default;

private:
    std::string filename_;
    std::vector<SolveTimesRecord> records_;
};
