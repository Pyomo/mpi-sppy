///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// utils/timer.hpp
#pragma once

#include <string_view>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <sstream>

// ---------------------------
// TicToc: simple wall timer
// ---------------------------
class TicToc {
public:
    using clock = std::chrono::steady_clock;

    TicToc()
        : start_(clock::now())
    {}

    // reset the timer
    void tic() {
        start_ = clock::now();
    }

    // seconds since last tic()
    double toc() const {
        auto now = clock::now();
        std::chrono::duration<double> d = now - start_;
        return d.count();
    }

private:
    clock::time_point start_;
};

// ---------------------------
// GlobalTimer
// ---------------------------
class GlobalTimer {
public:
  GlobalTimer(int rank = -1) : rank_(rank) { timer_.tic(); }

  void global_toc(std::string_view msg) const {
    double elapsed = timer_.toc();

    // buffer to avoid a split across ranks
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    
    if (rank_ == -1) {
        ss << "[" << elapsed << "s] " << msg << "\n";
    } else {
        ss << "[r" << rank_ << "; " << elapsed << "s] " << msg << "\n";
    }

    std::cout << ss.str() << std::flush;
  }

  // seconds since this timer was constructed (or last tic)
  double seconds() const {
    return timer_.toc();
  }

private:
  TicToc timer_;
  int rank_ = -1;
};

// Global timer hooks (implemented in timer.cpp)
void install_global_timer(const GlobalTimer* t);
void global_toc(std::string_view msg);
double global_seconds();
