///////////////////////////////////////////////////////////////////////////////
// mpi-sppy: MPI-based Stochastic Programming in PYthon
//
// Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
// Sustainable Energy, LLC, The Regents of the University of California, et al.
// All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
// full copyright and license information.
///////////////////////////////////////////////////////////////////////////////
// originally contributed by Jonathan Eckstein

#pragma once


#include <string>
#include <numeric>
#include <vector>
#include <list>
#include <set>
#include <map>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cassert>
#include <csignal>   // std::raise, SIGTRAP / SIGABRT
#include <cstdlib>   // std::abort (fallback)

//#include "mpi.h"
#include <mpi.h>

// Macro for debug printouts,  Will compile out to nothing if -DNODEBUGPRINT
// is specified in the compile options

#ifdef NODEBUGPRINT
#define DEBUGPR(LEVEL,STUFF)
#else
#define DEBUGPR(LEVEL,STUFF) if (verbosity >= LEVEL) std::cout << STUFF << std::endl
#endif


// Tiny debugging base class, which is intended to be one of the base classes for almost every class

class debugLevels
{
public:

  debugLevels() :
  verbosity(0)
  { };

  debugLevels(int level) :
  verbosity(level)
  { };

    // Set verbosity level

  void setVerbosity(int level)
  {
    verbosity = level;
  }

  ~debugLevels() { };

  protected :

  int verbosity;    
};


// Base class for objects that include an MPI communicator
// Also builds in the little debug class

class commBased :
virtual public debugLevels
{
public:

  // Constructor -- trying having only one, no-argument constructor

  commBased() :
  comm(MPI_COMM_NULL),
  commSize(0),
  myRank(MPI_PROC_NULL),
  rankTag("<Uninitialized rank tag> ")
  { };


  // To be called once you do know the communicator, or to change communicators

  void setupComm(MPI_Comm comm_) 
  {
    comm = comm_;
    MPI_Comm_size(comm,&commSize);
    MPI_Comm_rank(comm,&myRank);    
    rankTag  = '[';
    rankTag += std::to_string(myRank);
    rankTag += "] ";
  };


  // A templated routine for standard parallel performance statistic printing

  template<typename T> T sumRunStatPrint(T value, 
  const char*  explanation,
  MPI_Datatype dtype,
  int          width, 
  int          precision = 0)
    {
      T globalSum = -1000;
      MPI_Reduce(&value, &globalSum, 1, dtype, MPI_SUM, 0, comm);
      if (myRank == 0)
        std::cout << std::setw(width) << globalSum << ' ' << explanation << '\n';

      return globalSum;
    };


  // A templated routine to just print some run statistic on rank 0

  template<typename T> void printStatRank0(T value, 
  const char* explanation, 
  int precision,
  int width)
  {
    if (myRank == 0)
    {
      std::cout.precision(1);
      std::cout << std::fixed;    
      std::cout << std::setw(width) << value << ' ' 
                << explanation << "s\n";
    }
  }

  // Query routines

  int getRank(void) { return myRank; };

  int getCommSize(void) { return commSize; };


protected:

    MPI_Comm    comm;
    int         commSize;
    int         myRank;
    std::string rankTag;

};


// To reduce clutter when checking set membership

template<typename T> bool isInSet(T element, std::set<T> S)
  {
    return S.find(element) != S.end();
  }


// Common code for printing the contents of a container

template<typename C> std::string 
  containerToString(C& container,
                    const char* openDelim, 
                    const char* closeDelim)
  {
    std::string toReturn = openDelim;
    auto it = container.begin();
    int numElements = container.size();
    for (int i = 0; i < numElements - 1; i++)
    {
      toReturn += std::to_string(*(it++));
      toReturn += ", ";
    }
    if (numElements > 0)
      toReturn += std::to_string(*it);
    toReturn += closeDelim;
    return toReturn;
  }


// Compute the squared norm of a vector (in the interest of clutter reduction)
template<typename T>
double squaredNorm(const std::vector<T>& vec) {
    return std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
}


// Print a set

template<typename T> std::string setToString(std::set<T>& S)
  {
    return containerToString(S,"{","}");
  }


// Print a list

template<typename T> std::string listToString(std::list<T>& L)
  {
    return containerToString(L,"[","]");
  }


// MPI utilities added by DLW (and chatgpt)


static inline void die_mpi(const char* where, int err) {
    char msg[MPI_MAX_ERROR_STRING]; int len=0;
    MPI_Error_string(err, msg, &len);
    std::cerr << where << ": MPI error " << err << " : " << std::string(msg, len) << "\n";
    MPI_Abort(MPI_COMM_WORLD, err);
}


static inline void assert_valid_comm(MPI_Comm comm, const char* where) {
    if (comm == MPI_COMM_NULL) {
        std::cerr << where << ": MPI_COMM_NULL\n";
        MPI_Abort(MPI_COMM_WORLD, MPI_ERR_COMM);
    }
    // Try a cheap query to validate the handle
    int r=-1, s=-1;
    int rc = MPI_Comm_rank(comm, &r); if (rc != MPI_SUCCESS) die_mpi(where, rc);
    rc = MPI_Comm_size(comm, &s);     if (rc != MPI_SUCCESS) die_mpi(where, rc);
    std::cerr << where << ": comm(f)=" << MPI_Comm_c2f(comm) << " rank=" << r << " size=" << s << "\n";
}


inline void print_comm_info(const char* where, MPI_Comm comm) {
    int wRank=-1, wSize=-1;
    MPI_Comm_rank(MPI_COMM_WORLD, &wRank);
    MPI_Comm_size(MPI_COMM_WORLD, &wSize);

    if (comm == MPI_COMM_NULL) {
        std::cerr << where << ": COMM=NULL  (world rank " << wRank
                  << "/" << wSize << ")\n";
        return;
    }

    char name[MPI_MAX_OBJECT_NAME]; int len=0;
    name[0] = '\0';
    MPI_Comm_get_name(comm, name, &len);

    int r=-1, s=-1;
    int rc1 = MPI_Comm_rank(comm, &r);
    int rc2 = MPI_Comm_size(comm, &s);

    std::cerr << where
              << ": comm=" << MPI_Comm_c2f(comm)
              << " name='" << std::string(name, len) << "'"
              << " rank=" << (rc1==MPI_SUCCESS ? r : -1)
              << " size=" << (rc2==MPI_SUCCESS ? s : -1)
              << " (world " << wRank << "/" << wSize << ")\n";
}

inline bool ensure_comm_ok(const char* where, MPI_Comm comm) {
    if (comm == MPI_COMM_NULL) {
        print_comm_info(where, comm);
        // Pause here for gdb instead of aborting the whole job:
        std::raise(SIGTRAP);   // or: return false;
        return false;
    }
    return true;
}
