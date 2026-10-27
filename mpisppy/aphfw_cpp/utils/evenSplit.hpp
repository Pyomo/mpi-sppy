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


// Code for breaking up N items into m chunks. Each chunk has int(N/m) or
// int(N/m)+1 items. Adapted from Jonathan's MPISparseMatMul, but with regular
// ints intead of long long ints.


#include <algorithm>

#include "utilities.hpp"   // also includes MPI


class evenSplit
{
public:

        // How many items in my local chunk?

        int myNumItems() { return numLocalItems; };


        // What is the global index of my first ite?

        int myStartPos() { return localStartPos; };


        // Find the chunk position of a particular data item

        int whichChunk(int position)
        {
                if (position < longerChunkItems)
                        return position / (quotient + 1);
                return remainder + (position - longerChunkItems) / quotient;
        }


        // Find the start position of a particular chunk
        int startPosOfChunk(int chunkNum)
        {
                if (chunkNum <= remainder)
                        return chunkNum*(quotient + 1);
                return longerChunkItems + (chunkNum - remainder)*quotient;
        }


        // Basic setup routine

        void setup(int totalItems, int numChunks, int localChunkNumber)
        {
                quotient  = totalItems / numChunks;
                remainder = totalItems % numChunks;
                numLocalItems = quotient + (localChunkNumber < remainder);
                localStartPos = localChunkNumber*quotient 
                                  + std::min(remainder,localChunkNumber);
                longerChunkItems = remainder*(quotient + 1);
        };


        // Set up from a comm-based object

        void setup(int totalItems, commBased& commObj)
        {
                setup(totalItems, commObj.getCommSize(), commObj.getRank());
        };


        // Set up from an MPI communicator

        void setup(int totalItems, MPI_Comm comm)
        {
          commBased tempCommObj;
          tempCommObj.setupComm(comm);
          setup(totalItems, tempCommObj);
        };


    // Empty constructor

        evenSplit() :
        numLocalItems(-1),
        localStartPos(-1),
        quotient(-1),
        remainder(0),
        longerChunkItems(0)
        { };


    // reset everything to the beginning

        void reset()
        {
                numLocalItems    = -1;
                localStartPos    = -1;
                quotient         = -1;
                remainder            = 0;
                longerChunkItems = 0;
        };


    // constructors calling setup immediately

        evenSplit(int totalItems, int numChunks, int localChunkNumber)
        {
                setup(totalItems, numChunks, localChunkNumber);
        };

        evenSplit(int totalItems, commBased& commObj)
        {
                setup(totalItems, commObj);
        };

        evenSplit(int totalItems, MPI_Comm comm)
        {
                setup(totalItems, comm);
        };


protected:

        int numLocalItems;
        int localStartPos;
        int quotient;
        int remainder;
        int longerChunkItems;

};
