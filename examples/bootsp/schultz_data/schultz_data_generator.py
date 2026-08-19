###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Generate the schultz_data.csv dataset for the data-file bootstrap example.
#
#   python schultz_data_generator.py
#
# Each row is one observed realization of the two right-hand-side values
# (xi1, xi2) of the little Schultz problem. The values are sampled once, with a
# fixed seed, so the committed CSV is reproducible; the point of the example is
# that the bootstrap code treats this file as "the data".

import os
import csv
import numpy as np

N = 200            # number of observations in the dataset
SEED = 42          # fixed so the committed file is reproducible


def make_data(n=N, seed=SEED):
    rng = np.random.default_rng(seed)
    # integers in 5..15, centered near 10 (Binomial(10, p) + 5), so the
    # dataset has natural repetition/frequency rather than being a flat grid.
    xi1 = 5 + rng.binomial(10, 0.50, n)
    xi2 = 5 + rng.binomial(10, 0.45, n)
    return list(zip(xi1.tolist(), xi2.tolist()))


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "schultz_data.csv")
    rows = make_data()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["xi1", "xi2"])
        w.writerows(rows)
    print(f"wrote {len(rows)} observations to {out}")
