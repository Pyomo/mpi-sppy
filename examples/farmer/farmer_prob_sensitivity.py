###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Probability sensitivity on the Extensive Form (issue #797).
#
# The farmer decides how many acres to plant of each crop before the yields
# are known. Here we fix the scenario set and sweep the probability vector:
# we vary how much weight is placed on the low-yield scenario and watch the
# optimal first-stage planting decision respond.
#
# The point of the example is that the Extensive Form is built ONCE with
# mutable_probability=True. Each new probability vector is pushed in with
# set_scenario_probabilities() and re-solved with reuse_instance=True, so a
# persistent solver keeps its loaded instance across the whole sweep instead
# of rebuilding the model for every vector.
#
# Run, e.g.:
#   python farmer_prob_sensitivity.py --solver-name gurobi_persistent
#   python farmer_prob_sensitivity.py --solver-name appsi_highs --num-scens 5

import argparse

import farmer

from mpisppy.opt.ef import ExtensiveForm


def make_probability_vector(scenario_names, low_yield_weight):
    """Weight the first ("low yield") scenario by low_yield_weight and split
    the remaining probability uniformly over the others. Returns a mapping
    that sums to 1, as set_scenario_probabilities requires."""
    others = scenario_names[1:]
    rest = (1.0 - low_yield_weight) / len(others)
    pv = {scenario_names[0]: low_yield_weight}
    for sn in others:
        pv[sn] = rest
    return pv


def main():
    parser = argparse.ArgumentParser(
        description="Sweep scenario probabilities on a farmer EF without "
                    "rebuilding the model (issue #797).")
    parser.add_argument("--solver-name", default="gurobi_persistent",
                        help="Pyomo solver name. A persistent solver "
                             "(e.g. gurobi_persistent, cplex_persistent) or an "
                             "APPSI solver (e.g. appsi_highs) keeps its loaded "
                             "instance across the sweep. Default: "
                             "gurobi_persistent.")
    parser.add_argument("--num-scens", type=int, default=3,
                        help="Number of scenarios. Default: 3.")
    args = parser.parse_args()

    scenario_names = farmer.scenario_names_creator(args.num_scens)
    scenario_creator_kwargs = {"num_scens": args.num_scens}

    # Build the EF once. mutable_probability=True stores each scenario's
    # probability as a mutable Pyomo Param in the objective instead of baking
    # it in as a float constant, so the objective can be re-weighted in place.
    ef = ExtensiveForm(
        options={"solver": args.solver_name},
        all_scenario_names=scenario_names,
        scenario_creator=farmer.scenario_creator,
        scenario_creator_kwargs=scenario_creator_kwargs,
        mutable_probability=True,
    )

    # Sweep the weight on the low-yield scenario. A larger weight makes the
    # farmer more worried about a bad harvest and shifts the planting decision.
    low_yield_weights = [1.0 / args.num_scens, 0.25, 0.5, 0.75, 0.9]

    print(f"Probability sensitivity for {args.num_scens} farmer scenarios "
          f"using {args.solver_name}")
    print(f"(weight is the probability placed on {scenario_names[0]})\n")
    header = f"{'weight':>8}  {'objective':>14}   first-stage acreage"
    print(header)
    print("-" * len(header))

    for i, w in enumerate(low_yield_weights):
        pv = make_probability_vector(scenario_names, w)
        ef.set_scenario_probabilities(pv)
        # reuse_instance=True after the first solve keeps the persistent
        # solver's instance loaded; only the objective coefficients change.
        ef.solve_extensive_form(reuse_instance=(i > 0))

        obj = ef.get_objective_value()
        root = ef.get_root_solution()
        acreage = "  ".join(f"{name.split('[')[-1].rstrip(']')}={val:8.2f}"
                            for name, val in sorted(root.items()))
        print(f"{w:8.3f}  {obj:14.2f}   {acreage}")


if __name__ == "__main__":
    main()
