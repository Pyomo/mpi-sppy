###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""MPI-based test for the cylinder xhat generator in mrp_generic.

Must be run with mpiexec, e.g.:
    mpiexec -np 3 python -m mpi4py mpisppy/tests/test_mrp_cylinders.py

Tests that _cylinder_xhat_generator returns a valid xhat dict when
using hub-and-spoke decomposition (PH + lagrangian spoke).
"""

import sys

from mpi4py import MPI

from mpisppy.tests.utils import get_solver
from mpisppy.utils import config
from mpisppy.generic.mrp import _cylinder_xhat_generator
import mpisppy.tests.examples.farmer as farmer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

solver_available, solver_name, _, _ = get_solver()

if not solver_available:
    if rank == 0:
        print("SKIP: no solver available")
    sys.exit(0)


def _get_cylinder_cfg():
    """Build a Config with PH + lagrangian spoke args for farmer.

    Registers all the config groups that do_decomp needs (the same ones
    that generic_cylinders' parse_args registers).
    """
    cfg = config.Config()

    # Register all config groups that do_decomp / configure_extensions /
    # build_spoke_list may access.  This mirrors parse_args in
    # mpisppy/generic/parsing.py.
    cfg.proper_bundle_config()
    cfg.pickle_scenarios_config()
    cfg.EF_base()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()
    cfg.aph_args()
    cfg.subgradient_args()
    cfg.fixer_args()
    cfg.relaxed_ph_fixer_args()
    cfg.integer_relax_then_enforce_args()
    cfg.gapper_args()
    cfg.gapper_args(name="lagrangian")
    cfg.ph_primal_args()
    cfg.ph_dual_args()
    cfg.relaxed_ph_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.subgradient_bounder_args()
    cfg.xhatshuffle_args()
    cfg.xhatxbar_args()
    cfg.norm_rho_args()
    cfg.primal_dual_rho_args()
    cfg.converger_args()
    cfg.wxbar_read_write_args()
    cfg.tracking_args()
    cfg.gradient_args()
    cfg.dynamic_rho_args()
    cfg.reduced_costs_args()
    cfg.sep_rho_args()
    cfg.coeff_rho_args()
    cfg.sensi_rho_args()
    cfg.reduced_costs_rho_args()

    import pyomo.common.config as pyofig
    cfg.add_to_config("user_defined_extensions",
                      description="Space-delimited module names for user extensions",
                      domain=pyofig.ListOf(str),
                      default=None)
    cfg.add_to_config(name="solution_base_name",
                      description="Base name for solution output files",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="write_scenario_lp_mps_files_dir",
                      description="Directory for LP/MPS files (default None)",
                      domain=str,
                      default=None)
    cfg.add_to_config(name="module_name",
                      description="Name of the model module",
                      domain=str,
                      default=None)

    # Model-specific
    cfg.quick_assign("use_integer", bool, False)
    cfg.quick_assign("crops_multiplier", int, 1)
    cfg.quick_assign("num_scens", int, 3)

    # Solver
    cfg.solver_name = solver_name
    cfg.EF_solver_name = solver_name

    # PH settings
    cfg.default_rho = 1.0
    cfg.max_iterations = 3

    # Enable lagrangian spoke and xhatshuffle
    cfg.lagrangian = True
    cfg.xhatshuffle = True
    return cfg


def test_cylinder_xhat_generator():
    """Test that _cylinder_xhat_generator returns a valid xhat."""
    cfg = _get_cylinder_cfg()
    scenario_names = farmer.scenario_names_creator(3)

    xhat = _cylinder_xhat_generator(
        scenario_names,
        solver_name=solver_name,
        solver_options=None,
        cfg=cfg,
        module_name="mpisppy.tests.examples.farmer",
        module=farmer,
    )

    if rank == 0:
        assert "ROOT" in xhat, f"xhat missing ROOT key, got: {xhat.keys()}"
        assert len(xhat["ROOT"]) == 3, \
            f"farmer should have 3 first-stage vars, got {len(xhat['ROOT'])}"
        for i, val in enumerate(xhat["ROOT"]):
            assert val >= -1e-6, \
                f"Planting decision {i} should be non-negative, got {val}"
        print("test_cylinder_xhat_generator: PASS")


if __name__ == "__main__":
    test_cylinder_xhat_generator()
