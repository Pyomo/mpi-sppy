###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""FW-APH (Frank-Wolfe Asynchronous Projective Hedging) as an mpi-sppy hub.

The algorithm itself is C++ (mpisppy/aphfw_cpp), reached through the compiled
``aphfw_cpp`` module. That module is optional and is not built by a normal
mpi-sppy install, so it is imported only when the hub actually runs; importing
this file never needs it.
"""

import ctypes
import json
import logging
import os
import tempfile

import numpy as np

import mpisppy.phbase
from mpisppy import MPI
from mpisppy.extensions.scenario_lp_mps_files import Scenario_lp_mps_files

logger = logging.getLogger(__name__)

# The C++ sources, present only in a clone of the repository: the wheel
# carries no aphfw_cpp files.
_APHFW_SRC = os.path.join(os.path.dirname(mpisppy.__file__), "aphfw_cpp")


def _build_help(src_dir=_APHFW_SRC):
    """How to build aphfw_cpp from here, which depends on how mpi-sppy got here."""
    msg = ("The aphfw hub (--aphfw-hub) needs the compiled aphfw_cpp module, "
           "which a normal mpi-sppy install does not build. ")
    if not os.path.isfile(os.path.join(src_dir, "pyproject.toml")):
        return msg + (
            "Its C++ sources come only with a clone of the mpi-sppy repository "
            "from GitHub, not with the package from PyPI: install mpi-sppy from "
            "a clone, then run python -m mpisppy.aphfw_cpp.build there.")
    return msg + (
        "It needs MPI (the one mpi4py uses), a C++23 compiler, and Gurobi, "
        "CPLEX, Xpress or HiGHS. Build and install it with, e.g.,\n"
        "    python -m mpisppy.aphfw_cpp.build --solver gurobi "
        "--solver-home /path/to/gurobi/linux64\n"
        "(--help lists the options; the raw form is pip install " + src_dir +
        ", see the pyproject.toml there).")


def _import_aphfw_cpp():
    """Import the compiled module, or explain how to build it."""
    try:
        import aphfw_cpp
    except ImportError as e:
        raise RuntimeError(f"{_build_help()}\n(import failed: {e})") from e
    return aphfw_cpp


# Float options whose C++ default is 0, which the C++ parser (positive values
# only) cannot be given explicitly; leaving them out gives the same value.
_ZERO_IS_CPP_DEFAULT = {"aphfw_fw_tau", "aphfw_iter0_w_multiplier"}


def _check_dir_shared(comm, path):
    """Raise on every rank unless every rank of comm can see directory path."""
    if comm.allreduce(int(os.path.isdir(path)), op=MPI.MIN) == 0:
        raise RuntimeError(
            f"Not every hub rank can see the FW-APH scenario directory {path}. "
            "When the hub runs on more than one node, give a directory on a "
            "filesystem they share with --aphfw-scenario-dir.")


class APHFWOpt(mpisppy.phbase.PHBase):
    """Wraps C++ FWAPH for use as an mpi-sppy hub optimizer (hub only)."""

    def __init__(self, options, *args, **kwargs):
        super().__init__(options, *args, **kwargs)
        self._cpp = None
        self._terminate_flag = None
        self._scenario_dir = None
        self._local_nonant_length = None
        self._final_cycle = 0

    def aphfw_main(self, finalize=True):
        """Run the C++ FW-APH algorithm, syncing with spokes every cycle.

        Args:
            finalize: If True, clean up the C++ library on exit.

        Returns:
            trivial_bound: The initial Lagrangian bound from iter0.
        """
        # 1. Create Pyomo scenarios via PHBase (no prox — C++ handles that)
        self.PH_Prep(attach_duals=True, attach_prox=False)
        if not self.is_minimizing:
            raise RuntimeError(
                "--aphfw-hub supports only minimization: the C++ solves "
                "minimize the objective as written in the scenario files.")
        # The C++ does the solves, so there are no Pyomo solver plugins.
        # Extensions that update a persistent solver after fixing a variable
        # check for one; this tells them there is none (fixings reach the C++
        # through _push_bounds_to_cpp).
        for s in self.local_scenarios.values():
            s._solver_plugin = None

        # 2. Write MPS + JSON + rho files to a temporary directory
        self._write_scenario_files()

        # 3. Initialize C++ with cylinder_comm + file paths
        self._cpp = _import_aphfw_cpp().APHFWLibrary()

        # Set explicit scenario names so C++ matches our rank assignment
        local_names = list(self.local_scenarios.keys())
        self._cpp.set_explicit_scenario_names(local_names)

        cfg_dict = self._make_cpp_config()
        self._cpp.init(self.mpicomm, self._scenario_dir, cfg_dict)
        self._cpp.set_hub_mode(True)
        # The C++ read rho from the scenario files, in the original
        # objective's units; under --aphfw-obj-scale it needs scaled rho.
        self._local_nonant_length = self._cpp.get_local_nonant_length()
        self._push_rho_to_cpp()

        # Omicron multiplier: used at the post-iter-0 freeze call. C++ also
        # holds this value (registered option), but Python needs it to pass
        # through freeze_omicron(). Default matches the C++ default.
        self._omicron_rho_multiplier = cfg_dict.get("omicron-rho-multiplier", 1.0)

        # 4. Set up external terminate flag
        self._terminate_flag = ctypes.c_int(0)
        self._cpp.set_external_terminate_flag(
            ctypes.addressof(self._terminate_flag)
        )

        # The iter-0 Lagrangian bound is computed by the C++ startup sequence,
        # which the stepping API defers to the first run_cycles call, so it
        # cannot be read here -- only once the loop below has turned once.
        self.trivial_bound = None

        # 5. Stepping loop
        has_extensions = getattr(self, "extobject", None) is not None

        if has_extensions:
            # Nothing has been solved yet (the C++ startup runs in the first
            # run_cycles call), so the Pyomo side is as PH_Prep left it, as it
            # is for PH's pre_iter0.  There are no Pyomo solvers to create;
            # iter0_post_solver_creation marks the point the C++ is ready.
            self._PHIter = 0
            self.extobject.pre_iter0()
            self.extobject.iter0_post_solver_creation()
            # Push any rho changes from pre_iter0 (e.g., CoeffRho) to C++
            self._push_rho_to_cpp()

        while True:
            completed = self._cpp.run_cycles(1)

            # startupSequence() has now run, so the iter-0 bound exists.
            if self.trivial_bound is None:
                self.trivial_bound = self._unscale_obj(
                    self._cpp.get_current_lagrangian_bound())
                if self._bound_is_valid():
                    self.best_bound_obj_val = self.trivial_bound

            if self._terminate_flag.value:
                break

            # If no cycles completed, the algorithm has finished
            # (converged, max cycles reached, or external termination)
            if completed == 0:
                break
            self._PHIter = self._cpp.get_current_cycle()

            # 6. Run mpi-sppy extensions (e.g., dynamic rho)
            if has_extensions:
                self._sync_cpp_to_pyomo()

                if self._cpp.get_current_cycle() == 1:
                    # post_iter0 expects _PHIter == 0 so that wtracker
                    # can bootstrap its W history (local_Ws[0]).
                    saved_iter = self._PHIter
                    self._PHIter = 0
                    self.extobject.post_iter0()
                    self._PHIter = saved_iter

                # The cycle's solves are done, so both of PH's per-iteration
                # hooks run here.
                self.extobject.miditer()
                self.extobject.enditer()
                self._push_rho_to_cpp()

            # Freeze omicron exactly once, after iter 0 completes.
            # Fires whether or not extensions are enabled — post-iter-0
            # rho drivers have already run and rho has been pushed to C++
            # above, so the frozen value reflects any iter-0 rho adjustments.
            # If --aphfw-omicron-all or --aphfw-omicron-csv was set, C++ already froze
            # at setup and this call is a no-op (guarded by omicron_is_frozen).
            if (self._cpp.get_current_cycle() == 1
                    and not self._cpp.omicron_is_frozen()):
                self._cpp.freeze_omicron(self._omicron_rho_multiplier)

            # 7. Sync with spokes (send_nonants/send_ws are overridden
            #    in APHFWHub to use C++ values)
            if self.spcomm is not None:
                self.spcomm.sync()
                if has_extensions:
                    if self._cpp.get_current_cycle() == 1:
                        self.extobject.post_iter0_after_sync()
                    self.extobject.enditer_after_sync()
                    self._push_rho_to_cpp()
                # Push any bound changes from receive_nonant_bounds() or
                # ReducedCostsFixer to C++ (Lagrangian and Gradient solvers).
                self._push_bounds_to_cpp()
                if self.spcomm.is_converged():
                    self._terminate_flag.value = 1
                    break

            if self._cpp.is_converged():
                break

        if has_extensions:
            self._sync_cpp_to_pyomo()
            self.extobject.post_everything()

        # Optional final Lagrangian bound.  Every hub rank reaches this point
        # together (the loop exits on rank-symmetric conditions), which matters
        # because the C++ side reduces over the cylinder comm.
        if self._get_option("aphfw_final_lagrangian", False):
            self._do_final_lagrangian()

        if finalize:
            self.aphfw_finalize()

        return self.trivial_bound

    def _get_option(self, name, default=None):
        """Read an option from cfg if present, else from the options dict."""
        cfg = self.options.get("cfg", None)
        if cfg is not None and hasattr(cfg, name):
            return getattr(cfg, name)
        return self.options.get(name, default)

    def _unscale_obj(self, value):
        """Convert a C++ objective-space value back to the original objective.

        --aphfw-obj-scale multiplies every scenario objective coefficient in C++, so
        bounds come back scaled; the spokes and the inter-cylinder gap use the
        unscaled objective.
        """
        if value is None:
            return None
        obj_scale = self._get_option("aphfw_obj_scale", 1.0) or 1.0
        if obj_scale == 1.0:
            return value
        return value / obj_scale

    def _do_final_lagrangian(self):
        """Run the termination Lagrangian and let its bound compete.

        In hub mode the C++ termination sequence skips this solve, on the
        theory that a --lagrangian spoke is producing bounds continuously.
        With --aphfw-final-lagrangian we pay for it anyway, at the final W,
        and hand the result to the hub cylinder, which keeps it only if it
        beats every bound the bound spokes have produced.
        """
        if self._cpp is None:
            return None

        bound = self._unscale_obj(self._cpp.compute_final_lagrangian_bound())

        # With nonants fixed by an extension the Lagrangian is over a
        # restricted set and is not an outer bound.
        if not self._bound_is_valid():
            return None
        self.best_bound_obj_val = bound
        if self.spcomm is not None:
            self.spcomm.publish_outer_bound(bound)
        return bound

    def _bound_is_valid(self):
        """_can_update_best_bound, agreed over the hub's ranks.

        Collective: a bound is published from every hub rank or from none.
        """
        return self.mpicomm.allreduce(int(self._can_update_best_bound()),
                                      op=MPI.MIN) == 1

    def aphfw_finalize(self):
        """Clean up C++ resources and temp directory."""
        if self._cpp is not None:
            self._final_cycle = self._cpp.get_current_cycle()
            self._cpp.finalize()
            self._cpp = None
        if self._scenario_dir is not None and self._cleanup_scenario_dir:
            import shutil
            # Every hub rank has finished with the files; one rank removes the
            # parent directory (mkdtemp parent / "scen"), which they share.
            self.mpicomm.Barrier()
            if self.cylinder_rank == 0:
                shutil.rmtree(os.path.dirname(self._scenario_dir),
                              ignore_errors=True)
            self._scenario_dir = None

    def _write_scenario_files(self):
        """Write MPS/LP + JSON + rho files for C++ consumption."""
        cfg = self.options.get("cfg", None)

        keep_files = self.options.get("aphfw_keep_scenario_files", False)
        self._cleanup_scenario_dir = not keep_files

        # Generate a unique directory path WITHOUT creating it yet,
        # because Scenario_lp_mps_files.__init__ does makedirs(exist_ok=False).
        # Rank 0 picks the name, broadcasts to all ranks.  Every hub rank
        # writes its scenarios there and the C++ on every rank reads them all,
        # so all hub ranks must see the same directory.
        base = self._get_option("aphfw_scenario_dir", "") or None
        if self.cylinder_rank == 0:
            parent = tempfile.mkdtemp(prefix="aphfw_", dir=base)
            self._scenario_dir = os.path.join(parent, "scen")
        else:
            self._scenario_dir = None
        self._scenario_dir = self.mpicomm.bcast(self._scenario_dir, root=0)
        _check_dir_shared(self.mpicomm, os.path.dirname(self._scenario_dir))

        if self.cylinder_rank == 0:
            logger.debug(f"Writing scenario files to {self._scenario_dir}")

        # Use mpi-sppy's Scenario_lp_mps_files extension to write files
        opts = {
            "write_scenario_lp_mps_files_dir": self._scenario_dir,
            "cfg": cfg,
        }
        self.options["write_lp_mps_extension_options"] = opts
        writer = Scenario_lp_mps_files(self)
        writer.pre_iter0()

        # Solver parameter JSON files for the C++ solves.  The C++ side takes
        # only mipGap, timeLimit and threads: iter0.json is for the Lagrangian
        # solves (startup and termination), grad.json for the Frank-Wolfe
        # linear subproblems, nl.json for the proximal QPs.
        if self.cylinder_rank == 0:
            def _opt(name, default):
                value = self._get_option(name, default)
                return default if value is None else value
            solver_cfg = {
                "timeLimit": float(_opt("aphfw_solve_time_limit", 300.0)),
                "threads": int(_opt("max_solver_threads", 1)),
            }
            iter0_cfg = dict(solver_cfg)
            if _opt("iter0_mipgap", None) is not None:
                iter0_cfg["mipGap"] = float(_opt("iter0_mipgap", None))
            grad_cfg = dict(solver_cfg, mipGap=float(_opt("iterk_mipgap", 1e-4)))
            for name, params in (("iter0.json", iter0_cfg),
                                 ("nl.json", solver_cfg),
                                 ("grad.json", grad_cfg)):
                with open(os.path.join(self._scenario_dir, name), "w") as f:
                    json.dump(params, f)
            for name in ("solver_options", "solver_options_file"):
                if _opt(name, None):
                    logger.warning(
                        f"--{name.replace('_', '-')} does not reach the C++ "
                        "hub's solves, which take only --iter0-mipgap, "
                        "--iterk-mipgap, --max-solver-threads and "
                        "--aphfw-solve-time-limit; it still applies to the "
                        "spokes.")
        self.mpicomm.Barrier()

    def _make_cpp_config(self):
        """Build a config dict from mpi-sppy options for C++ init."""
        cfg = self.options.get("cfg", None)
        d = {}

        # Map mpi-sppy config names to C++ CLI option names
        mappings = {
            "aphfw_fw_tmax": "FW-tmax",
            "aphfw_fw_tau": "FW-tau",
            "aphfw_primal_term_thresh": "primal-term-thresh",
            "aphfw_dual_term_thresh": "dual-term-thresh",
            "aphfw_gamma_inverse": "gamma-inverse",
            "aph_nu": "nu",
            "aphfw_staleness_threshold": "staleness-threshhold",
            "aphfw_solve_fraction": "solve-fraction",
            "aphfw_display_progress": "display-progress",
            "aphfw_verbosity": "verbosity",
            "aphfw_workers_per_rank": "workersPerRank",
            "aphfw_block_sync": "block-sync",
            "aphfw_leading_dispatch_only": "leading-dispatch-only",
            "aphfw_obj_scale": "obj-scale",
            "aphfw_obj_cutoff": "FW-obj-cutoff",
            "aphfw_W_trace_prefix": "W-trace-prefix",
            "aphfw_solve_times_trace_prefix": "solve-times-trace-prefix",
            "aphfw_omicron_rho_multiplier": "omicron-rho-multiplier",
            "aphfw_omicron_all": "omicron-all",
            "aphfw_omicron_csv": "omicron-csv",
            "aphfw_omicron_min": "omicron-min",
            "aphfw_omicron_max": "omicron-max",
            "aphfw_check_omicron_invariant": "check-omicron-invariant",
            "aphfw_iter0_fw_step": "iter0-fw-step",
            "aphfw_iter0_w_multiplier": "iter0-w-multiplier",
        }

        for py_name, cpp_name in mappings.items():
            val = None
            if cfg is not None and hasattr(cfg, py_name):
                val = getattr(cfg, py_name)
            elif py_name in self.options:
                val = self.options[py_name]
            if val is not None:
                # aphfw_omicron_all uses -1.0 as the "unset" sentinel; C++ enforces
                # a positive range on the option, so don't push the sentinel.
                if py_name == "aphfw_omicron_all" and isinstance(val, float) and val < 0.0:
                    continue
                if isinstance(val, float) and val <= 0.0:
                    if val == 0.0 and py_name in _ZERO_IS_CPP_DEFAULT:
                        continue
                    raise ValueError(
                        f"--{py_name.replace('_', '-')} must be positive "
                        f"(got {val})")
                d[cpp_name] = val

        # --max-iterations (PHIterLimit) caps the cycles, as it caps every
        # hub's iterations.  The C++ runs at least one cycle per step, so 0,
        # which means iteration 0 only for PH, has no equivalent here.
        max_cycles = int(self.options["PHIterLimit"])
        if max_cycles < 1:
            raise ValueError("--max-iterations must be at least 1 with "
                             f"--aphfw-hub (got {max_cycles})")
        d["aph-max-cycles"] = max_cycles

        # Rho: let C++ read per-scenario {scenarioName}_rho.csv files
        # written by Scenario_lp_mps_files.  These contain per-variable rho
        # values from rho_setter (if provided) or defaultPHrho.  We do NOT
        # pass --rho-all (which would override per-variable values) or
        # --rho-csv (which would use a single CSV for all scenarios).
        # C++ derives each scenario's CSV path from its problemName.

        # Solver parameter JSON files (generated by _write_scenario_files)
        d["iter0-json"] = os.path.join(self._scenario_dir, "iter0.json")
        d["nl-json"] = os.path.join(self._scenario_dir, "nl.json")
        d["grad-json"] = os.path.join(self._scenario_dir, "grad.json")

        # Suppress solver stdout
        d["solver-log-file"] = os.path.join(self._scenario_dir, "solver.log")

        return d

    def _sync_cpp_to_pyomo(self):
        """Copy C++ x, W, xbar, rho into Pyomo scenario objects.

        This populates the Pyomo-side attributes that mpi-sppy extensions
        (e.g., dynamic rho updaters) read when computing new rho values.

        x comes from get_all_local_prox_x, the proximal minimizer the
        algorithm iterates on, not from get_all_local_nonants, which exports
        a vertex of K_s for the xhat spokes.  Every extension that reads x is
        measuring dispersion of the scenario solutions about a consensus
        point; on an integer problem that consensus point is generally
        fractional while the exported vertices are not, so vertices leave the
        dispersion floored away from zero however well converged the run is.
        """
        if self._cpp is None:
            return

        n = self._local_nonant_length
        x_buf = np.empty(n, dtype=np.float64)
        w_buf = np.empty(n, dtype=np.float64)
        xbar_buf = np.empty(n, dtype=np.float64)
        rho_buf = np.empty(n, dtype=np.float64)

        self._cpp.get_all_local_prox_x(x_buf)
        self._cpp.get_all_local_w(w_buf)
        self._cpp.get_all_local_xbar(xbar_buf)
        self._cpp.get_all_local_rho(rho_buf)
        # W and rho in the original objective's units, as send_ws gives the
        # spokes W
        w_buf = self._unscale_obj(w_buf)
        rho_buf = self._unscale_obj(rho_buf)

        idx = 0
        for sname, scenario in self.local_scenarios.items():
            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                if not xvar.fixed:  # a fixed one keeps the value it is fixed at
                    xvar._value = float(x_buf[idx])
                scenario._mpisppy_model.W[ndn_i]._value = float(w_buf[idx])
                scenario._mpisppy_model.xbars[ndn_i]._value = float(xbar_buf[idx])
                scenario._mpisppy_model.rho[ndn_i]._value = float(rho_buf[idx])
                idx += 1

        self._PHIter = self._cpp.get_current_cycle()

    def _push_rho_to_cpp(self):
        """Read rho from Pyomo scenarios and push to C++."""
        if self._cpp is None:
            return

        n = self._local_nonant_length
        rho_buf = np.empty(n, dtype=np.float64)

        idx = 0
        for sname, scenario in self.local_scenarios.items():
            for ndn_i in scenario._mpisppy_data.nonant_indices:
                rho_buf[idx] = float(scenario._mpisppy_model.rho[ndn_i]._value)
                idx += 1

        # rho in the original objective's units -> the C++'s scaled objective
        obj_scale = self._get_option("aphfw_obj_scale", 1.0) or 1.0
        self._cpp.set_all_local_rho(rho_buf * obj_scale)

    def _push_bounds_to_cpp(self):
        """Read nonant variable bounds from Pyomo scenarios and push to C++.

        A fixed variable is pushed with both bounds at its value: fixers
        (e.g. ReducedCostsFixer) act through xvar.fix(), which leaves lb and
        ub as they were.
        """
        if self._cpp is None:
            return

        n = self._local_nonant_length
        lb_buf = np.empty(n, dtype=np.float64)
        ub_buf = np.empty(n, dtype=np.float64)

        idx = 0
        for sname, scenario in self.local_scenarios.items():
            for ndn_i, xvar in scenario._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    lb_buf[idx] = ub_buf[idx] = float(xvar.value)
                else:
                    lb_buf[idx] = float(xvar.lb) if xvar.lb is not None else -np.inf
                    ub_buf[idx] = float(xvar.ub) if xvar.ub is not None else np.inf
                idx += 1

        self._cpp.set_all_local_nonant_bounds(lb_buf, ub_buf)

    def _write_back_to_pyomo(self):
        """Copy C++ nonant values back into Pyomo scenario variables.

        This is needed so that Pyomo-level operations (denouement, objective
        evaluation) can see the solution values that C++ computed.
        """
        if self._cpp is None:
            return

        n = self._local_nonant_length
        x_vals = np.empty(n, dtype=np.float64)
        self._cpp.get_all_local_nonants(x_vals)

        idx = 0
        for sname, scenario in self.local_scenarios.items():
            for xvar in scenario._mpisppy_data.nonant_indices.values():
                xvar._value = float(x_vals[idx])
                idx += 1

