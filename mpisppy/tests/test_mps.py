###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# test mps utilities
import os
import json
import types
import shutil
import tempfile
import unittest
from mip import OptimizationStatus
import mpisppy.problem_io.mps_reader as mps_reader
import mpisppy.problem_io.mps_module as mps_module
import mpisppy.utils.sputils as sputils
import mpisppy.utils.config as config
import mpisppy.utils.proper_bundler as proper_bundler
from mpisppy.generic.parsing import name_lists
from mpisppy.tests.utils import get_solver, limit_solver_threads
import pyomo.environ as pyo
import mip  # pip install mip (from coin-or)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES_DIR = os.path.join(_THIS_DIR, "examples")

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver(persistent_OK=False)

class TestMPSReader(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _reader_body(self, fname):
        pyomo_model = mps_reader.read_mps_and_create_pyomo_model(fname)

        opt = pyo.SolverFactory(solver_name)
        limit_solver_threads(opt, solver_name)
        opt.solve(pyomo_model)
        pyomo_obj = pyo.value(pyomo_model.objective)

        m = mip.Model(solver_name="CBC")
        m.read(fname)
        m.optimize()   # returns a status, btw
        coin_obj = m.objective_value

        status = m.optimize()
        # Optional: m.verbose = 1  # if you want CBC logging next time
        if status not in (OptimizationStatus.OPTIMAL, OptimizationStatus.FEASIBLE):
            # Drop helpful breadcrumbs
            m.write("cbc_readback.lp")      # what CBC thinks it read
            m.write("cbc_solution.sol")     # if any partial solution exists
            self.fail(f"CBC status={status.name}, num_solutions={m.num_solutions}. "
                      f"Objective is {m.objective_value}. "
                      f'Wrote "cbc_readback.lp" for inspection.')
        coin_obj = m.objective_value

        print(f"{fname=}, {pyomo_obj=}")
        self.assertAlmostEqual(coin_obj, pyomo_obj, places=3,
                               delta=None, msg=None)
        
    def test_mps_reader_scen0_densenames(self):
        self._reader_body(os.path.join(_EXAMPLES_DIR, "scen0_densenames.mps"))

    def test_mps_reader_test1(self):
        self._reader_body(os.path.join(_EXAMPLES_DIR, "test1.mps"))

    def test_mps_reader_sizes1(self):
        self._reader_body(os.path.join(_EXAMPLES_DIR, "sizes1.mps"))


# mps_module: .lp file support and per-nonant rho from {s}_rho.csv.
# The fixture's nonant labels contain parentheses (x(1), x(2)) so the
# ( -> _ normalization in both scenario_creator and rho_list_from_csv is
# exercised (the reader builds components named x_1_, x_2_).
_MPS_MODULE_DATA = os.path.join(_EXAMPLES_DIR, "mps_module_data")


class TestMPSModule(unittest.TestCase):
    def _cfg(self, directory):
        # mps_module.scenario_creator only needs cfg.mps_files_directory;
        # scenario_names_creator reads the module global, so set both.
        mps_module.mps_files_directory = directory
        return types.SimpleNamespace(mps_files_directory=directory)

    @unittest.skipIf(not solver_available, "no solver available")
    def test_lp_scenario_creator_build_solve(self):
        # Round-trip an .lp scenario through scenario_creator, then solve.
        cfg = self._cfg(_MPS_MODULE_DATA)
        model = mps_module.scenario_creator("Scenario1", cfg=cfg)
        # reader maps lp labels x(1)/x(2) to components x_1_/x_2_
        nonant_names = [v.name for v in
                        model._mpisppy_node_list[0].nonant_vardata_list]
        self.assertEqual(nonant_names, ["x_1_", "x_2_"])
        opt = pyo.SolverFactory(solver_name)
        limit_solver_threads(opt, solver_name)
        opt.solve(model)
        # min 3*x1 + 2*x2 + 5*y  s.t. x1+x2+y >= 10, x1 <= 4  ->  x2=10, obj=20
        self.assertAlmostEqual(pyo.value(model.objective), 20.0, places=4)

    def test_scenario_names_creator_lp_dir(self):
        self._cfg(_MPS_MODULE_DATA)
        names = mps_module.scenario_names_creator(None)
        self.assertEqual(names, ["Scenario1", "Scenario2"])

    @unittest.skipIf(not solver_available, "no solver available")
    def test_rho_setter_from_csv(self):
        # _rho_setter returns (id(vardata), rho) for each nonant in the csv,
        # with paren-normalized names resolving to the right components.
        cfg = self._cfg(_MPS_MODULE_DATA)
        model = mps_module.scenario_creator("Scenario1", cfg=cfg)
        rho_list = mps_module._rho_setter(model)
        id_to_name = {id(v): v.name
                      for v in model.component_data_objects(pyo.Var)}
        got = {id_to_name[vid]: rho for vid, rho in rho_list}
        self.assertEqual(got, {"x_1_": 11.0, "x_2_": 22.0})

    @unittest.skipIf(not solver_available, "no solver available")
    def test_rho_setter_no_csv_falls_back(self):
        # A dir with no {s}_rho.csv -> _rho_csv_path None -> _rho_setter == [].
        with tempfile.TemporaryDirectory() as td:
            for suffix in (".lp", "_nonants.json"):
                shutil.copy(
                    os.path.join(_MPS_MODULE_DATA, "Scenario1" + suffix),
                    os.path.join(td, "Scenario1" + suffix))
            cfg = self._cfg(td)
            model = mps_module.scenario_creator("Scenario1", cfg=cfg)
            self.assertIsNone(model._rho_csv_path)
            self.assertEqual(mps_module._rho_setter(model), [])

    def test_both_exts_lp_preferred(self):
        # When both {s}.lp and {s}.mps exist, the resolver returns the .lp.
        with tempfile.TemporaryDirectory() as td:
            open(os.path.join(td, "Scenario1.lp"), "w").close()
            open(os.path.join(td, "Scenario1.mps"), "w").close()
            path = mps_module._scenario_model_path(td, "Scenario1")
            self.assertTrue(path.endswith("Scenario1.lp"))

    def test_missing_model_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                mps_module._scenario_model_path(td, "Scenario1")


def _build_mps_bundle(cfg, scen_names):
    """Build a proper-bundle-shaped EF from mps_module scenarios.

    Mirrors mpisppy.utils.proper_bundler: create_EF over the sub-scenarios,
    attach the ROOT node from the non-surrogate ref Vars, then attach the
    SPBase-style nonant metadata (_mpisppy_data.nonant_indices) that
    mps_module._rho_setter reads for a bundle. No solve is involved.
    """
    bundle = sputils.create_EF(
        scen_names, mps_module.scenario_creator,
        scenario_creator_kwargs={"cfg": cfg},
        EF_name="Bundle_test", suppress_warnings=True,
    )
    nonantlist = [v for idx, v in bundle.ref_vars.items()
                  if idx[0] == "ROOT" and idx not in bundle.ref_surrogate_vars]
    surrogates = [v for idx, v in bundle.ref_surrogate_vars.items()
                  if idx[0] == "ROOT"]
    sputils.attach_root_node(bundle, 0, nonantlist, None, surrogates)
    nonant_indices = {}
    for node in bundle._mpisppy_node_list:
        for i, v in enumerate(node.nonant_vardata_list):
            nonant_indices[(node.name, i)] = v
    bundle._mpisppy_data.nonant_indices = nonant_indices
    return bundle


def _write_scenario(directory, name, prob, rho_map):
    """Write a {name}.lp/_nonants.json/_rho.csv triple into directory.

    The model is the shared fixture lp (vars x(1), x(2), y); prob and rho_map
    let a test pin the probability weighting and per-nonant rho independently.
    """
    shutil.copy(os.path.join(_MPS_MODULE_DATA, "Scenario1.lp"),
                os.path.join(directory, name + ".lp"))
    nonants = {
        "scenarioData": {"name": name, "scenProb": prob},
        "treeData": {"globalNodeCount": 1,
                     "nodes": {"ROOT": {"serialNumber": 0, "condProb": 1.0,
                                        "nonAnts": ["x(1)", "x(2)"]}}},
    }
    with open(os.path.join(directory, name + "_nonants.json"), "w") as f:
        json.dump(nonants, f)
    with open(os.path.join(directory, name + "_rho.csv"), "w") as f:
        f.write("varname,rho\n")
        for var, rho in rho_map.items():
            f.write(f"{var},{rho}\n")


class TestMPSModuleBundleRho(unittest.TestCase):
    """_rho_setter assembles bundle rho from the sub-scenarios' {s}_rho.csv."""

    def _cfg(self, directory):
        mps_module.mps_files_directory = directory
        return types.SimpleNamespace(mps_files_directory=directory)

    def _by_nonant(self, bundle, rho_list):
        # map returned (id(refVar), rho) back to the nonant suffix (x_1_/x_2_),
        # dropping the sub-scenario component prefix on the ref Var name.
        id_to_var = {id(v): v
                     for v in bundle._mpisppy_data.nonant_indices.values()}
        return {id_to_var[vid].name.split(".")[-1]: rho
                for vid, rho in rho_list}

    def test_bundle_rho_from_consistent_csvs(self):
        # Identical sub-scenario rho files -> bundle nonant rho is the common
        # value (independent of probabilities), keyed to the bundle's ref Vars.
        cfg = self._cfg(_MPS_MODULE_DATA)
        bundle = _build_mps_bundle(cfg, ["Scenario1", "Scenario2"])
        got = self._by_nonant(bundle, mps_module._rho_setter(bundle))
        self.assertEqual(got, {"x_1_": 11.0, "x_2_": 22.0})

    def test_bundle_rho_disagreement_is_error(self):
        # PH uses one rho per bundle nonant, so sub-scenarios that disagree on a
        # nonant's rho are a malformed input -> hard error. The check is local to
        # the bundle (no MPI collective). x(2) agrees; x(1) differs (10 vs 20).
        with tempfile.TemporaryDirectory() as td:
            _write_scenario(td, "Scenario1", 0.5, {"x(1)": 10.0, "x(2)": 20.0})
            _write_scenario(td, "Scenario2", 0.5, {"x(1)": 20.0, "x(2)": 20.0})
            cfg = self._cfg(td)
            bundle = _build_mps_bundle(cfg, ["Scenario1", "Scenario2"])
            with self.assertRaises(RuntimeError):
                mps_module._rho_setter(bundle)

    def test_bundle_no_csv_falls_back(self):
        # A bundle whose sub-scenarios carry no rho file -> [] (=> --default-rho).
        with tempfile.TemporaryDirectory() as td:
            for name in ("Scenario1", "Scenario2"):
                for suffix in (".lp", "_nonants.json"):
                    shutil.copy(
                        os.path.join(_MPS_MODULE_DATA, name + suffix),
                        os.path.join(td, name + suffix))
            cfg = self._cfg(td)
            bundle = _build_mps_bundle(cfg, ["Scenario1", "Scenario2"])
            self.assertEqual(mps_module._rho_setter(bundle), [])


class TestMPSModuleProperBundling(unittest.TestCase):
    """Proper bundling is reachable from the file path: the scenario count is
    implied by the directory (no --num-scens), so parsing.name_lists infers it
    and can size the bundles."""

    def test_bundle_count_inferred_from_mps_directory(self):
        # _MPS_MODULE_DATA holds 2 scenario files; at 1 scenario per bundle that
        # is 2 bundles, and cfg.num_scens must be filled in (to 2) for the bundle
        # math, even though no --num-scens was given.
        cfg = config.Config()
        cfg.add_branching_factors()       # default None -> two-stage
        cfg.proper_bundle_config()        # scenarios_per_bundle, *_bundles_dir
        cfg.quick_assign("scenarios_per_bundle", int, 1)
        mps_module.mps_files_directory = _MPS_MODULE_DATA
        bundle_wrapper = proper_bundler.ProperBundler(mps_module)
        names, _ = name_lists(mps_module, cfg, bundle_wrapper=bundle_wrapper)
        self.assertEqual(cfg.num_scens, 2)   # inferred from the 2 files
        self.assertEqual(len(names), 2)      # 2 bundles, 1 scenario each
        self.assertTrue(all(n.startswith("Bundle_") for n in names))


if __name__ == '__main__':
    unittest.main()
