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
import types
import shutil
import tempfile
import unittest
from mip import OptimizationStatus
import mpisppy.problem_io.mps_reader as mps_reader
import mpisppy.problem_io.mps_module as mps_module
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


if __name__ == '__main__':
    unittest.main()
