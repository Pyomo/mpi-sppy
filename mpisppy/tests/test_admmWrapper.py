###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# TBD: make these tests less fragile
"""Tests for AdmmWrapper.

Phase references (A, B.1, B.2, ...) in this file's docstrings track the
phased plan in doc/designs/admm_user_api_automation_design.md.
For the ADMM vocabulary used below (before-wrap scenario, wrapped
scenario, wrap, ADMM subproblem, ...), see the module docstring of
mpisppy.utils.admmWrapper.
"""
import unittest
import pyomo.environ as pyo
import mpisppy.utils.admmWrapper as admmWrapper
from mpisppy.utils.admmWrapper import (
    _admm_normalize_consensus_vars,
    _first_stage_var_names,
    _merge_first_stage_into_consensus_vars,
)
import mpisppy.tests.examples.distr.distr as distr
from mpisppy.utils import config
from mpisppy.tests.utils import get_solver
from mpisppy import MPI
import subprocess
import os
import sys

# Parse --python-args (extra args inserted after "python" in subcommands, e.g. for coverage)
python_args = ""
_remaining = []
_i = 1
while _i < len(sys.argv):
    if sys.argv[_i].startswith("--python-args="):
        python_args = sys.argv[_i].split("=", 1)[1]
    elif sys.argv[_i] == "--python-args" and _i + 1 < len(sys.argv):
        _i += 1
        python_args = sys.argv[_i]
    else:
        _remaining.append(sys.argv[_i])
    _i += 1
sys.argv = [sys.argv[0]] + _remaining

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_DISTR_DIR = os.path.join(_PROJECT_ROOT, "examples", "distr")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
_DISTR_DIR = os.path.join(_PROJECT_ROOT, "examples", "distr")

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()

class TestAdmmWrapper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def _cfg_creator(self, num_scens):
        cfg = config.Config()

        cfg.num_scens_required()
        cfg.num_scens = num_scens
        return cfg 


    def _make_admm(self, num_scens,verbose=False):
        cfg = self._cfg_creator(num_scens)
        options = {}
        all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
        scenario_creator = distr.scenario_creator
        scenario_creator_kwargs = distr.kw_creator(cfg)
        consensus_vars = distr.consensus_vars_creator(cfg.num_scens)
        n_cylinders = 1 #distr_admm_cylinders._count_cylinders(cfg)
        return admmWrapper.AdmmWrapper(options,
                            all_scenario_names, 
                            scenario_creator,
                            consensus_vars,
                            n_cylinders=n_cylinders,
                            mpicomm=MPI.COMM_WORLD,
                            scenario_creator_kwargs=scenario_creator_kwargs,
                            verbose=verbose,
                            )
    
    def test_constructor(self):
        self._make_admm(2,verbose=True)
        for i in range(3,5):
            self._make_admm(i)

    def test_variable_probability(self):        
        admm = self._make_admm(3)
        q = dict()
        for sname, s in admm.local_scenarios.items():
            q[sname] = admm.var_prob_list(s)
        self.assertEqual(q["Region1"][0][1], 0.5)
        self.assertEqual(q["Region3"][0][1], 0)

    def test_admmWrapper_scenario_creator(self):
        admm = self._make_admm(3)
        sname = "Region3"
        q = admm.admmWrapper_scenario_creator(sname)
        self.assertTrue(q.y__DC1DC2__.is_fixed())
        self.assertFalse(q.y["DC3_1DC1"].is_fixed())
    
    def _slack_name(self, dummy_node):
        return f"y[{dummy_node}]"

    def test_get_scenario_unscaled(self):
        admm = self._make_admm(3)
        sname = "Region1"
        scenario = admm.get_scenario_unscaled(sname)
        self.assertIs(scenario, admm.local_scenarios[sname])

    def test_assign_variable_probs_error1(self):
        admm = self._make_admm(3)
        admm.consensus_vars["Region1"].append(self._slack_name("DC2DC3"))
        self.assertRaises(RuntimeError, admm.assign_variable_probs)
        
    def test_assign_variable_probs_error2(self):
        admm = self._make_admm(3)
        admm.consensus_vars["Region1"].remove(self._slack_name("DC3_1DC1"))
        self.assertRaises(RuntimeError, admm.assign_variable_probs)

    
    def _extracting_output(self, line):
        import re
        pattern = r'\[\s*\d+\.\d+\]\s+\d+\s+(?:L\s*B?|B\s*L?)?\s+([-.\d]+)\s+([-.\d]+)'

        match = re.search(pattern, line)

        if match:
            outer_bound = match.group(1)
            inner_bound = match.group(2)
            return float(outer_bound), float(inner_bound)
        else:
            raise RuntimeError("Cannot find outer and inner bounds in pattern"
                               f" in this output {line=}")

    @unittest.skip(
        "mpiexec subprocesses die silently (returncode=1, empty stdout/stderr) "
        "when launched via subprocess.run from inside pytest, but the exact "
        "same command works when run from a bare Python script or a shell. "
        "Likely a pytest stdio-capture / file-descriptor interaction with "
        "Open MPI's I/O forwarding.  Run manually via examples/distr/go.bash "
        "to exercise this path until the root cause is diagnosed."
    )
    def test_values(self):
        command_line_pairs = [(f"mpiexec -np 3 python -u {python_args} -m mpi4py distr_admm_cylinders.py --num-scens 3 --default-rho 10 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.01 --ensure-xhat-feas" \
                         , f"python {python_args} distr_ef.py --solver-name {solver_name} --num-scens 3 --ensure-xhat-feas"), \
                         (f"mpiexec -np 6 python -u {python_args} -m mpi4py distr_admm_cylinders.py --num-scens 5 --default-rho 10 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --mnpr 6 --rel-gap 0.05 --scalable --ensure-xhat-feas" \
                         , f"python {python_args} distr_ef.py --solver-name {solver_name} --num-scens 5 --ensure-xhat-feas --mnpr 6 --scalable")]
        original_dir = os.getcwd()
        for command_line_pair in command_line_pairs:
            os.chdir(_DISTR_DIR)
            objectives = {}
            command = command_line_pair[0].split()
            
            result = subprocess.run(command, capture_output=True, text=True)
            # Filter out harmless MPI warnings from stderr
            stderr_lines = [line for line in result.stderr.splitlines()
                            if line.strip() and "btl_tcp" not in line
                            and "osc_ucx" not in line] if result.stderr else []
            if stderr_lines:
                print("Error output:")
                raise RuntimeError(result.stderr)
                

            # Check the standard output
            if result.stdout:
                result_by_line = result.stdout.strip().split('\n')
            else:
                print(f"{result.stdout=}, {result.returncode=}")
                raise RuntimeError(f"Cannot get output from {command=}")
            
                
            target_line = "Iter.           Best Bound  Best Incumbent      Rel. Gap        Abs. Gap"
            precedent_line_target = False
            i = 0
            for line in result_by_line:
                if precedent_line_target:
                    if i%2 == 1:
                        outer_bound, inner_bound = self._extracting_output(line)
                        objectives["outer bound"] = outer_bound
                        objectives["inner bound"] = inner_bound
                    precedent_line_target = False 
                    i += 1
                elif target_line in line:
                    precedent_line_target = True
            
            # For the EF
            command = command_line_pair[1].split()
            result = subprocess.run(command, capture_output=True, text=True)
            result_by_line = result.stdout.strip().split('\n')
            for i in range(len(result_by_line)):
                if "EF objective" in result_by_line[-i-1]: #should be on last line but we can check
                    decomposed_line = result_by_line[-i-1].split(': ')
                    objectives["EF objective"] = float(decomposed_line[1])
            try:
                correct_order = objectives["outer bound"] <= objectives["EF objective"] <= objectives["inner bound"]
            except Exception:
                raise RuntimeError("The output could not be read to capture the values")
            assert correct_order, f' We obtained {objectives["outer bound"]=}, {objectives["EF objective"]=}, {objectives["inner bound"]=}'
            os.chdir(original_dir)


class TestAdmmConsensusVarsNormalize(unittest.TestCase):
    """B.1: consensus_vars accepts Pyomo Var/VarData as well as strings."""

    def _model_with_named_vars(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.y = pyo.Var([("A", "B"), ("B", "C")])
        return m

    def test_flat_form_strings_unchanged(self):
        cv = {"Sub1": ["x", "y[('A', 'B')]"], "Sub2": ["x"]}
        out = _admm_normalize_consensus_vars(cv, tuple_form=False)
        self.assertEqual(out, cv)
        self.assertIsNot(out, cv)  # new dict, not aliased

    def test_flat_form_var_objects(self):
        m = self._model_with_named_vars()
        cv = {"Sub1": [m.x, m.y[("A", "B")]], "Sub2": [m.x]}
        out = _admm_normalize_consensus_vars(cv, tuple_form=False)
        self.assertEqual(out["Sub1"], [m.x.name, m.y[("A", "B")].name])
        self.assertEqual(out["Sub2"], [m.x.name])

    def test_flat_form_mixed(self):
        m = self._model_with_named_vars()
        cv = {"Sub1": ["x", m.y[("B", "C")]]}
        out = _admm_normalize_consensus_vars(cv, tuple_form=False)
        self.assertEqual(out["Sub1"], ["x", m.y[("B", "C")].name])

    def test_tuple_form_strings_unchanged(self):
        cv = {"Sub1": [("x", 1), ("y[('A', 'B')]", 2)]}
        out = _admm_normalize_consensus_vars(cv, tuple_form=True)
        self.assertEqual(out, cv)

    def test_tuple_form_var_objects(self):
        m = self._model_with_named_vars()
        cv = {"Sub1": [(m.x, 1), (m.y[("A", "B")], 2)]}
        out = _admm_normalize_consensus_vars(cv, tuple_form=True)
        self.assertEqual(out["Sub1"], [(m.x.name, 1), (m.y[("A", "B")].name, 2)])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            _admm_normalize_consensus_vars({"Sub1": [42]}, tuple_form=False)


class TestAdmmWrapperVarConsensusInputs(unittest.TestCase):
    """B.1: AdmmWrapper accepts Pyomo Var objects in consensus_vars
    and produces the same varprob_dict as the equivalent string form
    (no behavioral change other than the relaxed input type)."""

    def _cfg(self, num_scens):
        cfg = config.Config()
        cfg.num_scens_required()
        cfg.num_scens = num_scens
        return cfg

    def _make_admm_from(self, consensus_vars, cfg):
        options = {}
        all_scenario_names = distr.scenario_names_creator(num_scens=cfg.num_scens)
        return admmWrapper.AdmmWrapper(
            options,
            all_scenario_names,
            distr.scenario_creator,
            consensus_vars,
            n_cylinders=1,
            mpicomm=MPI.COMM_WORLD,
            scenario_creator_kwargs=distr.kw_creator(cfg),
            verbose=False,
        )

    def test_var_input_matches_string_input(self):
        cfg = self._cfg(3)
        consensus_vars_str = distr.consensus_vars_creator(cfg.num_scens)

        # Build a Var-flavored consensus_vars by resolving each name on
        # the ADMM subproblem's own before-wrap scenario.  Pyomo
        # VarData holds its parent block via a weakref, so we must
        # keep the source before-wrap scenarios alive until the
        # wrapper has snapshotted their .name attributes.
        kw = distr.kw_creator(cfg)
        live_scenarios = []
        consensus_vars_var = {}
        for sub, entries in consensus_vars_str.items():
            scen = distr.scenario_creator(sub, **kw)
            live_scenarios.append(scen)
            resolved = []
            for vstr in entries:
                v = scen.find_component(vstr)
                resolved.append(v if v is not None else vstr)
            consensus_vars_var[sub] = resolved

        admm_str = self._make_admm_from(consensus_vars_str, self._cfg(3))
        admm_var = self._make_admm_from(consensus_vars_var, self._cfg(3))
        del live_scenarios

        # consensus_vars on the wrapper should be string-equal post-normalization
        self.assertEqual(admm_str.consensus_vars, admm_var.consensus_vars)
        # ...and probability bookkeeping should match (compare by probability
        # values; the underlying Var id()s differ across constructor calls).
        for sname in admm_str.local_scenarios:
            probs_str = [p for (_, p) in admm_str.var_prob_list(admm_str.local_scenarios[sname])]
            probs_var = [p for (_, p) in admm_var.var_prob_list(admm_var.local_scenarios[sname])]
            self.assertEqual(probs_str, probs_var, f"mismatch for {sname}")


class TestMergeFirstStageIntoConsensusVars(unittest.TestCase):
    """B.2: helper that appends each ADMM subproblem's first-stage Var
    names to its consensus_vars entry, with per-subproblem semantics
    and de-dup."""

    def test_merge_basic(self):
        cv = {"A": [("x", 2)], "B": [("y", 2)]}
        fs = {"A": ["fsA"], "B": ["fsB"]}
        out = _merge_first_stage_into_consensus_vars(cv, fs, root_stage=1)
        self.assertEqual(out["A"], [("x", 2), ("fsA", 1)])
        self.assertEqual(out["B"], [("y", 2), ("fsB", 1)])

    def test_merge_dedup(self):
        cv = {"A": [("x", 2), ("fsA", 1)]}
        fs = {"A": ["fsA"]}
        out = _merge_first_stage_into_consensus_vars(cv, fs, root_stage=1)
        self.assertEqual(out["A"], [("x", 2), ("fsA", 1)])

    def test_merge_subs_with_different_first_stage(self):
        cv = {"R1": [("z", 2)], "R2": [("z", 2)]}
        fs = {"R1": ["y[F1_1]", "y[F1_2]"], "R2": ["y[F2_1]"]}
        out = _merge_first_stage_into_consensus_vars(cv, fs, root_stage=1)
        self.assertEqual(out["R1"], [("z", 2), ("y[F1_1]", 1), ("y[F1_2]", 1)])
        self.assertEqual(out["R2"], [("z", 2), ("y[F2_1]", 1)])

    def test_merge_missing_sub_is_noop(self):
        cv = {"A": [("x", 2)], "B": [("y", 2)]}
        fs = {"A": ["fsA"]}  # B absent
        out = _merge_first_stage_into_consensus_vars(cv, fs, root_stage=1)
        self.assertEqual(out["A"], [("x", 2), ("fsA", 1)])
        self.assertEqual(out["B"], [("y", 2)])


class TestFirstStageVarNames(unittest.TestCase):
    """B.2 helper: expand a first_stage_varlist return value to per-VarData
    names, so indexed Var containers don't end up in varprob_dict by their
    container name (which would later KeyError in variable_probability)."""

    def test_scalar_var(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        self.assertEqual(list(_first_stage_var_names([m.x])), ["x"])

    def test_indexed_var_expands(self):
        m = pyo.ConcreteModel()
        m.fs = pyo.Var([2025, 2026])
        self.assertEqual(list(_first_stage_var_names([m.fs])),
                         ["fs[2025]", "fs[2026]"])

    def test_vardata_passthrough(self):
        m = pyo.ConcreteModel()
        m.fs = pyo.Var([2025, 2026])
        self.assertEqual(list(_first_stage_var_names([m.fs[2025]])),
                         ["fs[2025]"])

    def test_mixed_inputs(self):
        m = pyo.ConcreteModel()
        m.x = pyo.Var()
        m.fs = pyo.Var(["a", "b"])
        self.assertEqual(
            list(_first_stage_var_names([m.x, m.fs, "explicit_name"])),
            ["x", "fs[a]", "fs[b]", "explicit_name"])

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            list(_first_stage_var_names([42]))


if __name__ == '__main__':
    unittest.main()
