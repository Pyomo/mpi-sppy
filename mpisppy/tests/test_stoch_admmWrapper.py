###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Tests for Stoch_AdmmWrapper.

Phase references (A, B.1, B.2, ...) in this file's docstrings track the
phased plan in doc/designs/admm_user_api_automation_design.md.
For the ADMM vocabulary used below (before-wrap scenario, wrapped
scenario, wrap, ADMM subproblem, ...), see the module docstring of
mpisppy.utils.admmWrapper.
"""
import unittest
import subprocess
import sys
import mpisppy.tests.examples.stoch_distr.stoch_distr_admm_cylinders as stoch_distr_admm_cylinders
import mpisppy.tests.examples.stoch_distr.stoch_distr as stoch_distr
from mpisppy.utils import config
from mpisppy.tests.utils import get_solver
import os

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
_TEST_STOCH_DISTR_DIR = os.path.join(_THIS_DIR, "examples", "stoch_distr")
_STOCH_DISTR_DIR = os.path.join(_PROJECT_ROOT, "examples", "stoch_distr")

solver_available, solver_name, persistent_available, persistent_solver_name= get_solver()

# Resolve paths at module load time (before any test changes cwd)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))

class TestStochAdmmWrapper(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def _cfg_creator(self, num_stoch_scens, num_admm_subproblems):
        # Needs to be modified
        cfg = config.Config()
        stoch_distr.inparser_adder(cfg)

        cfg.num_stoch_scens = num_stoch_scens
        cfg.num_admm_subproblems = num_admm_subproblems

        return cfg
    
    def _make_admm(self, num_stoch_scens, num_admm_subproblems, verbose=False):
        cfg = self._cfg_creator(num_stoch_scens, num_admm_subproblems)
        admm, all_admm_stoch_subproblem_scenario_names = stoch_distr_admm_cylinders._make_admm(cfg,n_cylinders=1,verbose=verbose)
        return admm

    def _get_base_options(self):
        cfg = config.Config()
        cfg.quick_assign("run_async", bool, False)
        cfg.quick_assign("num_stoch_scens", int, 4)
        cfg.quick_assign("num_admm_subproblems", int, 2)
        cfg.quick_assign("default_rho", int, 10)
        cfg.quick_assign("EF_2stage", bool, True)
        cfg.quick_assign("num_batches", int, 2)
        cfg.quick_assign("batch_size", int, 10)

    
    def test_constructor(self):
        for num_stoch_scens in range(3,5):
            for num_admm_subproblems in range(3,5):
                self._make_admm(num_stoch_scens, num_admm_subproblems)

    def test_variable_probability(self):
        admm = self._make_admm(4,3)
        q = dict()
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            q[sname] = admm.var_prob_list(s)
        self.assertEqual(q["ADMM_STOCH_Region1_StochasticScenario1"][0][1], 0.5)
        self.assertEqual(q["ADMM_STOCH_Region3_StochasticScenario1"][0][1], 0)

    def test_admmWrapper_scenario_creator(self):
        admm = self._make_admm(4,3)
        sname = "ADMM_STOCH_Region3_StochasticScenario1"
        q = admm.admmWrapper_scenario_creator(sname)
        self.assertTrue(q.y__DC1DC2__.is_fixed())
        self.assertFalse(q.y["DC3_1DC1"].is_fixed())
    
    def test_get_scenario_unscaled(self):
        admm = self._make_admm(4, 3)
        sname = "ADMM_STOCH_Region1_StochasticScenario1"
        scenario = admm.get_scenario_unscaled(sname)
        self.assertIs(scenario, admm.local_admm_stoch_subproblem_scenarios[sname])

    def _slack_name(self, dummy_node):
        return f"y[{dummy_node}]"

    def test_assign_variable_probs_error1(self):
        admm = self._make_admm(2,3)
        admm.consensus_vars["Region1"].append((self._slack_name("DC2DC3"),2)) # The variable is added in the second stage
        self.assertRaises(RuntimeError, admm.assign_variable_probs, admm)
        
    def test_assign_variable_probs_error2(self):
        admm = self._make_admm(2,3)
        admm.consensus_vars["Region1"].remove((self._slack_name("DC3_1DC1"),2))
        self.assertRaises(RuntimeError, admm.assign_variable_probs, admm)

    
    def _extracting_output(self, line):
        import re
        pattern = r'\[\s*\d+\.\d+\]\s+\d+\s+(?:L\s*B?|B\s*L?)?\s+([-.\d]+)\s+([-.\d]+)'

        match = re.search(pattern, line)

        if match:
            outer_bound = match.group(1)
            inner_bound = match.group(2)
            return float(outer_bound), float(inner_bound)
        else:
            raise RuntimeError("The test is probably not correctly adapted: can't match the format of the line")


    @unittest.skip(
        "mpiexec subprocesses die silently (returncode=1, empty stdout/stderr) "
        "when launched via subprocess.run from inside pytest, but the exact "
        "same command works when run from a bare Python script or a shell. "
        "Likely a pytest stdio-capture / file-descriptor interaction with "
        "Open MPI's I/O forwarding.  Run manually via "
        "examples/stoch_distr/go.bash to exercise this path until the root "
        "cause is diagnosed."
    )
    def test_values(self):
        command_line_pairs = [(f"mpiexec -np 3 python -u {python_args} -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 10 --num-admm-subproblems 2 --default-rho 10 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.001 --num-stages 3" \
                         , f"python {python_args} stoch_distr_ef.py --solver-name {solver_name} --num-stoch-scens 10 --num-admm-subproblems 2 --num-stages 3"), \
                         (f"mpiexec -np 3 python -u {python_args} -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 5 --num-admm-subproblems 3 --default-rho 5 --solver-name {solver_name} --max-iterations 50 --xhatxbar --lagrangian --rel-gap 0.01 --ensure-xhat-feas" \
                         , f"python {python_args} stoch_distr_ef.py --solver-name {solver_name} --num-stoch-scens 5 --num-admm-subproblems 3 --ensure-xhat-feas"), \
                              (f"mpiexec -np 6 python -u {python_args} -m mpi4py stoch_distr_admm_cylinders.py --num-stoch-scens 4 --num-admm-subproblems 5 --default-rho 15 --solver-name {solver_name} --max-iterations 30 --xhatxbar --lagrangian --mnpr 5 --scalable --ensure-xhat-feas" \
                         , f"python {python_args} stoch_distr_ef.py --solver-name {solver_name} --num-stoch-scens 4 --num-admm-subproblems 5 --mnpr 5 --scalable --ensure-xhat-feas")  ]
        #command_line = f"mpiexec -np 6 python -m mpi4py examples/stoch_distr/stoch_distr_admm_cylinders.py --num-admm-subproblems 2 --num-stoch-scens 4 --default-rho 10 --solver-name {solver_name} --max-iterations 100 --xhatxbar --lagrangian"
        original_dir = os.getcwd()
        for j in range(len(command_line_pairs)):
            if j == 0: # The first line is executed in the test directory because it has a 3-stage problem. This one does not insure xhatfeasibility but luckily works
                target_directory = _TEST_STOCH_DISTR_DIR
            else: # The other lines are executed in the real directory because it ensures xhat feasibility
                target_directory = _STOCH_DISTR_DIR
            os.chdir(target_directory)
            objectives = {}
            command = command_line_pairs[j][0].split()
            
            result = subprocess.run(command, capture_output=True, text=True)
            # Filter out harmless MPI warnings from stderr
            stderr_lines = [line for line in result.stderr.splitlines()
                            if line.strip() and "btl_tcp" not in line
                            and "osc_ucx" not in line] if result.stderr else []
            if stderr_lines:
                print("Error output:")
                print(result.stderr)
                raise RuntimeError("Error encountered as shown above.")
            # Check the standard output
            if result.stdout:
                result_by_line = result.stdout.strip().split('\n')
            else:
                raise RuntimeError(f"No results in stdout for {command=} \n {result.returncode=}.")
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
            command = command_line_pairs[j][1].split()
            result = subprocess.run(command, capture_output=True, text=True)
            result_by_line = result.stdout.strip().split('\n')

            for i in range(len(result_by_line)):
                if "EF objective" in result_by_line[-i-1]: #should be on last line but we can check
                    decomposed_line = result_by_line[-i-1].split(': ')
                    objectives["EF objective"] = float(decomposed_line[1])#math.ceil(float(decomposed_line[1]))
            try:
                correct_order = objectives["outer bound"] <= (objectives["EF objective"] +0.01)\
                    <= (objectives["inner bound"] + 0.02)
            except Exception:
                raise RuntimeError("The output could not be read to capture the values")
            assert correct_order, f' We obtained {objectives["outer bound"]=}, {objectives["EF objective"]=}, {objectives["inner bound"]=}'
            os.chdir(original_dir)
    


    def _extracting_outer_bound(self, stdout):
        """Extract the last outer bound from PH output."""
        import re
        target_line = "Iter.           Best Bound  Best Incumbent      Rel. Gap        Abs. Gap"
        result_by_line = stdout.strip().split('\n')
        outer_bound = None
        in_results = False
        for line in result_by_line:
            if target_line in line:
                in_results = True
                continue
            if in_results:
                # Match a line with iteration number and bounds
                match = re.search(r'\[\s*\d+\.\d+\]\s+\d+\s+(?:L\s*B?|B\s*L?)?\s+([-.\d]+)', line)
                if match:
                    outer_bound = float(match.group(1))
                in_results = False
        return outer_bound

    @unittest.skipUnless(solver_available, "no solver available")
    def test_bundled_admm_via_generic_cylinders(self):
        """Test stochastic ADMM with proper bundles via generic_cylinders.

        Runs with --scenarios-per-bundle equal to --num-stoch-scens
        (full bundling: one bundle per ADMM subproblem). Verifies that
        the Lagrangian outer bound matches the EF objective.
        """
        target_dir = os.path.join(_REPO_ROOT, 'examples', 'stoch_distr')
        generic_cyl = os.path.join(_REPO_ROOT, 'mpisppy', 'generic_cylinders.py')
        original_dir = os.getcwd()
        os.chdir(target_dir)
        try:
            # Run bundled stochastic ADMM via generic_cylinders
            ph_command = (
                f"mpiexec -np 2 python -u -m mpi4py "
                f"{generic_cyl} "
                f"--module-name stoch_distr "
                f"--stoch-admm --num-stoch-scens 4 --num-admm-subproblems 2 "
                f"--default-rho 10 --solver-name {solver_name} "
                f"--max-iterations 30 --lagrangian "
                f"--scenarios-per-bundle 4"
            ).split()
            # Clean MPI env vars that get set by the singleton MPI init
            # from module imports — they interfere with mpiexec subprocesses.
            clean_env = {k: v for k, v in os.environ.items()
                         if not k.startswith(('OMPI_', 'PMIX_', 'PMI_'))}
            result = subprocess.run(ph_command, capture_output=True, text=True,
                                    timeout=120, env=clean_env)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Bundled ADMM run failed (rc={result.returncode})\n"
                    f"CMD: {' '.join(ph_command)}\n"
                    f"CWD: {os.getcwd()}\n"
                    f"STDOUT: {result.stdout[-1500:]}\n"
                    f"STDERR: {result.stderr[-1500:]}"
                )

            outer_bound = self._extracting_outer_bound(result.stdout)
            self.assertIsNotNone(outer_bound, "Could not extract outer bound from output")

            # Run EF for comparison
            ef_command = (
                f"python stoch_distr_ef.py --solver-name {solver_name} "
                f"--num-stoch-scens 4 --num-admm-subproblems 2"
            ).split()
            ef_result = subprocess.run(ef_command, capture_output=True, text=True)
            ef_obj = None
            for line in ef_result.stdout.strip().split('\n'):
                if "EF objective" in line:
                    ef_obj = float(line.split(': ')[1])
            self.assertIsNotNone(ef_obj, "Could not extract EF objective")

            # Lagrangian outer bound should be close to EF objective
            # (within 1% for this small problem)
            self.assertAlmostEqual(
                outer_bound, ef_obj, delta=abs(ef_obj) * 0.01,
                msg=f"Outer bound {outer_bound} != EF objective {ef_obj}"
            )
        finally:
            os.chdir(original_dir)


    @unittest.skipUnless(solver_available, "no solver available")
    def test_bundled_vs_unbundled(self):
        """Run the same stochastic ADMM problem with and without bundling.

        Both should produce outer bounds close to the EF objective.
        This verifies that bundling does not change the answer.
        """
        target_dir = os.path.join(_REPO_ROOT, 'examples', 'stoch_distr')
        generic_cyl = os.path.join(_REPO_ROOT, 'mpisppy', 'generic_cylinders.py')
        original_dir = os.getcwd()
        os.chdir(target_dir)
        try:
            clean_env = {k: v for k, v in os.environ.items()
                         if not k.startswith(('OMPI_', 'PMIX_', 'PMI_'))}

            num_stoch_scens = 4
            num_admm = 2
            common_args = (
                f"--stoch-admm --num-stoch-scens {num_stoch_scens} "
                f"--num-admm-subproblems {num_admm} "
                f"--default-rho 10 --solver-name {solver_name} "
                f"--max-iterations 30 --lagrangian "
                f"--turn-off-names-check"
            )

            # --- unbundled run ---
            unbundled_cmd = (
                f"mpiexec -np 2 python -u -m mpi4py {generic_cyl} "
                f"--module-name stoch_distr {common_args}"
            ).split()
            unbundled_result = subprocess.run(
                unbundled_cmd, capture_output=True, text=True,
                timeout=120, env=clean_env)
            if unbundled_result.returncode != 0:
                raise RuntimeError(
                    f"Unbundled run failed (rc={unbundled_result.returncode})\n"
                    f"STDOUT: {unbundled_result.stdout[-1000:]}\n"
                    f"STDERR: {unbundled_result.stderr[-1000:]}")
            unbundled_bound = self._extracting_outer_bound(unbundled_result.stdout)
            self.assertIsNotNone(unbundled_bound,
                                 "Could not extract outer bound from unbundled run")

            # --- bundled run ---
            bundled_cmd = (
                f"mpiexec -np 2 python -u -m mpi4py {generic_cyl} "
                f"--module-name stoch_distr {common_args} "
                f"--scenarios-per-bundle {num_stoch_scens}"
            ).split()
            bundled_result = subprocess.run(
                bundled_cmd, capture_output=True, text=True,
                timeout=120, env=clean_env)
            if bundled_result.returncode != 0:
                raise RuntimeError(
                    f"Bundled run failed (rc={bundled_result.returncode})\n"
                    f"STDOUT: {bundled_result.stdout[-1000:]}\n"
                    f"STDERR: {bundled_result.stderr[-1000:]}")
            bundled_bound = self._extracting_outer_bound(bundled_result.stdout)
            self.assertIsNotNone(bundled_bound,
                                 "Could not extract outer bound from bundled run")

            # --- EF reference ---
            ef_cmd = (
                f"python stoch_distr_ef.py --solver-name {solver_name} "
                f"--num-stoch-scens {num_stoch_scens} "
                f"--num-admm-subproblems {num_admm}"
            ).split()
            ef_result = subprocess.run(ef_cmd, capture_output=True, text=True)
            ef_obj = None
            for line in ef_result.stdout.strip().split('\n'):
                if "EF objective" in line:
                    ef_obj = float(line.split(': ')[1])
            self.assertIsNotNone(ef_obj, "Could not extract EF objective")

            # Both bounds should be close to EF objective (within 5%)
            tol = abs(ef_obj) * 0.05
            self.assertAlmostEqual(
                unbundled_bound, ef_obj, delta=tol,
                msg=f"Unbundled bound {unbundled_bound} not close to EF {ef_obj}")
            self.assertAlmostEqual(
                bundled_bound, ef_obj, delta=tol,
                msg=f"Bundled bound {bundled_bound} not close to EF {ef_obj}")

            # And they should be close to each other (within 2%)
            self.assertAlmostEqual(
                unbundled_bound, bundled_bound,
                delta=abs(ef_obj) * 0.02,
                msg=f"Unbundled {unbundled_bound} vs bundled {bundled_bound} "
                    f"differ too much (EF={ef_obj})")
        finally:
            os.chdir(original_dir)


    def test_preserves_user_surrogates_and_ef_suppl(self):
        """User surrogate_nonant_list and nonant_ef_suppl_list attached to a
        stoch scenario's root node must survive the stage-rewrite that
        Stoch_AdmmWrapper performs.

        The wrapper re-builds each existing stage node with only the
        consensus vars in nonant_list; without preservation, anything the
        user attached via attach_root_node's optional lists would silently
        disappear.
        """
        import pyomo.environ as pyo
        import mpisppy.utils.sputils as sputils
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        def scenario_creator(sname, **kwargs):
            parts = sname.split("_")
            admm_part = parts[2]
            m = pyo.ConcreteModel()
            # consensus vars: subproblem A owns x, B owns y (partial)
            if admm_part == "A":
                m.x = pyo.Var(bounds=(0, 1))
                own = m.x
            else:
                m.y = pyo.Var(bounds=(0, 1))
                own = m.y
            m.z = pyo.Var(bounds=(0, 1))  # user surrogate
            m.e = pyo.Var(bounds=(0, 1))  # user EF-suppl nonant
            m.cost = pyo.Expression(expr=0)
            m.obj = pyo.Objective(expr=own, sense=pyo.minimize)
            sputils.attach_root_node(
                m, m.cost, [own],
                nonant_ef_suppl_list=[m.e],
                surrogate_nonant_list=[m.z])
            m._mpisppy_probability = "uniform"
            return m

        admm_names = ["A", "B"]
        stoch_names = ["S1", "S2"]
        all_names = [f"ADMM_STOCH_{a}_{s}"
                     for s in stoch_names for a in admm_names]

        def split(sname):
            parts = sname.split("_")
            return parts[2], "_".join(parts[3:])

        consensus_vars = {"A": [("x", 1)], "B": [("y", 1)]}

        admm = Stoch_AdmmWrapper(
            options={},
            all_admm_stoch_subproblem_scenario_names=all_names,
            split_admm_stoch_subproblem_scenario_name=split,
            admm_subproblem_names=admm_names,
            stoch_scenario_names=stoch_names,
            scenario_creator=scenario_creator,
            consensus_vars=consensus_vars,
            n_cylinders=1,
            mpicomm=MPI.COMM_WORLD,
            scenario_creator_kwargs={},
        )

        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            root = s._mpisppy_node_list[0]
            self.assertIn(
                s.z, root.surrogate_vardatas,
                f"{sname}: user surrogate m.z was dropped by stage rewrite")
            self.assertIn(
                s.e, root.nonant_ef_suppl_vardata_list,
                f"{sname}: user EF-suppl m.e was dropped by stage rewrite")
            # Also: the admm dummy for the absent consensus var (x on B,
            # y on A) must still be marked surrogate so EF doesn't force it.
            admm_part = sname.split("_")[2]
            absent_name = "x" if admm_part == "B" else "y"
            self.assertTrue(
                hasattr(s, absent_name),
                f"{sname}: expected dummy attribute {absent_name}")
            dummy = getattr(s, absent_name)
            self.assertIn(
                dummy, root.surrogate_vardatas,
                f"{sname}: admm dummy {absent_name} missing from surrogates")

    @unittest.skipUnless(solver_available, "no solver available")
    def test_ef_partial_consensus(self):
        """EF must not pin partial-consensus vars to zero.

        With >2 admm subproblems, some consensus vars are owned only by a
        subset of subproblems; dummy fixed-to-0 stand-ins are created for the
        others. PH weights those dummies at 0, but EF with the default
        nonant_for_fixed_vars=True had been enforcing hard nonant equalities
        against them, forcing the real owners to 0 and producing worse
        (higher-for-minimize) objectives than PH.

        The examples/stoch_distr/stoch_distr_ef.py driver hides this by
        passing nonant_for_fixed_vars=False to create_EF, so we drive the
        EF through generic_cylinders --EF (ExtensiveForm's default).

        Regression: with the fix, the EF objective should lie within the
        PH Lagrangian outer / xhatxbar inner bound envelope.
        """
        generic_cyl = os.path.join(_REPO_ROOT, 'mpisppy', 'generic_cylinders.py')
        target_dir = os.path.join(_REPO_ROOT, 'examples', 'stoch_distr')
        original_dir = os.getcwd()
        os.chdir(target_dir)
        try:
            clean_env = {k: v for k, v in os.environ.items()
                         if not k.startswith(('OMPI_', 'PMIX_', 'PMI_'))}

            ef_cmd = (
                f"python {generic_cyl} --module-name stoch_distr "
                f"--EF --EF-solver-name {solver_name} --stoch-admm "
                f"--num-admm-subproblems 4 --num-stoch-scens 3"
            ).split()
            ef_result = subprocess.run(ef_cmd, capture_output=True, text=True,
                                       env=clean_env, timeout=180)
            if ef_result.returncode != 0:
                raise RuntimeError(
                    f"EF run failed (rc={ef_result.returncode})\n"
                    f"STDOUT: {ef_result.stdout[-1000:]}\n"
                    f"STDERR: {ef_result.stderr[-1000:]}")
            ef_obj = None
            for line in ef_result.stdout.strip().split('\n'):
                if "EF objective" in line:
                    ef_obj = float(line.split(': ')[1])
            self.assertIsNotNone(ef_obj,
                                 f"Could not extract EF objective:\n{ef_result.stdout[-1000:]}")

            ph_cmd = (
                f"mpiexec -np 3 python -u -m mpi4py {generic_cyl} "
                f"--module-name stoch_distr --stoch-admm "
                f"--num-admm-subproblems 4 --num-stoch-scens 3 "
                f"--default-rho 10 --solver-name {solver_name} "
                f"--max-iterations 50 --lagrangian --xhatxbar "
                f"--rel-gap 0.01 --ensure-xhat-feas"
            ).split()
            ph_result = subprocess.run(ph_cmd, capture_output=True, text=True,
                                       env=clean_env, timeout=180)
            if ph_result.returncode != 0:
                raise RuntimeError(
                    f"PH run failed (rc={ph_result.returncode})\n"
                    f"STDOUT: {ph_result.stdout[-1000:]}\n"
                    f"STDERR: {ph_result.stderr[-1000:]}")

            # Grab the final iteration line (after the last "Iter." header).
            import re
            outer_bound = None
            inner_bound = None
            pattern = re.compile(
                r'\[\s*\d+\.\d+\]\s+\d+\s+(?:L\s*B?|B\s*L?)\s+([-.\d]+)\s+([-.\d]+)')
            for line in ph_result.stdout.strip().split('\n'):
                m = pattern.search(line)
                if m:
                    outer_bound = float(m.group(1))
                    inner_bound = float(m.group(2))
            self.assertIsNotNone(outer_bound,
                                 f"Could not extract PH bounds:\n{ph_result.stdout[-1000:]}")

            # For minimize, outer is the lower bound on opt, inner is the upper.
            # EF's optimal should sit in between (with a small slack).
            self.assertLessEqual(
                outer_bound, ef_obj + 0.01,
                msg=f"EF {ef_obj} below PH outer bound {outer_bound} "
                    f"(inner={inner_bound}) — partial-consensus EF regression")
            self.assertLessEqual(
                ef_obj, inner_bound + 0.02,
                msg=f"EF {ef_obj} above PH inner bound {inner_bound} "
                    f"(outer={outer_bound}) — partial-consensus EF regression")
        finally:
            os.chdir(original_dir)


class TestStochAdmmWrapperFirstStageHooks(unittest.TestCase):
    """first_stage_cost / first_stage_varlist hooks on the
    Stoch_AdmmWrapper API.

    When the hooks are supplied, the wrapper calls
    sputils.attach_root_node itself for each scenario, and
    scenario_creator must NOT call it.  When the hooks are absent
    (legacy path), scenario_creator must call attach_root_node
    itself.  Mixing the two (hooks + manual call, or no hooks + no
    call) raises with a clear message at scenario-construction time.
    """

    @staticmethod
    def _minimal_scenario_creator(call_attach):
        """Return a scenario_creator that either does or does not call
        sputils.attach_root_node, otherwise identical."""
        import pyomo.environ as pyo
        import mpisppy.utils.sputils as sputils

        def scenario_creator(sname, **kwargs):
            parts = sname.split("_")
            admm_part = parts[2]
            m = pyo.ConcreteModel()
            # consensus var owned per admm subproblem
            if admm_part == "A":
                m.x = pyo.Var(bounds=(0, 1))
                own = m.x
            else:
                m.y = pyo.Var(bounds=(0, 1))
                own = m.y
            m.fs = pyo.Var(bounds=(0, 1))  # original first-stage var
            m.FirstStageCost = pyo.Expression(expr=m.fs)
            m.obj = pyo.Objective(expr=own + m.fs, sense=pyo.minimize)
            # Stash for the hook to find (mirrors examples/stoch_distr
            # pattern):
            m._first_stage_vars = [m.fs]
            if call_attach:
                sputils.attach_root_node(m, m.FirstStageCost, [m.fs])
                m._mpisppy_probability = "uniform"
            return m

        return scenario_creator

    @staticmethod
    def _hooks():
        def first_stage_cost(s):
            return s.FirstStageCost

        def first_stage_varlist(s):
            return s._first_stage_vars

        return first_stage_cost, first_stage_varlist

    @staticmethod
    def _common_kwargs():
        admm_names = ["A", "B"]
        stoch_names = ["S1", "S2"]
        all_names = [f"ADMM_STOCH_{a}_{s}"
                     for s in stoch_names for a in admm_names]

        def split(sname):
            parts = sname.split("_")
            return parts[2], "_".join(parts[3:])

        consensus_vars = {"A": [("x", 1)], "B": [("y", 1)]}
        return {
            "all_admm_stoch_subproblem_scenario_names": all_names,
            "split_admm_stoch_subproblem_scenario_name": split,
            "admm_subproblem_names": admm_names,
            "stoch_scenario_names": stoch_names,
            "consensus_vars": consensus_vars,
            "n_cylinders": 1,
            "scenario_creator_kwargs": {},
        }

    def test_hooks_path_attaches_root_node(self):
        """With hooks defined and scenario_creator NOT calling
        attach_root_node, the wrapper must attach the root itself and
        produce a valid _mpisppy_node_list with the consensus stage
        appended."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        admm = Stoch_AdmmWrapper(
            options={},
            scenario_creator=self._minimal_scenario_creator(call_attach=False),
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **self._common_kwargs(),
        )
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            self.assertTrue(hasattr(s, "_mpisppy_node_list"),
                            f"{sname}: node list missing")
            # The wrapper appends an ADMM stage; expect 2 nodes for the
            # 2-stage-origin case (root + admm-consensus).
            self.assertEqual(
                len(s._mpisppy_node_list), 2,
                f"{sname}: expected 2-node list (root + admm), got "
                f"{[n.name for n in s._mpisppy_node_list]}")
            self.assertEqual(s._mpisppy_node_list[0].name, "ROOT")

    def test_hooks_and_legacy_paths_produce_same_node_list(self):
        """The first-stage-hooks path's B.2 auto-merge must produce
        the same final _mpisppy_node_list as the legacy path with
        first-stage entries pre-merged into consensus_vars by hand
        (i.e. wrap operates on equivalent inputs and yields equivalent
        wrapped scenarios)."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        admm_hooks = Stoch_AdmmWrapper(
            options={},
            scenario_creator=self._minimal_scenario_creator(call_attach=False),
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **self._common_kwargs(),
        )
        # Legacy path: pre-merge ("fs", 1) into both subproblems'
        # consensus_vars so the comparison is apples-to-apples.
        legacy_kwargs = self._common_kwargs()
        legacy_kwargs["consensus_vars"] = {
            sub: entries + [("fs", 1)]
            for sub, entries in legacy_kwargs["consensus_vars"].items()
        }
        admm_legacy = Stoch_AdmmWrapper(
            options={},
            scenario_creator=self._minimal_scenario_creator(call_attach=True),
            mpicomm=MPI.COMM_WORLD,
            **legacy_kwargs,
        )
        for sname in admm_hooks.local_admm_stoch_subproblem_scenarios:
            s_h = admm_hooks.local_admm_stoch_subproblem_scenarios[sname]
            s_l = admm_legacy.local_admm_stoch_subproblem_scenarios[sname]
            self.assertEqual(
                [n.name for n in s_h._mpisppy_node_list],
                [n.name for n in s_l._mpisppy_node_list],
                f"{sname}: node-name list differs between hooks and legacy")
            self.assertEqual(
                len(s_h._mpisppy_node_list[0].nonant_vardata_list),
                len(s_l._mpisppy_node_list[0].nonant_vardata_list),
                f"{sname}: root nonant count differs")

    def test_b2_auto_merge_first_stage_into_consensus_vars(self):
        """B.2: with first-stage hooks defined and a consensus_vars
        that does NOT pre-merge first-stage Vars, the wrapper merges
        them at construction time so the final self.consensus_vars
        carries both admm-consensus and first-stage entries."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        admm = Stoch_AdmmWrapper(
            options={},
            scenario_creator=self._minimal_scenario_creator(call_attach=False),
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **self._common_kwargs(),
        )
        self.assertIn(("fs", 1), admm.consensus_vars["A"])
        self.assertIn(("fs", 1), admm.consensus_vars["B"])
        # Existing admm entry preserved.
        self.assertIn(("x", 1), admm.consensus_vars["A"])
        self.assertIn(("y", 1), admm.consensus_vars["B"])

    def test_b2_per_subproblem_first_stage_vars(self):
        """B.2: when different ADMM subproblems carry different
        first-stage Vars (mirrors stoch_distr: each region has its
        own factory production decisions), each ADMM subproblem's
        consensus_vars must contain only ITS OWN first-stage Vars,
        not the other ADMM subproblem's."""
        import pyomo.environ as pyo
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        def per_sub_scenario_creator(sname, **kwargs):
            parts = sname.split("_")
            admm_part = parts[2]
            m = pyo.ConcreteModel()
            if admm_part == "A":
                m.x = pyo.Var(bounds=(0, 1))
                m.fs_A = pyo.Var(bounds=(0, 1))
                m._fs_vars = [m.fs_A]
                m.FirstStageCost = pyo.Expression(expr=m.fs_A)
                m.obj = pyo.Objective(expr=m.x + m.fs_A, sense=pyo.minimize)
            else:
                m.y = pyo.Var(bounds=(0, 1))
                m.fs_B = pyo.Var(bounds=(0, 1))
                m._fs_vars = [m.fs_B]
                m.FirstStageCost = pyo.Expression(expr=m.fs_B)
                m.obj = pyo.Objective(expr=m.y + m.fs_B, sense=pyo.minimize)
            return m

        def fs_cost(s):
            return s.FirstStageCost

        def fs_varlist(s):
            return s._fs_vars

        admm = Stoch_AdmmWrapper(
            options={},
            scenario_creator=per_sub_scenario_creator,
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **self._common_kwargs(),
        )
        self.assertIn(("fs_A", 1), admm.consensus_vars["A"])
        self.assertNotIn(("fs_B", 1), admm.consensus_vars["A"])
        self.assertIn(("fs_B", 1), admm.consensus_vars["B"])
        self.assertNotIn(("fs_A", 1), admm.consensus_vars["B"])

    def test_b2_first_stage_varlist_accepts_indexed_var(self):
        """B.2: a first_stage_varlist hook that returns an indexed Var
        container (rather than its individual VarData objects) must be
        expanded to per-index VarData names by the wrapper.  Without
        expansion the container name ('fs_idx') would land in
        varprob_dict but ScenarioNode would expand the container into
        VarData IDs that miss varprob_dict, tripping a KeyError in
        variable_probability."""
        import pyomo.environ as pyo
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        def indexed_fs_scenario_creator(sname, **kwargs):
            parts = sname.split("_")
            admm_part = parts[2]
            m = pyo.ConcreteModel()
            if admm_part == "A":
                m.x = pyo.Var(bounds=(0, 1))
                own = m.x
            else:
                m.y = pyo.Var(bounds=(0, 1))
                own = m.y
            m.fs_idx = pyo.Var([2025, 2026], bounds=(0, 1))
            m.FirstStageCost = pyo.Expression(
                expr=sum(m.fs_idx[t] for t in m.fs_idx))
            m.obj = pyo.Objective(
                expr=own + m.FirstStageCost, sense=pyo.minimize)
            # The hook returns the indexed container, not the VarDatas:
            m._first_stage_vars = [m.fs_idx]
            return m

        def fs_cost(s):
            return s.FirstStageCost

        def fs_varlist(s):
            return s._first_stage_vars

        admm = Stoch_AdmmWrapper(
            options={},
            scenario_creator=indexed_fs_scenario_creator,
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **self._common_kwargs(),
        )
        # Per-VarData names should be merged in, not the container name.
        for sub in ("A", "B"):
            self.assertIn(("fs_idx[2025]", 1), admm.consensus_vars[sub])
            self.assertIn(("fs_idx[2026]", 1), admm.consensus_vars[sub])
            self.assertNotIn(("fs_idx", 1), admm.consensus_vars[sub])
        # assign_variable_probs runs in __init__; reaching this point
        # means it did not KeyError on the indexed first-stage Var.
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            self.assertTrue(hasattr(s, "_mpisppy_node_list"),
                            f"{sname}: node list missing")

    def test_consensus_vars_accepts_var_objects(self):
        """B.1: consensus_vars may contain Pyomo Var/VarData objects in
        place of (or mixed with) name strings.  The wrapper's normalized
        consensus_vars and the resulting nonant lists must match the
        string-form result."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        sc = self._minimal_scenario_creator(call_attach=False)

        # Build a Var-flavored consensus_vars.  Each subproblem owns
        # exactly one consensus var per _minimal_scenario_creator.
        # Pyomo VarData holds its parent block via weakref, so we must
        # keep these sample scenarios alive across the wrapper call.
        sample_A = sc("ADMM_STOCH_A_S1")
        sample_B = sc("ADMM_STOCH_B_S1")
        cv_var = {"A": [(sample_A.x, 1)], "B": [(sample_B.y, 1)]}

        kw_str = self._common_kwargs()
        kw_var = dict(kw_str)
        kw_var["consensus_vars"] = cv_var

        admm_str = Stoch_AdmmWrapper(
            options={},
            scenario_creator=sc,
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **kw_str,
        )
        admm_var = Stoch_AdmmWrapper(
            options={},
            scenario_creator=sc,
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            **kw_var,
        )
        # sample_{A,B} can be released now; the wrapper has snapshotted
        # the names.
        del sample_A, sample_B

        self.assertEqual(admm_str.consensus_vars, admm_var.consensus_vars)
        for sname in admm_str.local_admm_stoch_subproblem_scenarios:
            s_s = admm_str.local_admm_stoch_subproblem_scenarios[sname]
            s_v = admm_var.local_admm_stoch_subproblem_scenarios[sname]
            self.assertEqual(
                [n.name for n in s_s._mpisppy_node_list],
                [n.name for n in s_v._mpisppy_node_list],
                f"{sname}: node list differs between str and var consensus_vars")

    def test_hooks_plus_manual_attach_errors(self):
        """Hooks defined AND scenario_creator also calls
        attach_root_node — error so the user knows the hooks make the
        manual call redundant (rather than silently overwriting it)."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        with self.assertRaises(RuntimeError) as cm:
            Stoch_AdmmWrapper(
                options={},
                scenario_creator=self._minimal_scenario_creator(call_attach=True),
                mpicomm=MPI.COMM_WORLD,
                first_stage_cost=fs_cost,
                first_stage_varlist=fs_varlist,
                **self._common_kwargs(),
            )
        self.assertIn("must NOT call", str(cm.exception))

    def test_no_hooks_no_attach_errors(self):
        """No hooks AND scenario_creator skips attach_root_node — error
        with a message pointing at both remediation options."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        with self.assertRaises(RuntimeError) as cm:
            Stoch_AdmmWrapper(
                options={},
                scenario_creator=self._minimal_scenario_creator(call_attach=False),
                mpicomm=MPI.COMM_WORLD,
                **self._common_kwargs(),
            )
        msg = str(cm.exception)
        self.assertIn("first_stage_cost", msg)
        self.assertIn("attach_root_node", msg)

    def test_wrapper_half_hooks_errors(self):
        """Exactly one hook passed to the wrapper — both-or-neither
        contract violated."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        for kw in (
            {"first_stage_cost": fs_cost},
            {"first_stage_varlist": fs_varlist},
        ):
            with self.assertRaises(RuntimeError) as cm:
                Stoch_AdmmWrapper(
                    options={},
                    scenario_creator=self._minimal_scenario_creator(call_attach=False),
                    mpicomm=MPI.COMM_WORLD,
                    **kw,
                    **self._common_kwargs(),
                )
            self.assertIn("must be defined together", str(cm.exception))

    def test_setup_stoch_admm_half_hooks_errors(self):
        """The setup_stoch_admm-level check fires before the wrapper-level
        check, naming the user's module in the error message."""
        import types
        from mpisppy.generic.admm import setup_stoch_admm

        fs_cost, _ = self._hooks()

        def admm_subproblem_names_creator(cfg):
            return ["A", "B"]

        def stoch_scenario_names_creator(cfg):
            return ["S1", "S2"]

        def admm_stoch_subproblem_scenario_names_creator(a_names, s_names):
            return [f"ADMM_STOCH_{a}_{s}" for s in s_names for a in a_names]

        def split_admm_stoch_subproblem_scenario_name(name):
            parts = name.split("_")
            return parts[2], "_".join(parts[3:])

        module = types.SimpleNamespace(
            __name__="fake_module",
            admm_subproblem_names_creator=admm_subproblem_names_creator,
            stoch_scenario_names_creator=stoch_scenario_names_creator,
            admm_stoch_subproblem_scenario_names_creator=admm_stoch_subproblem_scenario_names_creator,
            split_admm_stoch_subproblem_scenario_name=split_admm_stoch_subproblem_scenario_name,
            combining_names=lambda a, s: f"ADMM_STOCH_{a}_{s}",
            kw_creator=lambda cfg: {},
            consensus_vars_creator=lambda an, sn, **kw: {"A": [("x", 1)], "B": [("y", 1)]},
            scenario_creator=self._minimal_scenario_creator(call_attach=False),
            first_stage_cost=fs_cost,
            # first_stage_varlist intentionally missing
        )
        cfg = config.Config()
        cfg.add_to_config("branching_factors", description="",
                          domain=list, default=None)
        with self.assertRaises(RuntimeError) as cm:
            setup_stoch_admm(module, cfg, n_cylinders=1)
        msg = str(cm.exception)
        self.assertIn("fake_module", msg)
        self.assertIn("first_stage_cost", msg)
        self.assertIn("first_stage_varlist", msg)

    def test_setup_stoch_admm_with_bundles_half_hooks_errors(self):
        """The bundled path enforces the same both-or-neither contract."""
        import types
        from mpisppy.generic.admm import setup_stoch_admm_with_bundles

        fs_cost, _ = self._hooks()

        module = types.SimpleNamespace(
            __name__="fake_module",
            admm_subproblem_names_creator=lambda cfg: ["A", "B"],
            stoch_scenario_names_creator=lambda cfg: ["S1", "S2"],
            split_admm_stoch_subproblem_scenario_name=(
                lambda name: (name.split("_")[2],
                              "_".join(name.split("_")[3:]))),
            kw_creator=lambda cfg: {},
            consensus_vars_creator=(
                lambda an, sn, **kw: {"A": [("x", 1)], "B": [("y", 1)]}),
            combining_names=lambda a, s: f"ADMM_STOCH_{a}_{s}",
            scenario_creator=self._minimal_scenario_creator(call_attach=False),
            first_stage_cost=fs_cost,
            # first_stage_varlist intentionally missing
        )
        cfg = config.Config()
        cfg.add_to_config("scenarios_per_bundle", description="",
                          domain=int, default=2)
        with self.assertRaises(RuntimeError) as cm:
            setup_stoch_admm_with_bundles(module, cfg, n_cylinders=1)
        msg = str(cm.exception)
        self.assertIn("fake_module", msg)
        self.assertIn("first_stage_cost", msg)
        self.assertIn("first_stage_varlist", msg)

    # ----- B.3: advanced hooks for surrogate / EF-supplemental nonants -----

    @staticmethod
    def _scenario_creator_with_surrogate_and_ef_suppl():
        """Like _minimal_scenario_creator(call_attach=False), but each
        before-wrap scenario carries extra Vars m.z (a surrogate) and
        m.e (an EF-supplemental nonant) that B.3 forwards to
        attach_root_node via the new advanced hooks."""
        import pyomo.environ as pyo

        def sc(sname, **kwargs):
            parts = sname.split("_")
            admm_part = parts[2]
            m = pyo.ConcreteModel()
            if admm_part == "A":
                m.x = pyo.Var(bounds=(0, 1))
                own = m.x
            else:
                m.y = pyo.Var(bounds=(0, 1))
                own = m.y
            m.fs = pyo.Var(bounds=(0, 1))
            m.z = pyo.Var(bounds=(0, 1))   # surrogate nonant
            m.e = pyo.Var(bounds=(0, 1))   # EF-supplemental nonant
            m.FirstStageCost = pyo.Expression(expr=m.fs)
            m.obj = pyo.Objective(expr=own + m.fs, sense=pyo.minimize)
            m._first_stage_vars = [m.fs]
            m._surrogate_vars = [m.z]
            m._ef_suppl_vars = [m.e]
            return m

        return sc

    def test_advanced_hooks_forwarded(self):
        """B.3: when both advanced hooks are supplied, attach_root_node
        receives surrogate_nonant_list and nonant_ef_suppl_list, and
        the surrogate/EF-suppl Vars survive the wrapper's stage rewrite
        on the resulting wrapped scenario's root node."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        admm = Stoch_AdmmWrapper(
            options={},
            scenario_creator=self._scenario_creator_with_surrogate_and_ef_suppl(),
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            first_stage_surrogate_nonant_list=lambda s: s._surrogate_vars,
            first_stage_nonant_ef_suppl_list=lambda s: s._ef_suppl_vars,
            **self._common_kwargs(),
        )
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            root = s._mpisppy_node_list[0]
            self.assertIn(
                s.z, root.surrogate_vardatas,
                f"{sname}: surrogate Var m.z missing from root node")
            self.assertIn(
                s.e, root.nonant_ef_suppl_vardata_list,
                f"{sname}: EF-supplemental Var m.e missing from root node")

    def test_advanced_hook_alone_ok(self):
        """B.3: only one of the two advanced hooks supplied is fine
        (they are independent of each other; each is independent of
        the other but both depend on the two core hooks)."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        fs_cost, fs_varlist = self._hooks()
        admm = Stoch_AdmmWrapper(
            options={},
            scenario_creator=self._scenario_creator_with_surrogate_and_ef_suppl(),
            mpicomm=MPI.COMM_WORLD,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
            first_stage_surrogate_nonant_list=lambda s: s._surrogate_vars,
            # first_stage_nonant_ef_suppl_list NOT supplied
            **self._common_kwargs(),
        )
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            root = s._mpisppy_node_list[0]
            self.assertIn(s.z, root.surrogate_vardatas,
                          f"{sname}: surrogate Var missing")
            # No EF-suppl Vars attached.
            self.assertNotIn(s.e, root.nonant_ef_suppl_vardata_list)

    def test_advanced_hook_without_core_hooks_errors(self):
        """B.3: passing an advanced hook to Stoch_AdmmWrapper without
        also supplying first_stage_cost / first_stage_varlist is an
        error -- there is nothing for the wrapper to attach the
        advanced lists onto."""
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper
        from mpisppy import MPI

        for kw in (
            {"first_stage_surrogate_nonant_list": lambda s: []},
            {"first_stage_nonant_ef_suppl_list": lambda s: []},
        ):
            with self.assertRaises(RuntimeError) as cm:
                Stoch_AdmmWrapper(
                    options={},
                    scenario_creator=self._minimal_scenario_creator(call_attach=True),
                    mpicomm=MPI.COMM_WORLD,
                    **kw,
                    **self._common_kwargs(),
                )
            msg = str(cm.exception)
            self.assertIn("advanced hook", msg)
            self.assertIn("first_stage_cost", msg)

    def test_b2_probes_for_admm_subproblems_not_local_to_rank(self):
        """B.2 probe-fallback path: when this rank's slice of the
        before-wrap scenarios does not cover every ADMM subproblem,
        the wrapper must build a fresh probe before-wrap scenario for
        each missing ADMM subproblem and gather its first-stage Var
        names from there.  All ranks must end up with the same
        self.consensus_vars regardless of slice.

        Triggered with a synthetic 2-rank mpicomm (Get_size=2,
        Get_rank=0) and admm-major scenario ordering so the local
        slice covers only ADMM subproblem A; B's first-stage Var
        must arrive via the probe block.
        """
        from mpisppy.utils.stoch_admmWrapper import Stoch_AdmmWrapper

        class FakeMpicomm:
            def __init__(self, size, rank):
                self._size, self._rank = size, rank
            def Get_size(self):
                return self._size
            def Get_rank(self):
                return self._rank

        fs_cost, fs_varlist = self._hooks()
        base_sc = self._minimal_scenario_creator(call_attach=False)
        call_log = []
        def spy(sname, **kwargs):
            call_log.append(sname)
            return base_sc(sname, **kwargs)

        admm_names = ["A", "B"]
        stoch_names = ["S1", "S2"]
        # admm-major ordering: rank 0 will get the first two entries
        # (A_S1, A_S2) and miss B entirely.
        all_names = [f"ADMM_STOCH_{a}_{s}"
                     for a in admm_names for s in stoch_names]

        def split(sname):
            parts = sname.split("_")
            return parts[2], "_".join(parts[3:])

        admm = Stoch_AdmmWrapper(
            options={},
            all_admm_stoch_subproblem_scenario_names=all_names,
            split_admm_stoch_subproblem_scenario_name=split,
            admm_subproblem_names=admm_names,
            stoch_scenario_names=stoch_names,
            scenario_creator=spy,
            consensus_vars={"A": [("x", 1)], "B": [("y", 1)]},
            n_cylinders=1,
            mpicomm=FakeMpicomm(size=2, rank=0),
            scenario_creator_kwargs={},
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
        )

        # Only A_S1 and A_S2 are local on rank 0.
        local_names = list(admm.local_admm_stoch_subproblem_scenarios.keys())
        self.assertEqual(sorted(local_names),
                         ["ADMM_STOCH_A_S1", "ADMM_STOCH_A_S2"])

        # B's first-stage Var must have arrived via the probe.
        self.assertIn(("fs", 1), admm.consensus_vars["B"])
        self.assertIn(("fs", 1), admm.consensus_vars["A"])

        # scenario_creator was invoked 2x for local + 1x for B's
        # probe = 3 total.  The probe targets the first B-flavored
        # name in all_names, which is ADMM_STOCH_B_S1.
        self.assertEqual(len(call_log), 3,
                         f"expected 3 scenario_creator calls "
                         f"(2 local + 1 probe), got {call_log}")
        self.assertEqual(call_log[-1], "ADMM_STOCH_B_S1",
                         f"probe should target the first B-flavored "
                         f"name; got {call_log[-1]}")

    def test_setup_stoch_admm_advanced_without_core_errors(self):
        """B.3: the setup_stoch_admm-level discovery rejects a module
        that defines an advanced hook without the two core hooks."""
        import types
        from mpisppy.generic.admm import setup_stoch_admm

        module = types.SimpleNamespace(
            __name__="fake_module",
            admm_subproblem_names_creator=lambda cfg: ["A", "B"],
            stoch_scenario_names_creator=lambda cfg: ["S1", "S2"],
            admm_stoch_subproblem_scenario_names_creator=(
                lambda an, sn: [f"ADMM_STOCH_{a}_{s}" for s in sn for a in an]),
            split_admm_stoch_subproblem_scenario_name=(
                lambda name: (name.split("_")[2],
                              "_".join(name.split("_")[3:]))),
            combining_names=lambda a, s: f"ADMM_STOCH_{a}_{s}",
            kw_creator=lambda cfg: {},
            consensus_vars_creator=(
                lambda an, sn, **kw: {"A": [("x", 1)], "B": [("y", 1)]}),
            scenario_creator=self._minimal_scenario_creator(call_attach=False),
            # No first_stage_cost / first_stage_varlist; only an advanced hook.
            first_stage_surrogate_nonant_list=lambda s: [],
        )
        cfg = config.Config()
        cfg.add_to_config("branching_factors", description="",
                          domain=list, default=None)
        with self.assertRaises(RuntimeError) as cm:
            setup_stoch_admm(module, cfg, n_cylinders=1)
        msg = str(cm.exception)
        self.assertIn("fake_module", msg)
        self.assertIn("first_stage_surrogate_nonant_list", msg)
        self.assertIn("first_stage_cost", msg)


class TestStochAdmmDefaultNaming(unittest.TestCase):
    """Phase C: default combining / split / scen-names creator.

    Covers the three defaults in mpisppy.utils.stoch_admmWrapper that
    let a user omit the boilerplate inverse-pair plus the wrapper-
    and setup-level fallback paths that consume them.
    """

    def test_default_pair_round_trips(self):
        from mpisppy.utils.stoch_admmWrapper import (
            default_combining_names,
            default_split_admm_stoch_subproblem_scenario_name,
        )
        # Subproblem and stochastic-scenario names with underscores
        # exercise the delimiter; the legacy single-underscore
        # convention would have collided here.
        for admm_sub, stoch_scen in [
                ("Region1", "StochasticScenario1"),
                ("under_score_sub", "more_under_scores"),
                ("A", "B"),
        ]:
            name = default_combining_names(admm_sub, stoch_scen)
            a, s = default_split_admm_stoch_subproblem_scenario_name(name)
            self.assertEqual((a, s), (admm_sub, stoch_scen),
                             f"round-trip failed for {(admm_sub, stoch_scen)!r}")

    def test_default_split_rejects_malformed(self):
        from mpisppy.utils.stoch_admmWrapper import (
            default_split_admm_stoch_subproblem_scenario_name as split,
        )
        # No delimiter at all.
        with self.assertRaises(ValueError):
            split("Region1_StochasticScenario1")
        # Wrong leading sentinel.
        with self.assertRaises(ValueError):
            split("WRONG__ADMM__Region1__ADMM__S1")

    def test_default_scen_names_creator_nesting_and_combiner(self):
        """Default scen-names creator must use the same outer-stoch /
        inner-admm nesting as the canonical pattern (MPI rank
        assignment depends on it), and must honor a custom
        combining_fn override."""
        from mpisppy.utils.stoch_admmWrapper import (
            default_admm_stoch_subproblem_scenario_names_creator,
            default_combining_names,
        )
        admm = ["A", "B"]
        stoch = ["S1", "S2"]
        names = default_admm_stoch_subproblem_scenario_names_creator(
            admm, stoch)
        self.assertEqual(names, [
            default_combining_names("A", "S1"),
            default_combining_names("B", "S1"),
            default_combining_names("A", "S2"),
            default_combining_names("B", "S2"),
        ])
        # Custom combiner override.
        custom = default_admm_stoch_subproblem_scenario_names_creator(
            admm, stoch, combining_fn=lambda a, s: f"{a}|{s}")
        self.assertEqual(custom, ["A|S1", "B|S1", "A|S2", "B|S2"])

    def _hooks(self):
        # Reuse the minimal scenario fixture from
        # TestStochAdmmWrapperFirstStageHooks via a fresh instance
        # to keep this class self-contained.
        return TestStochAdmmWrapperFirstStageHooks._hooks()

    def _scenario_creator_using_default_split(self):
        """A scenario_creator that decodes its composite name via the
        package default split — what a user would write after the
        phase-D example migration."""
        import pyomo.environ as pyo
        from mpisppy.utils.stoch_admmWrapper import (
            default_split_admm_stoch_subproblem_scenario_name as split,
        )

        def sc(sname, **kwargs):
            admm_part, _ = split(sname)
            m = pyo.ConcreteModel()
            if admm_part == "A":
                m.x = pyo.Var(bounds=(0, 1))
                own = m.x
            else:
                m.y = pyo.Var(bounds=(0, 1))
                own = m.y
            m.fs = pyo.Var(bounds=(0, 1))
            m.FirstStageCost = pyo.Expression(expr=m.fs)
            m.obj = pyo.Objective(expr=own + m.fs, sense=pyo.minimize)
            m._first_stage_vars = [m.fs]
            return m

        return sc

    def test_wrapper_split_none_uses_default(self):
        """Stoch_AdmmWrapper(split=None, ...) must resolve to the
        package default split function."""
        from mpisppy.utils.stoch_admmWrapper import (
            Stoch_AdmmWrapper,
            default_combining_names,
        )
        from mpisppy import MPI

        admm_names = ["A", "B"]
        stoch_names = ["S1", "S2"]
        all_names = [default_combining_names(a, s)
                     for s in stoch_names for a in admm_names]
        fs_cost, fs_varlist = self._hooks()
        admm = Stoch_AdmmWrapper(
            options={},
            all_admm_stoch_subproblem_scenario_names=all_names,
            split_admm_stoch_subproblem_scenario_name=None,  # explicit opt-in
            admm_subproblem_names=admm_names,
            stoch_scenario_names=stoch_names,
            scenario_creator=self._scenario_creator_using_default_split(),
            consensus_vars={"A": [("x", 1)], "B": [("y", 1)]},
            n_cylinders=1,
            mpicomm=MPI.COMM_WORLD,
            scenario_creator_kwargs={},
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
        )
        # Every wrapped scenario decoded; node lists wired.
        self.assertEqual(
            sorted(admm.local_admm_stoch_subproblem_scenarios.keys()),
            sorted(all_names))
        for sname, s in admm.local_admm_stoch_subproblem_scenarios.items():
            self.assertTrue(hasattr(s, "_mpisppy_node_list"))

    def _module_omitting_naming(self):
        """A model module that omits combining_names, split, and
        admm_stoch_subproblem_scenario_names_creator — the phase-C
        target user shape."""
        import types
        fs_cost, fs_varlist = self._hooks()
        return types.SimpleNamespace(
            __name__="defaults_module",
            admm_subproblem_names_creator=lambda cfg: ["A", "B"],
            stoch_scenario_names_creator=lambda cfg: ["S1", "S2"],
            # NO combining_names, NO split, NO names_creator.
            kw_creator=lambda cfg: {},
            consensus_vars_creator=(
                lambda an, sn, **kw: {"A": [("x", 1)], "B": [("y", 1)]}),
            scenario_creator=self._scenario_creator_using_default_split(),
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
        )

    def test_setup_stoch_admm_uses_defaults(self):
        """setup_stoch_admm with a module that defines none of the
        three naming helpers builds wrapped names via the package
        defaults."""
        from mpisppy.generic.admm import setup_stoch_admm
        from mpisppy.utils.stoch_admmWrapper import default_combining_names

        module = self._module_omitting_naming()
        cfg = config.Config()
        cfg.add_to_config("branching_factors", description="",
                          domain=list, default=None)
        scen_creator, _, all_names, _ = setup_stoch_admm(
            module, cfg, n_cylinders=1)
        # Wrapped names follow the default delimiter convention.
        expected = [default_combining_names(a, s)
                    for s in ["S1", "S2"] for a in ["A", "B"]]
        self.assertEqual(all_names, expected)

    def test_discover_naming_half_pair_errors(self):
        """combining_names without split (or vice versa) violates the
        inverse-pair contract; _discover_naming_helpers must raise
        before the wrapper or downstream code runs."""
        import types
        from mpisppy.generic.admm import _discover_naming_helpers

        only_combining = types.SimpleNamespace(
            __name__="m_only_combining",
            combining_names=lambda a, s: f"{a}_{s}",
        )
        with self.assertRaises(RuntimeError) as cm:
            _discover_naming_helpers(only_combining)
        msg = str(cm.exception)
        self.assertIn("combining_names", msg)
        self.assertIn("split_admm_stoch_subproblem_scenario_name", msg)

        only_split = types.SimpleNamespace(
            __name__="m_only_split",
            split_admm_stoch_subproblem_scenario_name=lambda n: ("", ""),
        )
        with self.assertRaises(RuntimeError) as cm:
            _discover_naming_helpers(only_split)
        msg = str(cm.exception)
        self.assertIn("combining_names", msg)

    def test_setup_stoch_admm_custom_names_creator_without_split_errors(self):
        """A module that ships custom names via
        admm_stoch_subproblem_scenario_names_creator but omits the
        inverse pair would decode names through the default split at
        runtime and ValueError; catch the inconsistency at setup."""
        import types
        from mpisppy.generic.admm import setup_stoch_admm

        module = types.SimpleNamespace(
            __name__="m_names_no_split",
            admm_subproblem_names_creator=lambda cfg: ["A", "B"],
            stoch_scenario_names_creator=lambda cfg: ["S1", "S2"],
            admm_stoch_subproblem_scenario_names_creator=(
                lambda an, sn: [f"custom_{a}_{s}" for s in sn for a in an]),
            # NO combining_names, NO split.
            kw_creator=lambda cfg: {},
            consensus_vars_creator=(
                lambda an, sn, **kw: {"A": [("x", 1)], "B": [("y", 1)]}),
            scenario_creator=lambda *a, **kw: None,
        )
        cfg = config.Config()
        cfg.add_to_config("branching_factors", description="",
                          domain=list, default=None)
        with self.assertRaises(RuntimeError) as cm:
            setup_stoch_admm(module, cfg, n_cylinders=1)
        msg = str(cm.exception)
        self.assertIn("m_names_no_split", msg)
        self.assertIn("admm_stoch_subproblem_scenario_names_creator", msg)
        self.assertIn("split_admm_stoch_subproblem_scenario_name", msg)

    def test_setup_stoch_admm_custom_combining_drives_default_creator(self):
        """A module that defines a custom combining_names / split pair
        but no scen-names creator: the default scen-names creator
        composes with the user's combining_names (not the package
        default combiner)."""
        import types
        from mpisppy.generic.admm import setup_stoch_admm

        def combining(a, s):
            return f"my|{a}|{s}"

        def split(name):
            _, a, s = name.split("|")
            return a, s

        def scenario_creator(sname, **kwargs):
            import pyomo.environ as pyo
            a, _ = split(sname)
            m = pyo.ConcreteModel()
            if a == "A":
                m.x = pyo.Var(bounds=(0, 1))
                own = m.x
            else:
                m.y = pyo.Var(bounds=(0, 1))
                own = m.y
            m.fs = pyo.Var(bounds=(0, 1))
            m.FirstStageCost = pyo.Expression(expr=m.fs)
            m.obj = pyo.Objective(expr=own + m.fs, sense=pyo.minimize)
            m._first_stage_vars = [m.fs]
            return m

        fs_cost, fs_varlist = self._hooks()
        module = types.SimpleNamespace(
            __name__="m_custom_combining",
            admm_subproblem_names_creator=lambda cfg: ["A", "B"],
            stoch_scenario_names_creator=lambda cfg: ["S1", "S2"],
            combining_names=combining,
            split_admm_stoch_subproblem_scenario_name=split,
            kw_creator=lambda cfg: {},
            consensus_vars_creator=(
                lambda an, sn, **kw: {"A": [("x", 1)], "B": [("y", 1)]}),
            scenario_creator=scenario_creator,
            first_stage_cost=fs_cost,
            first_stage_varlist=fs_varlist,
        )
        cfg = config.Config()
        cfg.add_to_config("branching_factors", description="",
                          domain=list, default=None)
        _, _, all_names, _ = setup_stoch_admm(module, cfg, n_cylinders=1)
        # Outer stoch, inner admm; combining is the user's.
        self.assertEqual(all_names, [
            "my|A|S1", "my|B|S1", "my|A|S2", "my|B|S2",
        ])


if __name__ == '__main__':
    unittest.main()
