###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
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


if __name__ == '__main__':
    unittest.main()
