# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from pyomo.core.base.block import _BlockData, declare_custom_block
import pyomo.environ as pe
from pyomo.solvers.plugins.solvers.persistent_solver import PersistentSolver
import pyomo.contrib.benders.benders_cuts as bc

try:
    from mpi4py import MPI

    mpi4py_available = True
except:
    mpi4py_available = False
try:
    import numpy as np

    numpy_available = True
except:
    numpy_available = False
import logging

logger = logging.getLogger(__name__)


solver_dual_sign_convention = dict()
solver_dual_sign_convention['ipopt'] = -1
solver_dual_sign_convention['gurobi'] = -1
solver_dual_sign_convention['gurobi_direct'] = -1
solver_dual_sign_convention['gurobi_persistent'] = -1
solver_dual_sign_convention['cplex'] = -1
solver_dual_sign_convention['cplex_direct'] = -1
solver_dual_sign_convention['cplexdirect'] = -1
solver_dual_sign_convention['cplex_persistent'] = -1
solver_dual_sign_convention['glpk'] = -1
solver_dual_sign_convention['cbc'] = -1
solver_dual_sign_convention['xpress_direct'] = -1
solver_dual_sign_convention['xpress_persistent'] = -1


@declare_custom_block(name='LShapedCutGenerator')
class LShapedCutGeneratorData(bc.BendersCutGeneratorData):
    def __init__(self, component):
        super().__init__(component)
        # self.local_subproblem_count = 0
        # self.global_subproblem_count = 0

    def set_ls(self, ls):
        self.ls = ls
        self.global_subproblem_count = len(self.ls.all_scenario_names)
        self._subproblem_ndx_map = dict.fromkeys(range(len(self.ls.local_scenario_names)))
        for s in self._subproblem_ndx_map.keys():
            self._subproblem_ndx_map[s] = self.ls.all_scenario_names.index(self.ls.local_scenario_names[s])
        # print(self._subproblem_ndx_map)
        self.all_master_etas = list(self.ls.master.eta.values())

    def global_num_subproblems(self):
        return self.global_subproblem_count

    def add_subproblem(self, subproblem_fn, subproblem_fn_kwargs, master_eta, subproblem_solver='gurobi_persistent',
                       relax_subproblem_cons=False, subproblem_solver_options=None):
        # print(self._subproblem_ndx_map)
        # self.all_master_etas.append(master_eta)
        # self.global_subproblem_count += 1
        if subproblem_fn_kwargs['scenario_name'] in self.ls.local_scenario_names:
            # self.local_subproblem_count += 1
            self.master_etas.append(master_eta)
            subproblem, complicating_vars_map = subproblem_fn(**subproblem_fn_kwargs)
            self.subproblems.append(subproblem)
            self.complicating_vars_maps.append(complicating_vars_map)
            bc._setup_subproblem(subproblem, master_vars=[complicating_vars_map[i] for i in self.master_vars if
                                                       i in complicating_vars_map],
                              relax_subproblem_cons=relax_subproblem_cons)

            # self._subproblem_ndx_map[self.local_subproblem_count - 1] = self.global_subproblem_count - 1

            if isinstance(subproblem_solver, str):
                subproblem_solver = pe.SolverFactory(subproblem_solver)
            self.subproblem_solvers.append(subproblem_solver)
            if isinstance(subproblem_solver, PersistentSolver):
                subproblem_solver.set_instance(subproblem)
            if subproblem_solver_options:
                for k,v in subproblem_solver_options.items():
                    subproblem_solver.options[k] = v
