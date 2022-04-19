import pyomo.environ as pe
import farmer
import mpisppy.utils.sputils as sputils
from mpisppy.opt import ef, sc
import logging
from mpisppy import MPI
import sys


if MPI.COMM_WORLD.Get_rank() == 0:
    logging.basicConfig(level=logging.INFO)


"""
To run this example:

mpirun -np N python -m mpi4py schur_complement.py N
"""


def solve_with_extensive_form(scen_count):
    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]
    options = dict()
    options['solver'] = 'cplex_direct'
    scenario_kwargs = dict()
    scenario_kwargs['use_integer'] = False
    
    opt = ef.ExtensiveForm(options=options,
                           all_scenario_names=scenario_names,
                           scenario_creator=farmer.scenario_creator,
                           scenario_creator_kwargs=scenario_kwargs)
    results = opt.solve_extensive_form()
    opt.report_var_values_at_rank0()
    return opt


def solve_with_sc(scen_count, linear_solver=None):
    scenario_names = ['Scenario' + str(i) for i in range(scen_count)]
    options = dict()
    if linear_solver is not None:
        options['linalg'] = dict()
        options['linalg']['solver'] = linear_solver
    scenario_kwargs = dict()
    scenario_kwargs['use_integer'] = False
    opt = sc.SchurComplement(options=options,
                             all_scenario_names=scenario_names,
                             scenario_creator=farmer.scenario_creator,
                             scenario_creator_kwargs=scenario_kwargs)
    results = opt.solve()
    opt.report_var_values_at_rank0()
    return opt


if __name__ == '__main__':
    scen_count = int(sys.argv[1])
    # solve_with_extensive_form(scen_count)
    solve_with_sc(scen_count)
