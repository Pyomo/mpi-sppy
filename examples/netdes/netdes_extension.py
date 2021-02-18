# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import numpy as np
import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
from mpisppy.extensions.extension import PHExtension

class NetworkDesignTracker(PHExtension):

    def __init__(self, ph):
        self.ph = ph
        self.rank = ph.rank

    def pre_iter0(self):
        pass

    def post_iter0(self):
        print('Just completed iteration 0')

    def miditer(self):
        ''' Currently hard-coded for no parallelism
        '''
        print('About to do the solve in iteration', self.ph._PHIter)
        sols = self.get_sol()
        print_sol(sols, print_all=False)

        if (self.ph._PHIter < 5):
            return

        models = self.ph.local_scenarios
        arb_model = get_arb_elt(models)
        edges = arb_model.edges
        num_scenarios = len(sols)

        bundling = not hasattr(arb_model, '_solver_plugin')
        if (bundling):
            arb_bundle_model = get_arb_elt(self.ph.local_subproblems)
            persistent = sputils.is_persistent(arb_bundle_model._solver_plugin)
        else:
            persistent = sputils.is_persistent(arb_model._solver_plugin)

        for edge in edges:
            if (arb_model.x[edge].fixed): # Fixed in one model = fixed in all
                continue

            sol_values = {name: sols[name][edge]['value'] for name in sols}
            total = sum(sol_values.values())
            if (total < 0.1) or (total > num_scenarios - 0.1):
                ''' All scenarios agree '''
                pass

            if (total > (num_scenarios//2) + 1.9):
                print(f'Fixing edge {edge} to 1')
                for (sname, model) in models.items():
                    model.x[edge].fix(1.0)
                    if (persistent):
                        if (bundling):
                            solver = self.get_solver(sname) 
                        else:
                            solver = model._solver_plugin
                        solver.update_var(model.x[edge])

        fixed_edges = [edge for edge in edges if arb_model.x[edge].fixed]
        print(f'Fixed {len(fixed_edges)} total edges')

    def get_solver(self, sname):
        ''' Move this to PHBase if it is successful 
            Need to add some checks for bundling first 
        '''
        rank = self.rank
        names = self.ph.names_in_bundles[rank]
        bunnum = None
        for (num, scens) in names.items():
            if (sname in scens):
                bunnum = num
                break
        if (bunnum is None):
            raise RuntimeError(f'Could not find {sname}')
        bname = f'rank{rank}bundle{bunnum}'
        return self.ph.local_subproblems[bname]._solver_plugin

    def enditer(self):
        ''' Variable fixing must be done in miditer, so that a solve takes
            place after the variables are fixed (otherwise, we could end up with
            and infeasible solution).
        '''
        pass

    def post_everything(self):
        pass

    def get_sol(self):
        ph = self.ph
        arb_model = get_arb_elt(ph.local_scenarios)
        edges = arb_model.edges
        res = {
                  name: {
                      edge: {
                          'value': pyo.value(model.x[edge]),
                          'fixed': model.x[edge].fixed,
                      } for edge in edges
                  } for (name, model) in ph.local_scenarios.items()
              }
        return res

# My helper functions here
def get_arb_elt(dictionary):
    ''' Return an arbitrary element of the dictionary '''
    if not (len(dictionary)):
        return None
    return next(iter(dictionary.values()))

def print_sol(sol_dict, print_all=True):
    edges = list(get_arb_elt(sol_dict).keys())
    for edge in edges:
        if (not print_all):
            tot = sum(sol_dict[sname][edge]['value'] for sname in sol_dict)
            if (tot >= 0.9): # At least one scenario wants to take the edge
                row = ''.join(['x' if sol_dict[sname][edge]['value'] > 0.1 else '.'
                                        for sname in sol_dict])
                row = f'{edge[0]:2d}-->{edge[1]:2d} ' + row
                fixed = [sol_dict[sname][edge]['fixed'] for sname in sol_dict]
                assert(all(fixed) or (not any(fixed)))
                if (fixed[0]):
                    row += ' <-- fixed'
                print(row)
        else:
            row = ''.join(['x' if sol_dict[sname][edge]['value'] > 0.1 else '.'
                                    for sname in sol_dict])
            row = f'{edge[0]:2d}-->{edge[1]:2d} ' + row
            fixed = [sol_dict[sname][edge]['fixed'] for sname in sol_dict]
            assert(all(fixed) or (not any(fixed)))
            if (fixed[0]):
                row += ' <-- fixed'
            print(row)

