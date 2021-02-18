# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
from math import inf, isfinite
from pyomo.core.expr.numeric_expr import LinearExpression
from mpisppy.cylinders.hub import PHHub
from mpisppy.cylinders.cross_scen_spoke import CrossScenarioCutSpoke

import numpy as np
import mpisppy.utils.sputils as sputils

class CrossScenarioHub(PHHub):
    def setup_hub(self):
        super().setup_hub()
        if self.opt.multistage:
            raise RuntimeError('CrossScenarioHub only supports '
                                'two-stage models at this time')
        idx = self.cut_gen_spoke_index
        self.all_nonants_and_etas = np.zeros(self.local_lengths[idx - 1] + 1)

        arb_scen = self.opt.local_scenarios[self.opt.local_scenario_names[0]]
        # get an arbitrary scenario for the 
        # number of nonant variables
        self.nonant_len = len(arb_scen._nonant_indexes)

        # save the best bounds so far
        self.best_inner_bound = inf
        self.best_outer_bound = -inf

        # helping the extension track cuts
        self.new_cuts = False

    def initialize_spoke_indices(self):
        super().initialize_spoke_indices()
        for (i, spoke) in enumerate(self.spokes):
            if spoke["spoke_class"] == CrossScenarioCutSpoke:
                self.cut_gen_spoke_index = i + 1

    def sync(self):
        super().sync()
        self.send_to_cross_cuts()
        self.get_from_cross_cuts()

    def get_from_cross_cuts(self):
        idx = self.cut_gen_spoke_index
        receive_buffer = np.empty(self.remote_lengths[idx - 1] + 1, dtype="d") # Must be doubles
        is_new = self.hub_from_spoke(receive_buffer, idx)
        if is_new:
            self.make_cuts(receive_buffer)

    def send_to_cross_cuts(self):
        idx = self.cut_gen_spoke_index

        # get the stuff we want to send
        self.opt._save_nonants()
        ci = 0  ## index to self.nonant_send_buffer

        # get all the nonants
        all_nonants_and_etas = self.all_nonants_and_etas
        for k, s in self.opt.local_scenarios.items():
            for xvar in s._nonant_indexes.values():
                all_nonants_and_etas[ci] = xvar._value
                ci += 1

        # get all the etas
        for k, s in self.opt.local_scenarios.items():
            for sn in self.opt.all_scenario_names:
                all_nonants_and_etas[ci] = s.eta[sn]._value
                ci += 1
        self.hub_to_spoke(all_nonants_and_etas, idx)


    def make_cuts(self, coefs):
        # take the coefficient array and assemble cuts accordingly

        # this should have already been set in the extension !
        opt = self.opt

        # rows are 
        # [ const, eta_coeff, *nonant_coeffs ]
        row_len = 1+1+self.nonant_len
        outer_iter = int(coefs[-1])

        bundling = opt.bundling
        if opt.bundling:
            for bn,b in opt.local_subproblems.items():
                persistent_solver = sputils.is_persistent(b._solver_plugin)
                ## get an arbitrary scenario
                s = opt.local_scenarios[b.scen_list[0]]
                for idx, k in enumerate(opt.all_scenario_names):
                    row = coefs[row_len*idx:row_len*(idx+1)]
                    # the row could be all zeros,
                    # which doesn't do anything
                    if (row == 0.).all():
                        continue
                    # rows are 
                    # [ const, eta_coeff, *nonant_coeffs ]
                    linear_const = row[0]
                    linear_coefs = list(row[1:])
                    linear_vars = [b.eta[k]]

                    for ndn_i in s._nonant_indexes:
                        ## for bundles, we add the constrains only
                        ## to the reference first stage variables
                        linear_vars.append(b.ref_vars[ndn_i])

                    cut_expr = LinearExpression(constant=linear_const, linear_coefs=linear_coefs,
                                                linear_vars=linear_vars)
                    b._benders_cuts[outer_iter, k] = (None, cut_expr, 0)
                    if persistent_solver:
                        b._solver_plugin.add_constraint(b._benders_cuts[outer_iter, k])

        else:
            for sn,s in opt.local_subproblems.items():
                persistent_solver = sputils.is_persistent(s._solver_plugin)
                for idx, k in enumerate(opt.all_scenario_names):
                    row = coefs[row_len*idx:row_len*(idx+1)]
                    # the row could be all zeros,
                    # which doesn't do anything
                    if (row == 0.).all():
                        continue
                    # rows are 
                    # [ const, eta_coeff, *nonant_coeffs ]
                    linear_const = row[0]
                    linear_coefs = list(row[1:])
                    linear_vars = [s.eta[k]]
                    linear_vars.extend(s._nonant_indexes.values())

                    cut_expr = LinearExpression(constant=linear_const, linear_coefs=linear_coefs,
                                                linear_vars=linear_vars)
                    s._benders_cuts[outer_iter, k] = (None, cut_expr, 0.)
                    if persistent_solver:
                        s._solver_plugin.add_constraint(s._benders_cuts[outer_iter, k])

        # NOTE: the LShaped code negates the objective, so
        #       we do the same here for consistency
        ib = self.BestInnerBound
        ob = self.BestOuterBound
        if not opt.is_minimizing:
            ib = -ib
            ob = -ob
        add_cut = (isfinite(ib) or isfinite(ob)) and \
                ((ib < self.best_inner_bound) or (ob > self.best_outer_bound))
        if add_cut:
            self.best_inner_bound = ib
            self.best_outer_bound = ob
            for sn,s in opt.local_subproblems.items():
                persistent_solver = sputils.is_persistent(s._solver_plugin)
                prior_outer_iter = list(s._ib_constr.keys())
                s._ib_constr[outer_iter] = (ob, s._EF_obj, ib)
                if persistent_solver:
                    s._solver_plugin.add_constraint(s._ib_constr[outer_iter])
                # remove other ib constraints (we only need the tightest)
                for it in prior_outer_iter:
                    if persistent_solver:
                        s._solver_plugin.remove_constraint(s._ib_constr[it])
                    del s._ib_constr[it]

        ## helping the extention track cuts
        self.new_cuts = True
