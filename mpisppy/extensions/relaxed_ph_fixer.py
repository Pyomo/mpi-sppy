###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
from mpisppy.extensions.extension import Extension
from mpisppy.utils.sputils import is_persistent

from mpisppy.cylinders.spwindow import Field

class RelaxedPHFixer(Extension):

    def __init__(self, spobj):
        super().__init__(spobj)

        ph_options = spobj.options
        ph_fixer_options = ph_options.get("relaxed_ph_fixer_options", {})
        self.bound_tol = ph_fixer_options.get("bound_tol", 1e-4)
        self.verbose = ph_fixer_options.get("verbose", True)
        self.debug = ph_fixer_options.get("debug", False)

        self._heuristic_fixed_vars = {}
        self._current_relaxed_nonants = None

    def pre_iter0(self):
        self._modeler_fixed_nonants = set()
        self.nonant_length = self.opt.nonant_length
        for k,s in self.opt.local_scenarios.items():
            for ndn_i, xvar in s._mpisppy_data.nonant_indices.items():
                if xvar.fixed:
                    self._modeler_fixed_nonants.add(ndn_i)

        for k,sub in self.opt.local_subproblems.items():
            self._heuristic_fixed_vars[k] = 0

    def iter0_post_solver_creation(self):
        # wait for relaxed iter0:
        if self.relaxed_nonant_buf.id() == 0:
            while not self.opt.spcomm.get_receive_buffer(self.relaxed_nonant_buf, Field.RELAXED_NONANT, self.relaxed_ph_spoke_index):
                continue
        self.relaxed_ph_fixing(self.relaxed_nonant_buf.value_array(), pre_iter0=True)

    def register_receive_fields(self):
        spcomm = self.opt.spcomm
        relaxed_ph_ranks = spcomm.fields_to_ranks[Field.RELAXED_NONANT]
        assert len(relaxed_ph_ranks) == 1
        index = relaxed_ph_ranks[0]

        self.relaxed_ph_spoke_index = index

        self.relaxed_nonant_buf = spcomm.register_recv_field(
            Field.RELAXED_NONANT,
            self.relaxed_ph_spoke_index,
        )

        return

    def miditer(self):
        self.opt.spcomm.get_receive_buffer(
            self.relaxed_nonant_buf,
            Field.RELAXED_NONANT,
            self.relaxed_ph_spoke_index,
        )
        self.relaxed_ph_fixing(self.relaxed_nonant_buf.value_array(), pre_iter0=False)
        return

    def relaxed_ph_fixing(self, relaxed_solution, pre_iter0 = False):

        for k, sub in self.opt.local_subproblems.items():
            raw_fixed_this_iter = 0
            persistent_solver = is_persistent(sub._solver_plugin)
            for sn in sub.scen_list:
                s = self.opt.local_scenarios[sn]
                for ci, (ndn_i, xvar) in enumerate(s._mpisppy_data.nonant_indices.items()):
                    if ndn_i in self._modeler_fixed_nonants:
                        continue
                    if xvar in s._mpisppy_data.all_surrogate_nonants:
                        continue
                    relaxed_val = relaxed_solution[ci]
                    xvar_value = xvar._value
                    update_var = False
                    if not pre_iter0 and xvar.fixed:
                        if (relaxed_val - xvar.lb) > self.bound_tol and (xvar.ub - relaxed_val) > self.bound_tol:
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            if self.debug and self.opt.cylinder_rank == 0:
                                print(f"{k}: unfixing var {xvar.name}; {relaxed_val=} is off bounds {(xvar.lb, xvar.ub)}")
                        # in case somebody else unfixs a variable in another rank...
                        xb = s._mpisppy_model.xbars[ndn_i]._value
                        if abs(xb - xvar_value) > self.bound_tol:
                            xvar.unfix()
                            update_var = True
                            raw_fixed_this_iter -= 1
                            if self.debug and self.opt.cylinder_rank == 0:
                                print(f"{k}: unfixing var {xvar.name}; xbar {xb} differs from the fixed value {xvar_value}")
                    elif (relaxed_val - xvar.lb <= self.bound_tol) and (pre_iter0 or (xvar_value - xvar.lb <= self.bound_tol)):
                        xvar.fix(xvar.lb)
                        if self.debug and self.opt.cylinder_rank == 0:
                            print(f"{k}: fixing var {xvar.name} to lb {xvar.lb}; {relaxed_val=}, var value is {xvar_value}")
                        update_var = True
                        raw_fixed_this_iter += 1
                    elif (xvar.ub - relaxed_val <= self.bound_tol) and (pre_iter0 or (xvar.ub - xvar_value <= self.bound_tol)):
                        xvar.fix(xvar.ub)
                        if self.debug and self.opt.cylinder_rank == 0:
                            print(f"{k}: fixing var {xvar.name} to ub {xvar.ub}; {relaxed_val=}, var value is {xvar_value}")
                        update_var = True
                        raw_fixed_this_iter += 1

                    if update_var and persistent_solver:
                        sub._solver_plugin.update_var(xvar)

            # Note: might count incorrectly with bundling?
            self._heuristic_fixed_vars[k] += raw_fixed_this_iter
            if self.verbose:
                print(f"{k}: total unique vars fixed by heuristic: {int(round(self._heuristic_fixed_vars[k]))}/{self.nonant_length}")
