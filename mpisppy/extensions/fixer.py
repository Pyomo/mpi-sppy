# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
""" Code for WW style fixing of integers. This can be used
    as the only extension, but it could be called from a "master"
    extension.
"""
"""
Updated Feb 2019.
The options give lists of (id, sqrt_thresh, NB_lag, LB_lag, UB_lag) 
If the id is not there, don't fix (5-tuple).
There is one for iter0 and one for iterk.
For iter0, if the lag is not None, do it.
For other iters, use count;  None is also how you avoid.
"""

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import mpisppy.extensions.extension

def Fixer_tuple(xvar, th=None, nb=None, lb=None, ub=None):
    """ Somewhat self-documenting way to make a fixer tuple.
        For use in/by/for the so-called Reference Model.
    Args:
        xvar (Var): the Pyomo Var
        th (float): compared to sqrt(abs(xbar_sqared - xsquared_bar))
        nb: (int) None means ignore; for iter k, number of iters 
                  converged anywhere.
        lb: (int) None means ignore; for iter k, number of iters 
                  converged within th of xvar lower bound.
        ub: (int) None means ignore; for iter k, number of iters
                  converged within th of xvar upper bound

    Returns:
        tuple: a tuple to be appended to the iter0 or iterk list of tuples
    """
    if th is None and nb is None and lb is None and ub is None:
        print ("warning: Fixer_tuple called for Var=", xvar.name,
               "but no arguments were given")
    if th is None:
        th = 0
    if nb is not None and lb is not None and nb < lb:
        print ("warning: Fixer_tuple called for Var=", xvar.name,
               "with nb < lb, which means lb will be ignored.")
    if  nb is not None and ub is not None and nb < ub:
        print ("warning: Fixer_tuple called for Var=", xvar.name,
               "with nb < ub, which means ub will be ignored.")
        
    return (id(xvar), th, nb, lb, ub)

class Fixer(mpisppy.extensions.extension.Extension):

    def __init__(self, ph):
        self.ph = ph
        self.rank = self.ph.rank
        self.rank0 = self.ph.rank0
        self.PHoptions = ph.PHoptions
        self.fixeroptions = self.PHoptions["fixeroptions"] # required
        self.verbose = self.PHoptions["verbose"] \
                       or self.fixeroptions["verbose"]
        # this function is scenario specific (takes a scenario as an arg)
        self.id_fix_list_fct = self.fixeroptions["id_fix_list_fct"]
        self.dprogress = ph.PHoptions["display_progress"]
        self.fixed_so_far = 0
        self.boundtol = self.fixeroptions["boundtol"]

    def populate(self, local_scenarios):
        # [(ndn, i)] = iter count (i indexes into nonantlist)
        self.local_scenarios = local_scenarios

        self.iter0_fixer_tuples = {} # caller data
        self.fixer_tuples = {} # caller data
        self.threshold = {}  
        self.iter0_threshold = {}  
        # This count dict drives the loops later
        # NOTE: every scenario has a list.
        for k,s in self.local_scenarios.items():
            if hasattr(s, "_PySP_conv_iter_count"):
                raise RuntimeError("Scenario has _PySP_conv_iter_count")
            s._PySP_conv_iter_count = {} # key is (ndn, i)

            self.iter0_fixer_tuples[s], self.fixer_tuples[s] = \
                    self.id_fix_list_fct(s)

            if self.iter0_fixer_tuples[s] is not None:
                for (varid, th, nb, lb, ub) in self.iter0_fixer_tuples[s]:
                    (ndn, i) = s._varid_to_nonant_index[varid] #
                    if (ndn, i) not in self.iter0_threshold:
                        self.iter0_threshold[(ndn, i)] = th
                    else:
                        if th != self.iter0_threshold[(ndn, i)]:
                            print (s.name, ndn, i, th)
                            raise RuntimeError("Attempt to vary iter0 fixer "+\
                                               "threshold across scenarios.")
            if self.fixer_tuples[s] is not None:
                for (varid, th, nb, lb, ub) in self.fixer_tuples[s]:
                    (ndn, i) = s._varid_to_nonant_index[varid]
                    s._PySP_conv_iter_count[(ndn, i)] = 0
                    if (ndn, i) not in self.threshold:
                        self.threshold[(ndn, i)] = th
                    else:
                        if th != self.threshold[(ndn, i)]:
                            print (s.name, ndn, i, th)
                            raise RuntimeError("Attempt to vary fixer "+\
                                               "threshold across scenarios")

    def _vb(self, str):
        if self.verbose and self.rank == 0:
            print ("(rank0) " + str)

    def _dp(self, str):
        if (self.dprogress or self.verbose) and self.rank == 0:
            print ("(rank0) " + str)

    def _update_fix_counts(self):
        nodesdone = []  # avoid multiple updates of a node's Vars
        for k,s in self.local_scenarios.items():
            for ndn_i, xvar in s._nonant_indexes.items():
                if xvar.is_fixed():
                    continue
                xb = pyo.value(s._xbars[ndn_i])
                diff = xb * xb - pyo.value(s._xsqbars[ndn_i])
                tolval = self.threshold[ndn_i]
                tolval *= tolval  # the tol is on sqrt
                if -diff < tolval and diff < tolval:
                    ##print ("debug += diff, tolval", diff, tolval)
                    s._PySP_conv_iter_count[ndn_i] += 1
                else:
                    s._PySP_conv_iter_count[ndn_i] = 0
                    ##print ("debug reset fix diff, tolval", diff, tolval)
                    
    def iter0(self, local_scenarios):

        # first, do some persistent solver with bundles gymnastics        
        have_bundles = hasattr(self.ph, "saved_objs") # indicates bundles
        if have_bundles:
            subpname = next(iter(self.ph.local_subproblems))
            subp = self.ph.local_subproblems[subpname]
            solver_is_persistent = isinstance(subp._solver_plugin,
            pyo.pyomo.solvers.plugins.solvers.persistent_solver.PersistentSolver)
            if solver_is_persistent:
                vars_to_update = {}

        fixoptions = self.fixeroptions
        raw_fixed_so_far = 0   # count those fixed in each scenario
        for sname,s in self.ph.local_scenarios.items():
            if self.iter0_fixer_tuples[s] is None:
                print ("WARNING: No Iter0 fixer tuple for s.name=",s.name)
                return
            if not have_bundles:
                solver_is_persistent = isinstance(s._solver_plugin,
                    pyo.pyomo.solvers.plugins.solvers.persistent_solver.PersistentSolver)

            for (varid, th, nb, lb, ub) in self.iter0_fixer_tuples[s]:
                was_fixed = False
                try:
                    (ndn, i) = s._varid_to_nonant_index[varid]
                except:
                    print ("Are you trying to fix a Var that is not nonant?")
                    raise
                xvar = s._nonant_indexes[ndn,i]
                if not xvar.is_fixed():
                    xb = pyo.value(s._xbars[(ndn,i)])
                    diff = xb * xb - pyo.value(s._xsqbars[(ndn,i)])
                    tolval = self.iter0_threshold[(ndn, i)]
                    sqtolval = tolval*tolval  # the tol is on sqrt
                    if -diff > sqtolval or diff > sqtolval:
                        ##print ("debug0 NO fix diff, sqtolval", diff, sqtolval)
                        continue
                    else:
                        ##print ("debug0 fix diff, sqtolval", diff, sqtolval)
                        # if we are still here, it is converged
                        if nb is not None:
                            xvar.fix(xb)
                            self._vb("Fixed0 nb %s %s at %s" % \
                                     (s.name, xvar.name, str(xvar._value)))
                            was_fixed = True
                        elif lb is not None and xb - xvar.lb < self.boundtol:
                            xvar.fix(xvar.lb)
                            self._vb("Fixed0 lb %s %s at %s" % \
                                     (s.name, xvar.name, str(xvar._value)))
                            was_fixed = True
                        elif ub is not None and xvar.ub - xb < self.boundtol:
                            xvar.fix(xvar.ub)
                            self._vb("Fixed0 ub %s %s at %s" % \
                                     (s.name, xvar.name, str(xvar._value)))
                            was_fixed = True

                    if was_fixed:
                        raw_fixed_so_far += 1
                        if not have_bundles and solver_is_persistent:
                            s._solver_plugin.update_var(xvar)
                        if have_bundles and solver_is_persistent:
                            if sname not in vars_to_update:
                                vars_to_update[sname] = []
                            vars_to_update[sname].append(xvar)

        if have_bundles and solver_is_persistent:
            for k,subp in self.ph.local_subproblems.items():
                subpnum = sputils.extract_num(k)
                rank_local = self.ph.rank
                for sname in self.ph.names_in_bundles[rank_local][subpnum]:
                    if sname in vars_to_update:
                        for xvar in vars_to_update[sname]:
                            subp._solver_plugin.update_var(xvar)
                        
        self.fixed_so_far += raw_fixed_so_far / len(local_scenarios)
        self._dp("Unique Vars fixed so far %s" % (self.fixed_so_far))
        if raw_fixed_so_far % len(local_scenarios) != 0:
            raise RuntimeError ("Variation in fixing across scenarios detected "
                                "in fixer.py (iter0)")
            # maybe to do mpicomm.abort()

    def iterk(self, PHIter):
        """ Before iter k>1 solves, but after x-bar update.
        """
        # first, do some persistent solver with bundles gymnastics        
        have_bundles = hasattr(self.ph, "saved_objs") # indicates bundles
        if have_bundles:
            subpname = next(iter(self.ph.local_subproblems))
            subp = self.ph.local_subproblems[subpname]
            solver_is_persistent = isinstance(subp._solver_plugin,
            pyo.pyomo.solvers.plugins.solvers.persistent_solver.PersistentSolver)
            if solver_is_persistent:
                vars_to_update = {}

        fixoptions = self.fixeroptions
        raw_fixed_so_far = 0
        self._update_fix_counts()
        for sname,s in self.local_scenarios.items():
            if self.fixer_tuples[s] is None:
                print ("MAJOR WARNING: No Iter k fixer tuple for s.name=",s.name)
                return
            if not have_bundles:
                solver_is_persistent = isinstance(s._solver_plugin, pyo.pyomo.solvers.plugins.solvers.persistent_solver.PersistentSolver)
            for (varid, th, nb, lb, ub) in self.fixer_tuples[s]:
                was_fixed = False
                try:
                    (ndn, i) = s._varid_to_nonant_index[varid]
                except:
                    print ("Are you trying to fix a Var that is not nonant?")
                    raise
                tolval = self.threshold[(ndn, i)]
                xvar = s._nonant_indexes[ndn,i]
                if not xvar.is_fixed():
                    xb = pyo.value(s._xbars[(ndn,i)])
                    fx = s._PySP_conv_iter_count[(ndn,i)]
                    if fx > 0:
                        xbar = pyo.value(s._xbars[(ndn,i)])
                        was_fixed = False
                        if  nb is not None and nb <= fx:
                            xvar.fix(xbar)
                            self._vb("Fixed nb %s %s at %s" % \
                                     (s.name, xvar.name, str(xvar._value)))
                            was_fixed = True
                        elif lb is not None and lb < fx \
                             and xb - xvar.lb < self.boundtol:
                            xvar.fix(xvar.lb)
                            self._vb("Fixed lb %s %s at %s" % \
                                     (s.name, xvar.name, str(xvar._value)))
                            was_fixed = True
                        elif ub is not None and ub < fx \
                             and xvar.ub - xb < self.boundtol:
                            xvar.fix(xvar.ub)
                            self._vb("Fixed ub %s %s at %s" % \
                                     (s.name, xvar.name, str(xvar._value)))
                            was_fixed = True

                    if was_fixed:
                        raw_fixed_so_far += 1
                        if not have_bundles and solver_is_persistent:
                            s._solver_plugin.update_var(xvar)
                        if have_bundles and solver_is_persistent:
                            if sname not in vars_to_update:
                                vars_to_update[sname] = []
                            vars_to_update[sname].append(xvar)


        if have_bundles and solver_is_persistent:
            for k,subp in self.ph.local_subproblems.items():
                subpnum = sputils.extract_num(k)
                rank_local = self.ph.rank
                for sname in self.ph.names_in_bundles[rank_local][subpnum]:
                    if sname in vars_to_update:
                        for xvar in vars_to_update[sname]:
                            subp._solver_plugin.update_var(xvar)
                        
        self.fixed_so_far += raw_fixed_so_far / len(self.local_scenarios)
        self._dp("Unique Vars fixed so far %s" % (self.fixed_so_far))
        if raw_fixed_so_far % len(self.local_scenarios) != 0:
            raise RuntimeError ("Variation in fixing across scenarios detected "
                                "in fixer.py")

    def pre_iter0(self):
        return
                                        
    def post_iter0(self):
        """ initialize data structures; that's all we can do at this point
        """
        self.populate(self.ph.local_scenarios)

    def miditer(self):
        """ Check for fixing before in the middle of PHIter (after
        the xbar update for PHiter-1).
        """
        PHIter = self.ph._PHIter
        if PHIter == 1:  # before iter 1 solves
            self.iter0(self.ph.local_scenarios)
        else:
            self.iterk(PHIter)

    def enditer(self):
        return

    def post_everything(self):
        self._dp("Final unique Vars fixed by fixer= %s" % \
                      (self.fixed_so_far))

