###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Agnostic.py
# This software is distributed under the 3-clause BSD License.

"""
notes by dlw:
   - The cfg will happen to have an Agnostic object added to it so mpisppy code can find it;
   however, this code does not care about that.
   - If a function in mpisppy has two callouts (a rarity), then the kwargs need to distinguish.
"""

import sys
import inspect
import pyomo.environ as pyo
from mpisppy.utils import sputils


#========================================
class Agnostic():
    """
    Args:
        module (python module): None or the module with the callout fcts, 
                                scenario_creator and other helper fcts.
        cfg (Config): controls 
    """

    def __init__(self, module, cfg):
        self.module = module
        self.cfg = cfg

        
    def callout_agnostic(self, kwargs):
        """ callout from mpi-sppy for AML-agnostic support
        Args:
           cfg (Config): the field "AML_agnostic" might contain a module with callouts
	   kwargs (dict): the keyword args for the callout function (e.g. scenario)
        Calls:
           a callout function that presumably has side-effects
        Returns:
           True if the callout was done and False if not
        Note:
           Throws an error if the module exists, but the fct is missing
        """
       
        if self.module is not None:
            fname = inspect.stack()[1][3] 
            fct = getattr(self.module, fname, None)
            if fct is None:
                raise RuntimeError(f"AML-agnostic module or object is missing function {fname}")
            try:
                fct(Ag=self, **kwargs)
            except Exception as e:
                print(f"ERROR: AML-agnostic module or object had an error when calling {fname}", file=sys.stderr)
                raise e
            return True
        else:
            return False

       
    def scenario_creator(self, sname):
        """ create scenario sname by calling guest language, then attach stuff
        Args:
            sname (str): the scenario name that usually ends with a number
        Returns:
            scenario (Pyomo concrete model): a skeletal pyomo model with
                                             a lot attached to it.
        Note:
            The python function scenario_creator in the module needs to
            return a dict that we will call gd.
            gd["scenario"]: the guest language model handle
            gd["nonants"]: dict [(ndn,i)]: guest language Var handle
            gd["nonant_names"]: dict [(ndn,i)]: str with name of variable
            gd["nonant_fixedness"]: dict [(ndn,i)]: indicator of fixed variable
            gd["nonant_start"]: dict [(ndn,i)]: float with starting value
            gd["probability"]: float prob or str "uniform"
            gd["obj_fct"]: the objective function from the guest
            gd["sense"]: pyo.minimize or pyo.maximize
            gd["BFs"]: scenario tree branching factors list or None
            (for two stage models, the only value of ndn is "ROOT";
             i in (ndn, i) is always just an index)
        """
        
        crfct = getattr(self.module, "scenario_creator", None)
        if crfct is None:
            raise RuntimeError(f"AML-agnostic module {self.module.__name__} is missing the scenario_creator function")
        kwfct = getattr(self.module, "kw_creator", None)
        if kwfct is not None:
           kwargs = kwfct(self.cfg)
           gd = crfct(sname, **kwargs)
        else:
            gd = crfct(sname)
        s = pyo.ConcreteModel(sname)

        ndns = [ndn for (ndn,i) in gd["nonants"].keys()]
        iis = [i for (ndn,i) in gd["nonants"].keys()]  # is is reserved...
        s.nonantVars = pyo.Var(ndns, iis)
        for idx,v  in s.nonantVars.items():
            v._value = gd["nonant_start"][idx]
            v.fixed = gd["nonant_fixedness"][idx]
        
        # we don't really need an objective, but we do need a sense
        # note that other code may put W's and prox's on it
        s.Obj = pyo.Objective(expr=0, sense=gd["sense"])
        s._agnostic_dict = gd

        assert gd["BFs"] is None, "We are only doing two stage for now"
        # (it would not be that hard to be multi-stage; see hydro.py)

        sputils.attach_root_node(s, s.Obj, [s.nonantVars])

        s._mpisppy_probability = gd["probability"]
        
        return s


############################################################################################################

        
if __name__ == "__main__":
    # For use by developers doing ad hoc testing
    print("no main")
