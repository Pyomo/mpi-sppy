# Agnostic.py
# This software is distributed under the 3-clause BSD License.

"""
notes by dlw:
   - The cfg will happen to have an Agnostic object added to it so mpisppy code can find it;
   however, this code does not care about that.
   - If a function in mpisppy has two callouts (a rarity), then the kwargs need to distinguish.
"""

import importlib
import inspect
import pyomo.environ as pyo
import argparse
import copy
import pyomo.common.config as pyofig
from mpisppy.utils import config
import mpisppy.utils.solver_spec as solver_spec

from mpisppy.extensions.fixer import Fixer

#========================================
class Agnostic():
    """
    Args:
        module (python module): None or the module with the callout fcts
        cfg (Config): controls 
    """

    def __init__(self, module, cfg):
        self.module = module
        self.cfg = cfg

   def callout_agnostic(self, **kwargs):
       """ callout for AML-agnostic support
       Args:
           cfg (Config): the field "AML_agnostic" might contain a module with callouts
	   kwargs (dict): the keywords args for the callout function
       Calls:
           a callout function that presumably has side-effects
       Returns:
           True if the callout was done and False if not
       """
       
       if self.module is not None:
	   name = inspect.stack()[1][3] 
           if not hasattr(self.module, name):
	       raise RuntimeError(f"AML-agnostic module is missing function {name}")
           fct = getattr(self.module, name)
	   fct(**kwargs)
	   return True
       else:
           return False
        

if __name__ == "__main__":
    # For use by developers doing ad hoc testing
    print("begin ad hoc main for agnostic.py")

    """
    Wow. How do you want to do _models_have_same_sense?
    I think you are going to have to use find_active_objective?
    But it would be better not to call out from there at all, maybe...
    so that means the create scenarios needs to be involved...

...
   pyomomodel.component_data_objects(Objective, active=True, descend_into=True)
is called from 15 different places in mpisppy.  

   So think first about the scenario creator

    """
