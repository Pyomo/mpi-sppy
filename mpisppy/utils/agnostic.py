# Agnostic.py
# This software is distributed under the 3-clause BSD License.

"""
notes by dlw:
   - The cfg will happen to have an Agnostic object added to it so mpisppy code can find it;
   however, this code does not care about that.
   - If a function in mpisppy has two callouts (a rarity), then the kwargs need to distinguish.
"""

import inspect
import pyomo.environ as pyo
from mpisppy.utils import sputils
import mpisppy.utils.solver_spec as solver_spec

from mpisppy.extensions.fixer import Fixer

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

        
   def callout_agnostic(self, **kwargs):
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
	       raise RuntimeError(f"AML-agnostic module {self.module.__name__} is missing function {fname}")
	   fct(**kwargs)
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
            gd["probability"]: float prob or str "uniform"
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
        iis = [str(i) for (ndn,i) in gd["nonants"].keys()]  # is is reserved...
        s.nonantVars = pyo.Var(ndns, iis)
        
        # we don't really need an objective, but we do need a sense
        # note that other code may put W's and prox's on it
        s.Obj = pyo.Objective(expr=0, sense=gd["sense"])
        s._agnostic_dict = gd

        assert BFs is None, "We are only doing two stage for now"
        # (it would not be that hard to be multi-stage; see hydro.py)

        sputils.attach_root_node(s, s.Obj, [s.nonantVars])


############################################################################################################

def _farmer_parse_args():
    # create a config object and parse JUST FOR TESTING
    cfg = config.Config()
    
    cfg.num_scens_required()
    cfg.popular_args()
    cfg.two_sided_args()
    cfg.ph_args()    
    cfg.aph_args()    
    cfg.xhatlooper_args()
    cfg.fwph_args()
    cfg.lagrangian_args()
    cfg.lagranger_args()
    cfg.xhatshuffle_args()
    cfg.add_to_config("crops_mult",
                         description="There will be 3x this many crops (default 1)",
                         domain=int,
                         default=1)                
    cfg.add_to_config("use_norm_rho_updater",
                         description="Use the norm rho updater extension",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use-norm-rho-converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)
    cfg.add_to_config("run_async",
                         description="Run with async projective hedging instead of progressive hedging",
                         domain=bool,
                         default=False)
    cfg.add_to_config("use_norm_rho_converger",
                         description="Use the norm rho converger",
                         domain=bool,
                         default=False)

    cfg.parse_command_line("farmer_cylinders")
    return cfg


        
if __name__ == "__main__":
    # For use by developers doing ad hoc testing
    print("begin ad hoc main for agnostic.py")
    import farmer_agnostic
    cfg = _farmer_parse_args()
    A = Agnostic(farmer_agnostic, cfg)

