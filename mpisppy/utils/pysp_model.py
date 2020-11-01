# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import os.path
from inspect import signature
import pyomo.environ as pyo
from pyomo.pysp.scenariotree.tree_structure_model import \
        CreateAbstractScenarioTreeModel, ScenarioTreeModelFromNetworkX
from pyomo.pysp.phutils import isVariableNameIndexed,\
                                extractVariableNameAndIndex
from pyutilib.misc.import_file import import_file

from mpisppy.scenario_tree import ScenarioNode as mpisppyScenarioNode
from mpisppy.utils.sputils import create_EF as create_mpisppy_EF

hasnetworkx = False
try:
    import networkx
    hasnetworkx = True
except:
    pass

def _extract_name_and_index(name):
    '''Determine if a variable name is indexed, strip off the
       index, and return the name (without index) and the index.
       The index will be None if no index is found'''
    index = None
    if isVariableNameIndexed(name):
        name, index = extractVariableNameAndIndex(name)
    return name, index

class PySPModel:
    """A class for instantiating PySP models for use in mpisppy. 

    Args:
      scenario_creator (str or fct): 
                    is a path to the file that contains the scenario 
                    callback for concrete or the reference model for abstract,
                    or the scenario creator callback function
      tree_model (concrete model, or networkx tree, or str, or None): 
                    gives the tree as a concrete model 
                    (which could be a fct) or a valid networkx scenario tree
                    or path to AMPL data file.
      scenarios_dir (str or None):
                    For abstract models, gives the directory of the scenarios.
                    If None for an abstract model, will be inferred from the
                    directory of the tree_model
      scenario_creator_callback_name (str or None):
                    The name of the scenario creator callback function
                    if specifying an concrete model module in
                    scenario_creator
      tree_model_callback_name (str or None):
                    The name of the tree model creator callback function
                    if specifying an concrete model module in
                    scenario_creator

    Properties:
      all_scenario_names (list):
                    A list of scenario names base on the pysp model for
                    use in mpisppy
      scenario_creator (fct):
                    A scenario creator function based on the pysp model
                    for use in mpisppy
      scenario_denouement (fct):
                    A blank scenario_denouement function for use in mpisppy
    """
    def __init__(self, scenario_creator, tree_model=None,
            scenarios_dir=None,
            scenario_creator_callback_name=None,
            tree_model_callback_name=None):

        ## first, attempt to determine abstract vs concrete
        ## and get a scenario instance creator

        ## if callable, a instance creator
        if callable(scenario_creator):
            self.pysp_instance_creator = scenario_creator
            self.abstract = False
        else: ## else, either and abstract model or a module with a callback
            if scenario_creator_callback_name is None:
                scenario_creator_callback_name = 'pysp_instance_creation_callback'
            module = import_file(scenario_creator)
            if hasattr(module, scenario_creator_callback_name):
                self.pysp_instance_creator = \
                        getattr(module, scenario_creator_callback_name)
                self.abstract = False
            else:
                self.pysp_instance_creator = module.model.create_instance
                self.abstract = True

        ## attempt to find and construct a tree model
        if tree_model is None:
            if tree_model_callback_name is None:
                tree_model_callback_name = 'pysp_scenario_tree_model_callback'
            tree_maker = getattr(module, tree_model_callback_name)

            tree_model = tree_maker()
        ## if we get a *.dat file, assume the scenarios are here unless
        ## otherwise specified
        if isinstance(tree_model, str):
            self.tree_model = CreateAbstractScenarioTreeModel(\
                                ).create_instance(tree_model)
            self.scenarios_dir = os.path.dirname(tree_model)
        elif hasnetworkx and isinstance(tree_model, networkx.DiGraph):
            self.tree_model = ScenarioTreeModelFromNetworkX(tree_model)
        elif isinstance(tree_model, pyo.ConcreteModel):
            self.tree_model = tree_model
        else:
            raise RuntimeError("Type of tree_model {} unrecongized".format(
                                type(tree_model)))

        ## set the scenarios_dir if specified, but complain if 
        ## we don't have an abstract model
        if scenarios_dir is not None:
            if not self.abstract:
                raise RuntimeError("An abstract model is required for "
                        "scenarios_dir")
            self.scenarios_dir = scenarios_dir

        self._init()
    
    def _init(self):
        """Sets up things needed for mpisppy"""

        ## get the things out of the tree model we need
        tree_model = self.tree_model
        self._scenario_names = list(tree_model.Scenarios)
        
        ## check for more than two stages
        if len(tree_model.Stages) != 2:
            raise RuntimeError("Models for PySPModel must be 2-stage, "\
                    "found {} stages".format(len(tree_model.Stages)))
        ## gripe if we see bundles
        if len(tree_model.Bundles) != 0:
            logger.warning("Bundles are ignored in PySPModel")

        ## extract first stage information from the scenario tree
        first_stage = tree_model.Stages[1]
        first_stage_cost_str = tree_model.StageCost[first_stage].value 

        first_stage_cost_name, first_stage_cost_index = \
                _extract_name_and_index(first_stage_cost_str)

        ## this is the function that becomes the scenario creator callback
        def scenario_creator_callback(scenario_name, **kwargs):
            ## fist, get the model out
            if self.abstract:
                model = self.pysp_instance_creator(
                        os.path.join(self.scenarios_dir,scenario_name+'.dat'))
            else:
                ## try to support both callback types, but pass in Nones for 
                ## args that aren't scenario_name
                # TBD (use inspect to match kwargs with signature)
                try:
                    model = self.pysp_instance_creator(None,
                                                       scenario_name,
                                                       None,
                                                       **kwargs)
                except TypeError:
                    try:
                        model = self.pysp_instance_creator(scenario_name,
                                                           **kwargs)
                    except TypeError:
                        try:
                            model = self.pysp_instance_creator(scenario_name, None)
                            for key,val in kwargs.items():
                                if val is not None:
                                    print("WARNING: did not use {}={}".\
                                          format(key, val))
                        except:
                            print("signature=",
                                  str(signature(self.pysp_instance_creator)))
                            raise RuntimeError("Could not match callback")

            ## extract the first stage cost expression
            stage_cost_expr = getattr(model, first_stage_cost_name)
            if first_stage_cost_index is None:
                cost_expression = stage_cost_expr
            else:
                cost_expression = stage_cost_expr[first_stage_cost_index]

            ## now collect the nonant vars from the model based
            ## on the tree_model
            nonant_list = list()
            for var_str in tree_model.StageVariables[first_stage]:
                var_name, var_index = _extract_name_and_index(var_str)
                pyovar = getattr(model, var_name)
                ## if there's no index, we can append the var
                if var_index is None:
                    nonant_list.append(pyovar)
                ## if there is an index, it should be a single 
                ## index or the whole thing.
                ## NOTE: it would not be too difficult to enable
                ##        slicing, if necessary
                elif isinstance(var_index,tuple):
                    if '*' in var_index:
                        for i in var_index:
                            if i != '*':
                                raise RuntimeError("PySPModel does not "
                                                    "support slicing")
                        nonant_list.append(pyovar)
                ## If we only have one index
                elif var_index == '*':
                    nonant_list.append(pyovar)
                else: ## the index is not a slice
                    nonant_list.append(pyovar[var_index])

            ## get the probability from the tree_model
            scen_prob = tree_model.ConditionalProbability[
                            tree_model.ScenarioLeafNode[scenario_name].value
                            ].value

            ## add the things mpisppy expects to the model
            model.PySP_prob = scen_prob

            model._PySPnode_list = [mpisppyScenarioNode(
                                        name="ROOT",
                                        cond_prob=1.0,
                                        stage=1,
                                        cost_expression=cost_expression,
                                        scen_name_list=self._scenario_names,
                                        nonant_list = nonant_list,
                                        scen_model=model)
                                    ]
            return model

        ## add this function to the class
        self._mpisppy_instance_creator = scenario_creator_callback

    @property
    def scenario_creator(self):
        return self._mpisppy_instance_creator

    @property
    def all_scenario_names(self):
        return self._scenario_names

    @property
    def scenario_denouement(self):
        return lambda *args,**kwargs: None
