# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
import pyomo.environ as pyo
from pysp.scenariotree.instance_factory import ScenarioTreeInstanceFactory

from mpisppy.scenario_tree import ScenarioNode as mpisppyScenarioNode

def _get_cost_expression(model, cost_variable):
    return model.find_component(cost_variable[0])[cost_variable[1]]

def _get_nonant_list(model, variable_templates):
    nonant_list = []
    for varname, index_list in variable_templates.items():
        var = model.find_component(varname)
        for index in index_list:
            if type(index) is tuple:
                indices = tuple(idx if idx != '*' else slice(None) for idx in index)
                for vardata in var.__getitem__(indices):
                    nonant_list.append(vardata)
            elif index == '*':
                for vardata in var.values():
                    nonant_list.append(vardata)
            else:
                nonant_list.append(var.__getitem__(index))
    return nonant_list

def _get_nodenames(root_node):
    root_node._mpisppy_stage = 1
    root_node._mpisppy_name = "ROOT"
    root_node._mpisppy_parent_name = None
    nodenames = ["ROOT"]
    _add_next_stage(root_node, nodenames)
    return nodenames

def _add_next_stage(node, nodenames):
    for idx, child in enumerate(node.children):
        child._mpisppy_name = node._mpisppy_name + f"_{idx}"
        child._mpisppy_stage = node._mpisppy_stage + 1
        child._mpisppy_parent_name = node._mpisppy_name
        nodenames.append(child._mpisppy_name)
        _add_next_stage(child, nodenames)


class PySPModel:
    """A class for instantiating PySP models for use in mpisppy. 

    Args:
        model: The reference scenario model. Can be set
            to Pyomo model or the name of a file
            containing a Pyomo model. For historical
            reasons, this argument can also be set to a
            directory name where it is assumed a file
            named ReferenceModel.py exists.
        scenario_tree: The scenario tree. Can be set to
            a Pyomo model, a file containing a Pyomo
            model, or a .dat file containing data for an
            abstract scenario tree model representation,
            which defines the structure of the scenario
            tree. It can also be a .py file that
            contains a networkx scenario tree or a
            networkx scenario tree object.  For
            historical reasons, this argument can also
            be set to a directory name where it is
            assumed a file named ScenarioStructure.dat
            exists.
        data_dir: Directory containing .dat files necessary
            for building the scenario instances
            associated with the scenario tree. This
            argument is required if no directory
            information can be extracted from the first
            two arguments and the reference model is an
            abstract Pyomo model. Otherwise, it is not
            required or the location will be inferred
            from the scenario tree location (first) or
            from the reference model location (second),
            where it is assumed the data files reside in
            the same directory.

    Properties:
      all_scenario_names (list):
                    A list of scenario names based on the pysp model for
                    use in mpisppy
      all_node_names (list):
                    A list of all node names based on the pysp model for
                    use in mpisppy
      scenario_creator (fct):
                    A scenario creator function based on the pysp model
                    for use in mpisppy
      scenario_denouement (fct):
                    A blank scenario_denouement function for use in mpisppy
    """
    def __init__(self,
                 model,
                 scenario_tree,
                 data_dir=None):

        self._scenario_tree_instance_factory = \
                ScenarioTreeInstanceFactory(model, scenario_tree, data=data_dir)
        self._pysp_scenario_tree = self._scenario_tree_instance_factory.generate_scenario_tree()

        ## get the things out of the tree model we need
        self._all_scenario_names = [s.name for s in self._pysp_scenario_tree.scenarios]
        
        ## check for more than two stages
        ## gripe if we see bundles
        if self._pysp_scenario_tree.bundles:
            logger.warning("Bundles are ignored in PySPModel")

        self._all_nodenames = _get_nodenames(self._pysp_scenario_tree.findRootNode())

    def scenario_creator_callback(self, scenario_name, **kwargs):
        ## fist, get the model out
        model = self._scenario_tree_instance_factory.construct_scenario_instance(
                scenario_name, self._pysp_scenario_tree)

        tree_scenario = self._pysp_scenario_tree.get_scenario(scenario_name)

        non_leaf_nodes = tree_scenario.node_list[:-1]

        for node in non_leaf_nodes:
            if node.is_leaf_node():
                raise Exception("Unexpected leaf node")

            node._mpisppy_cost_expression = _get_cost_expression(model, node.stage._cost_variable)
            node._mpisppy_nonant_list = _get_nonant_list(model, node.stage._variable_templates)
            node._mpisppy_nonant_ef_suppl_list = _get_nonant_list(model, node.stage._derived_variable_templates)

        ## add the things mpisppy expects to the model
        model._mpisppy_probability = tree_scenario.probability

        model._mpisppy_node_list = [mpisppyScenarioNode(
                                            name=node._mpisppy_name,
                                            cond_prob=node.conditional_probability,
                                            stage=node._mpisppy_stage,
                                            cost_expression=node._mpisppy_cost_expression,
                                            scen_name_list=None,
                                            nonant_list=node._mpisppy_nonant_list,
                                            scen_model=None,
                                            nonant_ef_suppl_list=node._mpisppy_nonant_ef_suppl_list,
                                            parent_name=node._mpisppy_parent_name,
                                    )
                                    for node in non_leaf_nodes
                                   ]

        for _ in model.component_data_objects(pyo.Objective, active=True, descend_into=True):
            break
        else: # no break
            print("Provided model has no objective; using PySP auto-generated objective")

            # attach PySP objective
            leaf_node = tree_scenario.node_list[-1]
            leaf_node._mpisppy_cost_expression = _get_cost_expression(model, leaf_node.stage._cost_variable)

            if hasattr(model, "_PySPModel_objective"):
                raise RuntimeError("provided model has attribute _PySPModel_objective")

            model._PySPModel_objective = pyo.Objective(expr=\
                    pyo.quicksum(node._mpisppy_cost_expression for node in tree_scenario.node_list))

        return model

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def close(self):
        self._scenario_tree_instance_factory.close()

    @property
    def scenario_creator(self):
        return lambda *args,**kwargs : self.scenario_creator_callback(*args,**kwargs)

    @property
    def all_scenario_names(self):
        return self._all_scenario_names

    @property
    def all_nodenames(self):
        return self._all_nodenames

    @property
    def scenario_denouement(self):
        return lambda *args,**kwargs: None
