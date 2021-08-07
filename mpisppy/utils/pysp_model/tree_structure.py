#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright 2017 National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and 
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain 
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
#
# This file was originally part of PySP and Pyomo, available: https://github.com/Pyomo/pysp
# Copied with modification from pysp/scenariotree/tree_structure.py

__all__ = ('ScenarioTreeNode',
           'ScenarioTreeStage',
           'Scenario',
           'ScenarioTreeBundle',
           'ScenarioTree')

import sys
import random
import copy
import math
import logging

try:
    from collections import OrderedDict
except ImportError:                         #pragma:nocover
    from ordereddict import OrderedDict

from pyomo.common.collections import ComponentMap
from pyomo.core import (value, minimize, maximize,
                        Var, Expression, Block,
                        Objective, SOSConstraint,
                        ComponentUID)
from pyomo.core.base.sos import _SOSConstraintData
from pyomo.repn import generate_standard_repn
from .phutils import (BasicSymbolMap,
                      indexToString,
                      isVariableNameIndexed,
                      extractVariableNameAndIndex,
                      extractComponentIndices,
                     )

logger = logging.getLogger("mpisppy.utils.pysp_model")

CUID_repr_version = 1

class _CUIDLabeler:
    def __init__(self):
        self._cuid_map = ComponentMap()

    def update_cache(self, block):
        self._cuid_map.update(
            ComponentUID.generate_cuid_string_map(
                block, repr_version=CUID_repr_version))

    def clear_cache(self):
        self._cuid_map = {}

    def __call__(self, obj):
        if obj in self._cuid_map:
            return self._cuid_map[obj]
        else:
            cuid = ComponentUID(obj).get_repr(version=1)
            self._cuid_map[obj] = cuid
            return cuid

class ScenarioTreeNode:

    """ Constructor
    """

    VARIABLE_FIXED = 0
    VARIABLE_FREED = 1

    def __init__(self, name, conditional_probability, stage):

        # self-explanatory!
        self._name = name

        # the stage to which this tree node belongs.
        self._stage = stage

        # defines the tree structure
        self._parent = None

        # a collection of ScenarioTreeNodes
        self._children = []

        # conditional on parent
        self._conditional_probability = conditional_probability

        # a collection of all Scenario objects passing through this
        # node in the tree
        self._scenarios = []

        # the cumulative probability of scenarios at this node.
        # cached for efficiency.
        self._probability = 0.0

        # a map between a variable name and a list of original index
        # match templates, specified as strings.  we want to maintain
        # these for a variety of reasons, perhaps the most important
        # being that for output purposes. specific indices that match
        # belong to the tree node, as that may be specific to a tree
        # node.
        self._variable_templates = {}
        self._derived_variable_templates = {}

        #
        # information relating to all variables blended at this node, whether
        # of the standard or derived varieties.
        #
        # maps id -> (name, index)
        self._variable_ids = {}
        # maps (name,index) -> id
        self._name_index_to_id = {}
        # maps id -> list of (vardata,probability) across all scenarios
        self._variable_datas = {}

        # keep track of the variable indices at this node, independent
        # of type.  this is useful for iterating. maps variable name
        # to a list of indices.
        self._variable_indices = {}

        # variables are either standard or derived - but not both.
        # partition the ids into two sets, as we deal with these
        # differently in algorithmic and reporting contexts.
        self._standard_variable_ids = set()
        self._derived_variable_ids = set()
        # A temporary solution to help wwphextension and other code
        # for when pyomo instances no longer live on the master node
        # when using PHPyro
        self._integer = set()
        self._binary = set()
        self._semicontinuous = set()

        # a tuple consisting of (1) the name of the variable that
        # stores the stage-specific cost in all scenarios and (2) the
        # corresponding index *string* - this is converted in the tree
        # node to a real index.
        # TODO: Change the code so that this is a ComponentUID string
        self._cost_variable = None

        # a list of _VarData objects, representing the cost variables
        # for each scenario passing through this tree node.
        # NOTE: This list actually contains tuples of
        #       (_VarData, scenario-probability) pairs.
        self._cost_variable_datas = []

        # node variables ids that are fixed (along with the value to fix)
        self._fixed = {}


    @property
    def name(self):
        return self._name

    @property
    def stage(self):
        return self._stage

    @property
    def parent(self):
        return self._parent

    @property
    def children(self):
        return tuple(self._children)

    @property
    def scenarios(self):
        return self._scenarios

    @property
    def conditional_probability(self):
        return self._conditional_probability

    @property
    def probability(self):
        return self._probability

    #
    # a simple predicate to check if this tree node belongs to the
    # last stage in the scenario tree.
    #
    def is_leaf_node(self):

        return self._stage.is_last_stage()

    #
    # a utility to determine if the input variable name/index pair is
    # a derived variable.
    #
    def is_derived_variable(self, variable_name, variable_index):
        return (variable_name, variable_index) in self._name_index_to_id


class ScenarioTreeStage:

    """ Constructor
    """
    def __init__(self):

        self._name = ""

        # a collection of ScenarioTreeNode objects associated with this stage.
        self._tree_nodes = []

        # the parent scenario tree for this stage.
        self._scenario_tree = None

        # a map between a variable name and a list of original index
        # match templates, specified as strings.  we want to maintain
        # these for a variety of reasons, perhaps the most important
        # being that for output purposes. specific indices that match
        # belong to the tree node, as that may be specific to a tree
        # node.
        self._variable_templates = {}

        # same as above, but for derived stage variables.
        self._derived_variable_templates = {}

        # a tuple consisting of (1) the name of the variable that
        # stores the stage-specific cost in all scenarios and (2) the
        # corresponding index *string* - this is converted in the tree
        # node to a real index.
        self._cost_variable = None

    @property
    def name(self):
        return self._name

    @property
    def nodes(self):
        return self._tree_nodes

    @property
    def scenario_tree(self):
        return self._scenario_tree

    #
    # a simple predicate to check if this stage is the last stage in
    # the scenario tree.
    #
    def is_last_stage(self):

        return self == self._scenario_tree._stages[-1]

class Scenario:

    """ Constructor
    """
    def __init__(self):

        self._name = None
        # allows for construction of node list
        self._leaf_node = None
        # sequence from parent to leaf of ScenarioTreeNodes
        self._node_list = []
        # the unconditional probability for this scenario, computed from the node list
        self._probability = 0.0
        # the Pyomo instance corresponding to this scenario.
        self._instance = None
        self._instance_cost_expression = None
        self._instance_objective = None
        self._instance_original_objective_object = None
        self._objective_sense = None
        self._objective_name = None

        # The value of the (possibly augmented) objective function
        self._objective = None
        # The value of the original objective expression
        # (which should be the sum of the stage costs)
        self._cost = None
        # The individual stage cost values
        self._stage_costs = {}
        # The value of the ph weight term piece of the objective (if it exists)
        self._weight_term_cost = None
        # The value of the ph proximal term piece of the objective (if it exists)
        self._proximal_term_cost = None
        # The value of the scenariotree variables belonging to this scenario
        # (dictionary nested by node name)
        self._x = {}
        # The value of the weight terms belonging to this scenario
        # (dictionary nested by node name)
        self._w = {}
        # The value of the rho terms belonging to this scenario
        # (dictionary nested by node name)
        self._rho = {}

        # This set of fixed or reported stale variables
        # in each tree node
        self._fixed = {}
        self._stale = {}

    @property
    def name(self):
        return self._name

    @property
    def leaf_node(self):
        return self._leaf_node

    @property
    def node_list(self):
        return tuple(self._node_list)

    @property
    def probability(self):
        return self._probability

    @property
    def instance(self):
        return self._instance

    def get_current_objective(self):
        return self._objective

    def get_current_cost(self):
        return self._cost

    def get_current_stagecost(self, stage_name):
        return self._stage_costs[stage_name]

    #
    # a utility to compute the stage index for the input tree node.
    # the returned index is 0-based.
    #

    def node_stage_index(self, tree_node):
        return self._node_list.index(tree_node)



class ScenarioTreeBundle:

    def __init__(self):

        self._name = None
        self._scenario_names = []
        # This is a compressed scenario tree, just for the bundle.
        self._scenario_tree = None
        # the absolute probability of scenarios associated with this
        # node in the scenario tree.
        self._probability = 0.0

    @property
    def name(self):
        return self._name

    @property
    def scenario_names(self):
        return self._scenario_names

    @property
    def scenario_tree(self):
        return self._scenario_tree

    @property
    def probability(self):
        return self._probability

class ScenarioTree:

    # a utility to construct scenario bundles.
    def _construct_scenario_bundles(self, bundles):

        for bundle_name in bundles:

            scenario_list = []
            bundle_probability = 0.0
            for scenario_name in bundles[bundle_name]:
                scenario_list.append(scenario_name)
                bundle_probability += \
                    self._scenario_map[scenario_name].probability

            scenario_tree_for_bundle = self.make_compressed(scenario_list,
                                                            normalize=True)

            scenario_tree_for_bundle.validate()

            new_bundle = ScenarioTreeBundle()
            new_bundle._name = bundle_name
            new_bundle._scenario_names = scenario_list
            new_bundle._scenario_tree = scenario_tree_for_bundle
            new_bundle._probability = bundle_probability

            self._scenario_bundles.append(new_bundle)
            self._scenario_bundle_map[new_bundle.name] = new_bundle

    #
    # a utility to construct the stage objects for this scenario tree.
    # operates strictly by side effects, initializing the self
    # _stages and _stage_map attributes.
    #

    def _construct_stages(self,
                          stage_names,
                          stage_variable_names,
                          stage_cost_variable_names,
                          stage_derived_variable_names):

        # construct the stage objects, which will leave them
        # largely uninitialized - no variable information, in particular.
        for stage_name in stage_names:

            new_stage = ScenarioTreeStage()
            new_stage._name = stage_name
            new_stage._scenario_tree = self

            for variable_string in stage_variable_names[stage_name]:
                if isVariableNameIndexed(variable_string):
                    variable_name, match_template = \
                        extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_stage._variable_templates:
                    new_stage._variable_templates[variable_name] = []
                new_stage._variable_templates[variable_name].append(match_template)

            # not all stages have derived variables defined
            if stage_name in stage_derived_variable_names:
                for variable_string in stage_derived_variable_names[stage_name]:
                    if isVariableNameIndexed(variable_string):
                        variable_name, match_template = \
                            extractVariableNameAndIndex(variable_string)
                    else:
                        variable_name = variable_string
                        match_template = ""
                    if variable_name not in new_stage._derived_variable_templates:
                        new_stage._derived_variable_templates[variable_name] = []
                    new_stage._derived_variable_templates[variable_name].append(match_template)

            # de-reference is required to access the parameter value
            # TBD March 2020: make it so the stages always know their cost names.
            # dlw March 2020: when coming from NetworkX, we don't know these yet!!
            cost_variable_string = stage_cost_variable_names[stage_name].value
            if cost_variable_string is not None:
                if isVariableNameIndexed(cost_variable_string):
                    cost_variable_name, cost_variable_index = \
                        extractVariableNameAndIndex(cost_variable_string)
                else:
                    cost_variable_name = cost_variable_string
                    cost_variable_index = None
                new_stage._cost_variable = (cost_variable_name, cost_variable_index)

            self._stages.append(new_stage)
            self._stage_map[stage_name] = new_stage

    """ Constructor
        Arguments:
            scenarioinstance     - the reference (deterministic) scenario instance.
            scenariotreeinstance - the pyomo model specifying all scenario tree (text) data.
            scenariobundlelist   - a list of scenario names to retain, i.e., cull the rest to create a reduced tree!
    """
    def __init__(self,
                 scenariotreeinstance=None,
                 scenariobundlelist=None):

        # some arbitrary identifier
        self._name = None

        # should be called once for each variable blended across a node
        #self._id_labeler = CounterLabeler()
        self._id_labeler = _CUIDLabeler()

        #
        # the core objects defining the scenario tree.
        #

        # collection of ScenarioTreeNodes
        self._tree_nodes = []
        # collection of ScenarioTreeStages - assumed to be in
        # time-order. the set (provided by the user) itself *must* be
        # ordered.
        self._stages = []
        # collection of Scenarios
        self._scenarios = []
        # collection of ScenarioTreeBundles
        self._scenario_bundles = []

        # dictionaries for the above.
        self._tree_node_map = {}
        self._stage_map = {}
        self._scenario_map = {}
        self._scenario_bundle_map = {}

        # a boolean indicating how data for scenario instances is specified.
        # possibly belongs elsewhere, e.g., in the PH algorithm.
        self._scenario_based_data = None

        if scenariotreeinstance is None:
            assert scenariobundlelist is None
            return

        node_ids = scenariotreeinstance.Nodes
        node_child_ids = scenariotreeinstance.Children
        node_stage_ids = scenariotreeinstance.NodeStage
        node_probability_map = scenariotreeinstance.ConditionalProbability
        stage_ids = scenariotreeinstance.Stages
        stage_variable_ids = scenariotreeinstance.StageVariables
        node_variable_ids = scenariotreeinstance.NodeVariables
        stage_cost_variable_ids = scenariotreeinstance.StageCost
        node_cost_variable_ids = scenariotreeinstance.NodeCost
        if any(scenariotreeinstance.StageCostVariable[i].value is not None
               for i in scenariotreeinstance.StageCostVariable):
            logger.warning("DEPRECATED: The 'StageCostVariable' scenario tree "
                           "model parameter has been renamed to 'StageCost'. "
                           "Please update your scenario tree structure model.")
            if any(stage_cost_variable_ids[i].value is not None
                   for i in stage_cost_variable_ids):
                raise ValueError("The 'StageCostVariable' and 'StageCost' "
                                 "parameters can not both be used on a scenario "
                                 "tree structure model.")
            else:
                stage_cost_variable_ids = scenariotreeinstance.StageCostVariable

        if any(stage_cost_variable_ids[i].value is not None
               for i in stage_cost_variable_ids) and \
           any(node_cost_variable_ids[i].value is not None
               for i in node_cost_variable_ids):
            raise ValueError(
                "The 'StageCost' and 'NodeCost' parameters "
                "can not both be used on a scenario tree "
                "structure model.")
        stage_derived_variable_ids = scenariotreeinstance.StageDerivedVariables
        node_derived_variable_ids = scenariotreeinstance.NodeDerivedVariables
        scenario_ids = scenariotreeinstance.Scenarios
        scenario_leaf_ids = scenariotreeinstance.ScenarioLeafNode
        scenario_based_data = scenariotreeinstance.ScenarioBasedData

        # save the method for instance data storage.
        self._scenario_based_data = scenario_based_data()

        # the input stages must be ordered, for both output purposes
        # and knowledge of the final stage.
        if not stage_ids.isordered():
            raise ValueError(
                "An ordered set of stage IDs must be supplied in "
                "the ScenarioTree constructor")

        for node_id in node_ids:
            node_stage_id = node_stage_ids[node_id].value
            if node_stage_id != stage_ids.last():
                if (len(stage_variable_ids[node_stage_id]) == 0) and \
                   (len(node_variable_ids[node_id]) == 0):
                    raise ValueError(
                        "Scenario tree node %s, belonging to stage %s, "
                        "has not been declared with any variables. "
                        "To fix this error, make sure that one of "
                        "the sets StageVariables[%s] or NodeVariables[%s] "
                        "is declared with at least one variable string "
                        "template (e.g., x, x[*]) on the scenario tree "
                        "or in ScenarioStructure.dat."
                        % (node_id, node_stage_id, node_stage_id, node_id))

        #
        # construct the actual tree objects
        #

        # construct the stage objects w/o any linkages first; link them up
        # with tree nodes after these have been fully constructed.
        self._construct_stages(stage_ids,
                               stage_variable_ids,
                               stage_cost_variable_ids,
                               stage_derived_variable_ids)

        # construct the tree node objects themselves in a first pass,
        # and then link them up in a second pass to form the tree.
        # can't do a single pass because the objects may not exist.
        for tree_node_name in node_ids:

            if tree_node_name not in node_stage_ids:
                raise ValueError("No stage is assigned to tree node=%s"
                                 % (tree_node_name))

            stage_name = value(node_stage_ids[tree_node_name])
            if stage_name not in self._stage_map:
                raise ValueError("Unknown stage=%s assigned to tree node=%s"
                                 % (stage_name, tree_node_name))

            node_stage = self._stage_map[stage_name]
            new_tree_node = ScenarioTreeNode(
                tree_node_name,
                value(node_probability_map[tree_node_name]),
                node_stage)

            # extract the node variable match templates
            for variable_string in node_variable_ids[tree_node_name]:
                if isVariableNameIndexed(variable_string):
                    variable_name, match_template = \
                        extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_tree_node._variable_templates:
                    new_tree_node._variable_templates[variable_name] = []
                new_tree_node._variable_templates[variable_name].append(match_template)

            cost_variable_string = node_cost_variable_ids[tree_node_name].value
            if cost_variable_string is not None:
                assert node_stage._cost_variable is None
                if isVariableNameIndexed(cost_variable_string):
                    cost_variable_name, cost_variable_index = \
                        extractVariableNameAndIndex(cost_variable_string)
                else:
                    cost_variable_name = cost_variable_string
                    cost_variable_index = None
            else:
                assert node_stage._cost_variable is not None
                cost_variable_name, cost_variable_index = \
                    node_stage._cost_variable
            new_tree_node._cost_variable = (cost_variable_name, cost_variable_index)

            # extract the node derived variable match templates
            for variable_string in node_derived_variable_ids[tree_node_name]:
                if isVariableNameIndexed(variable_string):
                    variable_name, match_template = \
                        extractVariableNameAndIndex(variable_string)
                else:
                    variable_name = variable_string
                    match_template = ""
                if variable_name not in new_tree_node._derived_variable_templates:
                    new_tree_node._derived_variable_templates[variable_name] = []
                new_tree_node._derived_variable_templates[variable_name].append(match_template)

            self._tree_nodes.append(new_tree_node)
            self._tree_node_map[tree_node_name] = new_tree_node
            self._stage_map[stage_name]._tree_nodes.append(new_tree_node)

        # link up the tree nodes objects based on the child id sets.
        for this_node in self._tree_nodes:
            this_node._children = []
            # otherwise, you're at a leaf and all is well.
            if this_node.name in node_child_ids:
                child_ids = node_child_ids[this_node.name]
                for child_id in child_ids:
                    if child_id in self._tree_node_map:
                        child_node = self._tree_node_map[child_id]
                        this_node._children.append(child_node)
                        if child_node._parent is None:
                            child_node._parent = this_node
                        else:
                            raise ValueError(
                                "Multiple parents specified for tree node=%s; "
                                "existing parent node=%s; conflicting parent "
                                "node=%s"
                                % (child_id,
                                   child_node._parent.name,
                                   this_node.name))
                    else:
                        raise ValueError("Unknown child tree node=%s specified "
                                         "for tree node=%s"
                                         % (child_id, this_node.name))

        # at this point, the scenario tree nodes and the stages are set - no
        # two-pass logic necessary when constructing scenarios.
        for scenario_name in scenario_ids:
            new_scenario = Scenario()
            new_scenario._name = scenario_name

            if scenario_name not in scenario_leaf_ids:
                raise ValueError("No leaf tree node specified for scenario=%s"
                                 % (scenario_name))
            else:
                scenario_leaf_node_name = value(scenario_leaf_ids[scenario_name])
                if scenario_leaf_node_name not in self._tree_node_map:
                    raise ValueError("Uknown tree node=%s specified as leaf "
                                     "of scenario=%s" %
                                     (scenario_leaf_node_name, scenario_name))
                else:
                    new_scenario._leaf_node = \
                        self._tree_node_map[scenario_leaf_node_name]

            current_node = new_scenario._leaf_node
            while current_node is not None:
                new_scenario._node_list.append(current_node)
                # links the scenarios to the nodes to enforce
                # necessary non-anticipativity
                current_node._scenarios.append(new_scenario)
                current_node = current_node._parent
            new_scenario._node_list.reverse()
            # This now loops root -> leaf
            probability = 1.0
            for current_node in new_scenario._node_list:
                probability *= current_node._conditional_probability
                # NOTE: The line placement below is a little weird, in that
                #       it is embedded in a scenario loop - so the probabilities
                #       for some nodes will be redundantly computed. But this works.
                current_node._probability = probability

                new_scenario._stage_costs[current_node.stage.name] = None
                new_scenario._x[current_node.name] = {}
                new_scenario._w[current_node.name] = {}
                new_scenario._rho[current_node.name] = {}
                new_scenario._fixed[current_node.name] = set()
                new_scenario._stale[current_node.name] = set()

            new_scenario._probability = probability

            self._scenarios.append(new_scenario)
            self._scenario_map[scenario_name] = new_scenario

        # for output purposes, it is useful to known the maximal
        # length of identifiers in the scenario tree for any
        # particular category. I'm building these up incrementally, as
        # they are needed. 0 indicates unassigned.
        self._max_scenario_id_length = 0

        # does the actual traversal to populate the members.
        self.computeIdentifierMaxLengths()

        # if a sub-bundle of scenarios has been specified, mark the
        # active scenario tree components and compress the tree.
        if scenariobundlelist is not None:
            self.compress(scenariobundlelist)

        # NEW SCENARIO BUNDLING STARTS HERE
        if value(scenariotreeinstance.Bundling[None]):
            bundles = OrderedDict()
            for bundle_name in scenariotreeinstance.Bundles:
                bundles[bundle_name] = \
                    list(scenariotreeinstance.BundleScenarios[bundle_name])
            self._construct_scenario_bundles(bundles)

    @property
    def scenarios(self):
        return self._scenarios

    @property
    def bundles(self):
        return self._scenario_bundles

    @property
    def subproblems(self):
        if self.contains_bundles():
            return self._scenario_bundles
        else:
            return self._scenarios

    @property
    def stages(self):
        return self._stages

    @property
    def nodes(self):
        return self._tree_nodes

    def is_bundle(self, object_name):
        return object_name in self._scenario_bundle_map

    def is_scenario(self, object_name):
        return object_name in self._scenario_map

    #
    # is the indicated scenario / bundle in the tree?
    #

    def contains_scenario(self, name):
        return name in self._scenario_map

    def contains_bundles(self):
        return len(self._scenario_bundle_map) > 0

    def contains_bundle(self, name):
        return name in self._scenario_bundle_map

    #
    # get the scenario / bundle object from the tree.
    #

    def get_scenario(self, name):
        return self._scenario_map[name]

    def get_bundle(self, name):
        return self._scenario_bundle_map[name]

    def get_subproblem(self, name):
        if self.contains_bundles():
            return self._scenario_bundle_map[name]
        else:
            return self._scenario_map[name]

    def get_scenario_bundle(self, name):
        if not self.contains_bundles():
            return None
        else:
            return self._scenario_bundle_map[name]

    # there are many contexts where manipulators of a scenario
    # tree simply need an arbitrary scenario to proceed...
    def get_arbitrary_scenario(self):
        return self._scenarios[0]

    def contains_node(self, name):
        return name in self._tree_node_map

    #
    # get the scenario tree node object from the tree
    #
    def get_node(self, name):
        return self._tree_node_map[name]

    #
    # utility for compressing or culling a scenario tree based on
    # a provided list of scenarios (specified by name) to retain -
    # all non-referenced components are eliminated. this particular
    # method compresses *in-place*, i.e., via direct modification
    # of the scenario tree structure. If normalize=True, all probabilities
    # (and conditional probabilities) are renormalized.
    #

    def compress(self,
                 scenario_bundle_list,
                 normalize=True):

        # scan for and mark all referenced scenarios and
        # tree nodes in the bundle list - all stages will
        # obviously remain.
        try:

            for scenario_name in scenario_bundle_list:

                scenario = self._scenario_map[scenario_name]
                scenario.retain = True

                # chase all nodes comprising this scenario,
                # marking them for retention.
                for node in scenario._node_list:
                    node.retain = True

        except KeyError:
            raise ValueError("Scenario=%s selected for "
                             "bundling not present in "
                             "scenario tree"
                             % (scenario_name))

        # scan for any non-retained scenarios and tree nodes.
        scenarios_to_delete = []
        tree_nodes_to_delete = []
        for scenario in self._scenarios:
            if hasattr(scenario, "retain"):
                delattr(scenario, "retain")
            else:
                scenarios_to_delete.append(scenario)
                del self._scenario_map[scenario.name]

        for tree_node in self._tree_nodes:
            if hasattr(tree_node, "retain"):
                delattr(tree_node, "retain")
            else:
                tree_nodes_to_delete.append(tree_node)
                del self._tree_node_map[tree_node.name]

        # JPW does not claim the following routines are
        # the most efficient. rather, they get the job
        # done while avoiding serious issues with
        # attempting to remove elements from a list that
        # you are iterating over.

        # delete all references to unmarked scenarios
        # and child tree nodes in the scenario tree node
        # structures.
        for tree_node in self._tree_nodes:
            for scenario in scenarios_to_delete:
                if scenario in tree_node._scenarios:
                    tree_node._scenarios.remove(scenario)
            for node_to_delete in tree_nodes_to_delete:
                if node_to_delete in tree_node._children:
                    tree_node._children.remove(node_to_delete)

        # delete all references to unmarked tree nodes
        # in the scenario tree stage structures.
        for stage in self._stages:
            for tree_node in tree_nodes_to_delete:
                if tree_node in stage._tree_nodes:
                    stage._tree_nodes.remove(tree_node)

        # delete all unreferenced entries from the core scenario
        # tree data structures.
        for scenario in scenarios_to_delete:
            self._scenarios.remove(scenario)
        for tree_node in tree_nodes_to_delete:
            self._tree_nodes.remove(tree_node)

        #
        # Handle re-normalization of probabilities if requested
        #
        if normalize:

            # re-normalize the conditional probabilities of the
            # children at each tree node (leaf-to-root stage order).
            for stage in reversed(self._stages[:-1]):

                for tree_node in stage._tree_nodes:
                    norm_factor = sum(child_tree_node._conditional_probability
                                      for child_tree_node
                                      in tree_node._children)
                    # the user may specify that the probability of a
                    # scenario is 0.0, and while odd, we should allow the
                    # edge case.
                    if norm_factor == 0.0:
                        for child_tree_node in tree_node._children:
                            child_tree_node._conditional_probability = 0.0
                    else:
                        for child_tree_node in tree_node._children:
                            child_tree_node._conditional_probability /= norm_factor

            # update absolute probabilities (root-to-leaf stage order)
            for stage in self._stages[1:]:
                for tree_node in stage._tree_nodes:
                    tree_node._probability = \
                        tree_node._parent._probability * \
                        tree_node._conditional_probability

            # update scenario probabilities
            for scenario in self._scenarios:
                scenario._probability = \
                    scenario._leaf_node._probability

        # now that we've culled the scenarios, cull the bundles. do
        # this in two passes. in the first pass, we identify the names
        # of bundles to delete, by looking for bundles with deleted
        # scenarios. in the second pass, we delete the bundles from
        # the scenario tree, and normalize the probabilities of the
        # remaining bundles.

        # indices of the objects in the scenario tree bundle list
        bundles_to_delete = []
        for i in range(0,len(self._scenario_bundles)):
            scenario_bundle = self._scenario_bundles[i]
            for scenario_name in scenario_bundle._scenario_names:
                if scenario_name not in self._scenario_map:
                    bundles_to_delete.append(i)
                    break
        bundles_to_delete.reverse()
        for i in bundles_to_delete:
            deleted_bundle = self._scenario_bundles.pop(i)
            del self._scenario_bundle_map[deleted_bundle.name]

        sum_bundle_probabilities = \
            sum(bundle._probability for bundle in self._scenario_bundles)
        for bundle in self._scenario_bundles:
            bundle._probability /= sum_bundle_probabilities

    #
    # Returns a compressed tree using operations on the order of the
    # number of nodes in the compressed tree rather than the number of
    # nodes in the full tree (this method is more efficient than in-place
    # compression). If normalize=True, all probabilities
    # (and conditional probabilities) are renormalized.
    #
    # *** Bundles are ignored. The compressed tree will not have them ***
    #
    def make_compressed(self,
                        scenario_bundle_list,
                        normalize=False):

        compressed_tree = ScenarioTree()
        compressed_tree._scenario_based_data = self._scenario_based_data
        #
        # Copy Stage Data
        #
        for stage in self._stages:
            # copy everything but the list of tree nodes
            # and the reference to the scenario tree
            compressed_tree_stage = ScenarioTreeStage()
            compressed_tree_stage._name = stage.name
            compressed_tree_stage._variable_templates = copy.deepcopy(stage._variable_templates)
            compressed_tree_stage._derived_variable_templates = \
                copy.deepcopy(stage._derived_variable_templates)
            compressed_tree_stage._cost_variable = copy.deepcopy(stage._cost_variable)
            # add the stage object to the compressed tree
            compressed_tree._stages.append(compressed_tree_stage)
            compressed_tree._stages[-1]._scenario_tree = compressed_tree

        compressed_tree._stage_map = \
            dict((stage.name, stage) for stage in compressed_tree._stages)

        #
        # Copy Scenario and Node Data
        #
        compressed_tree_root = None
        for scenario_name in scenario_bundle_list:
            full_tree_scenario = self.get_scenario(scenario_name)

            compressed_tree_scenario = Scenario()
            compressed_tree_scenario._name = full_tree_scenario.name
            compressed_tree_scenario._probability = full_tree_scenario._probability
            compressed_tree._scenarios.append(compressed_tree_scenario)

            full_tree_node = full_tree_scenario._leaf_node
            ### copy the node
            compressed_tree_node = ScenarioTreeNode(
                full_tree_node.name,
                full_tree_node._conditional_probability,
                compressed_tree._stage_map[full_tree_node._stage.name])
            compressed_tree_node._variable_templates = \
                copy.deepcopy(full_tree_node._variable_templates)
            compressed_tree_node._derived_variable_templates = \
                copy.deepcopy(full_tree_node._derived_variable_templates)
            compressed_tree_node._scenarios.append(compressed_tree_scenario)
            compressed_tree_node._stage._tree_nodes.append(compressed_tree_node)
            compressed_tree_node._probability = full_tree_node._probability
            compressed_tree_node._cost_variable = full_tree_node._cost_variable
            ###

            compressed_tree_scenario._node_list.append(compressed_tree_node)
            compressed_tree_scenario._leaf_node = compressed_tree_node
            compressed_tree._tree_nodes.append(compressed_tree_node)
            compressed_tree._tree_node_map[compressed_tree_node.name] = \
                compressed_tree_node

            previous_compressed_tree_node = compressed_tree_node
            full_tree_node = full_tree_node._parent
            while full_tree_node.name not in compressed_tree._tree_node_map:

                ### copy the node
                compressed_tree_node = ScenarioTreeNode(
                    full_tree_node.name,
                    full_tree_node._conditional_probability,
                    compressed_tree._stage_map[full_tree_node.stage.name])
                compressed_tree_node._variable_templates = \
                    copy.deepcopy(full_tree_node._variable_templates)
                compressed_tree_node._derived_variable_templates = \
                    copy.deepcopy(full_tree_node._derived_variable_templates)
                compressed_tree_node._probability = full_tree_node._probability
                compressed_tree_node._cost_variable = full_tree_node._cost_variable
                compressed_tree_node._scenarios.append(compressed_tree_scenario)
                compressed_tree_node._stage._tree_nodes.append(compressed_tree_node)
                ###

                compressed_tree_scenario._node_list.append(compressed_tree_node)
                compressed_tree._tree_nodes.append(compressed_tree_node)
                compressed_tree._tree_node_map[compressed_tree_node.name] = \
                    compressed_tree_node
                previous_compressed_tree_node._parent = compressed_tree_node
                compressed_tree_node._children.append(previous_compressed_tree_node)
                previous_compressed_tree_node = compressed_tree_node

                full_tree_node = full_tree_node._parent
                if full_tree_node is None:
                    compressed_tree_root = compressed_tree_node
                    break

            # traverse the remaining nodes up to the root and update the
            # tree structure elements
            if full_tree_node is not None:
                compressed_tree_node = \
                    compressed_tree._tree_node_map[full_tree_node.name]
                previous_compressed_tree_node._parent = compressed_tree_node
                compressed_tree_node._scenarios.append(compressed_tree_scenario)
                compressed_tree_node._children.append(previous_compressed_tree_node)
                compressed_tree_scenario._node_list.append(compressed_tree_node)

                compressed_tree_node = compressed_tree_node._parent
                while compressed_tree_node is not None:
                    compressed_tree_scenario._node_list.append(compressed_tree_node)
                    compressed_tree_node._scenarios.append(compressed_tree_scenario)
                    compressed_tree_node = compressed_tree_node._parent

            # makes sure this list is in root to leaf order
            compressed_tree_scenario._node_list.reverse()
            assert compressed_tree_scenario._node_list[-1] is \
                compressed_tree_scenario._leaf_node
            assert compressed_tree_scenario._node_list[0] is \
                compressed_tree_root

            # initialize solution related dictionaries
            for compressed_tree_node in compressed_tree_scenario._node_list:
                compressed_tree_scenario._stage_costs[compressed_tree_node._stage.name] = None
                compressed_tree_scenario._x[compressed_tree_node.name] = {}
                compressed_tree_scenario._w[compressed_tree_node.name] = {}
                compressed_tree_scenario._rho[compressed_tree_node.name] = {}
                compressed_tree_scenario._fixed[compressed_tree_node.name] = set()
                compressed_tree_scenario._stale[compressed_tree_node.name] = set()

        compressed_tree._scenario_map = \
            dict((scenario.name, scenario) for scenario in compressed_tree._scenarios)

        #
        # Handle re-normalization of probabilities if requested
        #
        if normalize:

            # update conditional probabilities (leaf-to-root stage order)
            for compressed_tree_stage in reversed(compressed_tree._stages[:-1]):

                for compressed_tree_node in compressed_tree_stage._tree_nodes:
                    norm_factor = \
                        sum(compressed_tree_child_node._conditional_probability
                            for compressed_tree_child_node
                            in compressed_tree_node._children)
                    # the user may specify that the probability of a
                    # scenario is 0.0, and while odd, we should allow the
                    # edge case.
                    if norm_factor == 0.0:
                        for compressed_tree_child_node in \
                               compressed_tree_node._children:
                            compressed_tree_child_node._conditional_probability = 0.0

                    else:
                        for compressed_tree_child_node in \
                               compressed_tree_node._children:
                            compressed_tree_child_node.\
                                _conditional_probability /= norm_factor

            assert abs(compressed_tree_root._probability - 1.0) < 1e-5
            assert abs(compressed_tree_root._conditional_probability - 1.0) < 1e-5

            # update absolute probabilities (root-to-leaf stage order)
            for compressed_tree_stage in compressed_tree._stages[1:]:
                for compressed_tree_node in compressed_tree_stage._tree_nodes:
                    compressed_tree_node._probability = \
                            compressed_tree_node._parent._probability * \
                            compressed_tree_node._conditional_probability

            # update scenario probabilities
            for compressed_tree_scenario in compressed_tree._scenarios:
                compressed_tree_scenario._probability = \
                    compressed_tree_scenario._leaf_node._probability

        return compressed_tree

    #
    # Adds a bundle to this scenario tree by calling make compressed
    # with normalize=True
    # Returns a compressed tree using operations on the order of the
    # number of nodes in the compressed tree rather than the number of
    # nodes in the full tree (this method is more efficient than in-place
    # compression). If normalize=True, all probabilities
    # (and conditional probabilities) are renormalized.
    #
    #
    def add_bundle(self, name, scenario_bundle_list):

        if name in self._scenario_bundle_map:
            raise ValueError("Cannot add a new bundle with name '%s', a bundle "
                             "with that name already exists." % (name))

        bundle_scenario_tree = self.make_compressed(scenario_bundle_list,
                                                    normalize=True)
        bundle = ScenarioTreeBundle()
        bundle._name = name
        bundle._scenario_names = scenario_bundle_list
        bundle._scenario_tree = bundle_scenario_tree
        # make sure this is computed with the un-normalized bundle scenarios
        bundle._probability = sum(self._scenario_map[scenario_name]._probability
                                  for scenario_name in scenario_bundle_list)

        self._scenario_bundle_map[name] = bundle
        self._scenario_bundles.append(bundle)

    def remove_bundle(self, name):

        if name not in self._scenario_bundle_map:
            raise KeyError("Cannot remove bundle with name '%s', no bundle "
                           "with that name exists." % (name))
        bundle = self._scenario_bundle_map[name]
        del self._scenario_bundle_map[name]
        self._scenario_bundles.remove(bundle)

    #
    # utility for automatically selecting a proportion of scenarios from the
    # tree to retain, eliminating the rest.
    #

    def downsample(self, fraction_to_retain, random_seed, verbose=False):

        random_state = random.getstate()
        random.seed(random_seed)
        try:
            number_to_retain = \
                max(int(round(float(len(self._scenarios)*fraction_to_retain))), 1)
            random_list=random.sample(range(len(self._scenarios)), number_to_retain)

            scenario_bundle_list = []
            for i in range(number_to_retain):
                scenario_bundle_list.append(self._scenarios[random_list[i]].name)

            if verbose:
                print("Downsampling scenario tree - retained %s "
                      "scenarios: %s"
                      % (len(scenario_bundle_list),
                         str(scenario_bundle_list)))

            self.compress(scenario_bundle_list) # do the downsampling
        finally:
            random.setstate(random_state)


    #
    # returns the root node of the scenario tree
    #

    def findRootNode(self):

        for tree_node in self._tree_nodes:
            if tree_node._parent is None:
                return tree_node
        return None

    #
    # a utility function to compute, based on the current scenario tree content,
    # the maximal length of identifiers in various categories.
    #

    def computeIdentifierMaxLengths(self):

        self._max_scenario_id_length = 0
        for scenario in self._scenarios:
            if len(str(scenario.name)) > self._max_scenario_id_length:
                self._max_scenario_id_length = len(str(scenario.name))

    #
    # a utility function to (partially, at the moment) validate a scenario tree
    #

    def validate(self):

        # for any node, the sum of conditional probabilities of the children should sum to 1.
        for tree_node in self._tree_nodes:
            sum_probabilities = 0.0
            if len(tree_node._children) > 0:
                for child in tree_node._children:
                    sum_probabilities += child._conditional_probability
                if abs(1.0 - sum_probabilities) > 0.000001:
                    raise ValueError("ScenarioTree validation failed. "
                                     "Reason: child conditional "
                                     "probabilities for tree node=%s "
                                     " sum to %s"
                                     % (tree_node.name,
                                        sum_probabilities))

        # ensure that there is only one root node in the tree
        num_roots = 0
        root_ids = []
        for tree_node in self._tree_nodes:
            if tree_node._parent is None:
                num_roots += 1
                root_ids.append(tree_node.name)

        if num_roots != 1:
            raise ValueError("ScenarioTree validation failed. "
                             "Reason: illegal set of root "
                             "nodes detected: " + str(root_ids))

        # there must be at least one scenario passing through each tree node.
        for tree_node in self._tree_nodes:
            if len(tree_node._scenarios) == 0:
                raise ValueError("ScenarioTree validation failed. "
                                 "Reason: there are no scenarios "
                                 "associated with tree node=%s"
                                 % (tree_node.name))
                return False

        return True

    def create_random_bundles(self,
                              num_bundles,
                              random_seed):

        random_state = random.getstate()
        random.seed(random_seed)
        try:
            num_scenarios = len(self._scenarios)

            sequence = list(range(num_scenarios))
            random.shuffle(sequence)

            next_scenario_index = 0

            # this is a hack-ish way to re-initialize the Bundles set of a
            # scenario tree instance, which should already be there
            # (because it is defined in the abstract model).  however, we
            # don't have a "clear" method on a set, so...
            bundle_names = ["Bundle"+str(i)
                            for i in range(1, num_bundles+1)]
            bundles = OrderedDict()
            for i in range(num_bundles):
                bundles[bundle_names[i]] = []

            scenario_index = 0
            while (scenario_index < num_scenarios):
                for bundle_index in range(num_bundles):
                    if (scenario_index == num_scenarios):
                        break
                    bundles[bundle_names[bundle_index]].append(
                        self._scenarios[sequence[scenario_index]].name)
                    scenario_index += 1

            self._construct_scenario_bundles(bundles)
        finally:
            random.setstate(random_state)

    #
    # a utility function to pretty-print the static/non-cost
    # information associated with a scenario tree
    #

    def pprint(self):

        print("Scenario Tree Detail")

        print("----------------------------------------------------")
        print("Tree Nodes:")
        print("")
        for tree_node_name in sorted(self._tree_node_map.keys()):
            tree_node = self._tree_node_map[tree_node_name]
            print("\tName=%s" % (tree_node_name))
            if tree_node._stage is not None:
                print("\tStage=%s" % (tree_node._stage._name))
            else:
                print("\t Stage=None")
            if tree_node._parent is not None:
                print("\tParent=%s" % (tree_node._parent._name))
            else:
                print("\tParent=" + "None")
            if tree_node._conditional_probability is not None:
                print("\tConditional probability=%4.4f" % tree_node._conditional_probability)
            else:
                print("\tConditional probability=" + "***Undefined***")
            print("\tChildren:")
            if len(tree_node._children) > 0:
                for child_node in sorted(tree_node._children, key=lambda x: x._name):
                    print("\t\t%s" % (child_node._name))
            else:
                print("\t\tNone")
            print("\tScenarios:")
            if len(tree_node._scenarios) == 0:
                print("\t\tNone")
            else:
                for scenario in sorted(tree_node._scenarios, key=lambda x: x._name):
                    print("\t\t%s" % (scenario._name))
            if len(tree_node._variable_templates) > 0:
                print("\tVariables: ")
                for variable_name in sorted(tree_node._variable_templates.keys()):
                    match_templates = tree_node._variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            if len(tree_node._derived_variable_templates) > 0:
                print("\tDerived Variables: ")
                for variable_name in sorted(tree_node._derived_variable_templates.keys()):
                    match_templates = tree_node._derived_variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            print("")
        print("----------------------------------------------------")
        print("Stages:")
        for stage_name in sorted(self._stage_map.keys()):
            stage = self._stage_map[stage_name]
            print("\tName=%s" % (stage_name))
            print("\tTree Nodes: ")
            for tree_node in sorted(stage._tree_nodes, key=lambda x: x._name):
                print("\t\t%s" % (tree_node._name))
            if len(stage._variable_templates) > 0:
                print("\tVariables: ")
                for variable_name in sorted(stage._variable_templates.keys()):
                    match_templates = stage._variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            if len(stage._derived_variable_templates) > 0:
                print("\tDerived Variables: ")
                for variable_name in sorted(stage._derived_variable_templates.keys()):
                    match_templates = stage._derived_variable_templates[variable_name]
                    sys.stdout.write("\t\t "+variable_name+" : ")
                    for match_template in match_templates:
                       sys.stdout.write(indexToString(match_template)+' ')
                    print("")
            print("\tCost Variable: ")
            if stage._cost_variable is not None:
                cost_variable_name, cost_variable_index = stage._cost_variable
            else:
                # kind of a hackish way to get around the fact that we are transitioning
                # away from storing the cost_variable identifier on the stages
                cost_variable_name, cost_variable_index = stage.nodes[0]._cost_variable
            if cost_variable_index is None:
                print("\t\t" + cost_variable_name)
            else:
                print("\t\t" + cost_variable_name + indexToString(cost_variable_index))
            print("")
        print("----------------------------------------------------")
        print("Scenarios:")
        for scenario_name in sorted(self._scenario_map.keys()):
            scenario = self._scenario_map[scenario_name]
            print("\tName=%s" % (scenario_name))
            print("\tProbability=%4.4f" % scenario._probability)
            if scenario._leaf_node is None:
                print("\tLeaf node=None")
            else:
                print("\tLeaf node=%s" % (scenario._leaf_node._name))
            print("\tTree node sequence:")
            for tree_node in scenario._node_list:
                print("\t\t%s" % (tree_node._name))
            print("")
        print("----------------------------------------------------")
        if len(self._scenario_bundles) > 0:
            print("Scenario Bundles:")
            for bundle_name in sorted(self._scenario_bundle_map.keys()):
                scenario_bundle = self._scenario_bundle_map[bundle_name]
                print("\tName=%s" % (bundle_name))
                print("\tProbability=%4.4f" % scenario_bundle._probability            )
                sys.stdout.write("\tScenarios:  ")
                for scenario_name in sorted(scenario_bundle._scenario_names):
                    sys.stdout.write(str(scenario_name)+' ')
                sys.stdout.write("\n")
                print("")
            print("----------------------------------------------------")


    #
    # Save the tree structure in DOT file format
    # Nodes are labeled with absolute probabilities and
    # edges are labeled with conditional probabilities
    #
    def save_to_dot(self, filename):

        def _visit_node(node):
            f.write("%s%s [label=\"%s\"];\n"
                    % (node.name,
                       id(node),
                       str(node.name)+("\n(%.6g)" % (node._probability))))
            for child_node in node._children:
                _visit_node(child_node)
                f.write("%s%s -> %s%s [label=\"%.6g\"];\n"
                        % (node.name,
                           id(node),
                           child_node.name,
                           id(child_node),
                           child_node._conditional_probability))
            if len(node._children) == 0:
                assert len(node._scenarios) == 1
                scenario = node._scenarios[0]
                f.write("%s%s [label=\"%s\"];\n"
                        % (scenario.name,
                           id(scenario),
                           "scenario\n"+str(scenario.name)))
                f.write("%s%s -> %s%s [style=dashed];\n"
                        % (node.name,
                           id(node),
                           scenario.name,
                           id(scenario)))

        with open(filename, 'w') as f:

            f.write("digraph ScenarioTree {\n")
            root_node = self.findRootNode()
            _visit_node(root_node)
            f.write("}\n")
