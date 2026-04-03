###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""AdmmBundler: proper bundles for stochastic ADMM problems.

Bundles stochastic scenarios within each ADMM subproblem into EF bundles.
Each bundle contains scenarios from the same subproblem, so they share
identical consensus variable patterns (real/dummy). From PH's perspective,
bundles are "big scenarios" at ROOT with all consensus vars as nonants.

Unlike the standard ProperBundler + Stoch_AdmmWrapper pipeline, the
AdmmBundler creates and processes scenarios on-the-fly in scenario_creator,
so PH can distribute bundles across ranks independently.
"""

import numpy as np
import pyomo.environ as pyo

import mpisppy.utils.sputils as sputils
import mpisppy.scenario_tree as scenario_tree
from mpisppy.utils.stoch_admmWrapper import _consensus_vars_number_creator


class AdmmBundler:
    """Bundles stochastic scenarios within each ADMM subproblem.

    Creates EF bundles from groups of stochastic scenarios that belong
    to the same ADMM subproblem. The bundle flattens all consensus
    variables (from all tree levels) into a single ROOT node.

    Unlike Stoch_AdmmWrapper, this class does NOT pre-create scenarios.
    Instead, it creates them on-the-fly in scenario_creator (following the
    ProperBundler pattern), so PH can distribute bundles across ranks.

    Args:
        module: The model module with scenario_creator, etc.
        scenarios_per_bundle (int): Number of stochastic scenarios per bundle.
        admm_subproblem_names (list of str): ADMM subproblem names.
        stoch_scenario_names (list of str): Stochastic scenario names.
        consensus_vars (dict): Maps subproblem name to list of
            (var_name, stage) tuples.
        combining_fn (callable): Creates virtual scenario name from
            (subproblem_name, stochastic_scenario_name).
        split_fn (callable): Splits virtual scenario name into
            (subproblem_name, stochastic_scenario_name).
        scenario_creator_kwargs (dict): kwargs for module.scenario_creator.
    """

    def __init__(self, module, scenarios_per_bundle,
                 admm_subproblem_names, stoch_scenario_names,
                 consensus_vars, combining_fn, split_fn,
                 scenario_creator_kwargs=None):
        self.module = module
        self.scenarios_per_bundle = scenarios_per_bundle
        self.admm_subproblem_names = admm_subproblem_names
        self.stoch_scenario_names = stoch_scenario_names
        self.consensus_vars = consensus_vars
        self.combining_fn = combining_fn
        self.split_fn = split_fn
        self.scenario_creator_kwargs = scenario_creator_kwargs or {}
        self.number_admm_subproblems = len(admm_subproblem_names)
        self.consensus_vars_number = _consensus_vars_number_creator(consensus_vars)

        # Collect all consensus vars with stages
        self.all_consensus_vars = {}
        for sub in consensus_vars:
            for var_stage_tuple in consensus_vars[sub]:
                self.all_consensus_vars[var_stage_tuple[0]] = var_stage_tuple[1]

        # Maps bundle model object → var_prob list
        self._bundle_varprob = {}
        # Maps bundle name → list of (subproblem_name, stoch_scenario_name) tuples
        self._bundle_to_scenarios = {}

    def bundle_names_creator(self):
        """Generate all bundle names grouped by ADMM subproblem.

        Returns:
            list of str: Bundle names, e.g. ["Bundle_ADMM_Region1_0", ...]
        """
        spb = self.scenarios_per_bundle
        num_stoch = len(self.stoch_scenario_names)

        if spb != num_stoch:
            raise RuntimeError(
                f"For stochastic ADMM bundling, scenarios_per_bundle ({spb}) "
                f"must equal num_stoch_scens ({num_stoch}). Partial bundling "
                f"is not supported because different stochastic paths require "
                f"independent ADMM consensus coordination."
            )

        bundle_names = []
        for sub_name in self.admm_subproblem_names:
            num_bundles = num_stoch // spb
            for b in range(num_bundles):
                bname = f"Bundle_ADMM_{sub_name}_{b}"
                stoch_slice = self.stoch_scenario_names[b * spb : (b + 1) * spb]
                self._bundle_to_scenarios[bname] = [
                    (sub_name, sn) for sn in stoch_slice
                ]
                bundle_names.append(bname)

        return bundle_names

    def _process_scenario(self, sname, s, admm_subproblem_name):
        """Process a single scenario: add dummy vars, compute variable probs,
        augment scenario tree with ADMM leaf node.

        This replicates the per-scenario logic from
        Stoch_AdmmWrapper.assign_variable_probs.

        Args:
            sname (str): Virtual scenario name.
            s (ConcreteModel): Scenario model from module.scenario_creator.
            admm_subproblem_name (str): Which ADMM subproblem this belongs to.

        Returns:
            list of (int, float): Variable probability pairs (id(var), prob).
        """
        depth = len(s._mpisppy_node_list) + 1
        varlist = [[] for _ in range(depth)]

        if s._mpisppy_probability == "uniform":
            s._mpisppy_probability = 1 / len(self.stoch_scenario_names)

        varprob = []
        for vstr, stage in self.all_consensus_vars.items():
            v = s.find_component(vstr)
            var_stage_tuple = vstr, stage
            if var_stage_tuple in self.consensus_vars[admm_subproblem_name]:
                if v is None:
                    raise RuntimeError(
                        f"Scenario {sname}: consensus var {vstr} should be "
                        f"in the model for subproblem {admm_subproblem_name} "
                        f"but was not found"
                    )
                if stage == depth:
                    cond_prob = 1.0
                else:
                    prob_node = np.prod([
                        s._mpisppy_node_list[a - 1].cond_prob
                        for a in range(1, stage + 1)
                    ])
                    cond_prob = s._mpisppy_probability / prob_node
                varprob.append((id(v), cond_prob / self.consensus_vars_number[vstr]))
            else:
                if v is None:
                    v2str = vstr.replace("[", "__").replace("]", "__")
                    v = pyo.Var()
                    s.add_component(v2str, v)
                    v.fix(0)
                    varprob.append((id(v), 0))
                else:
                    raise RuntimeError(
                        f"Scenario {sname}: var {vstr} found in model but "
                        f"not in consensus_vars for {admm_subproblem_name}"
                    )
            varlist[stage - 1].append(v)

        # Augment the tree with ADMM leaf node
        parent = s._mpisppy_node_list[-1]
        _, stoch_scenario_name = self.split_fn(sname)
        num_scen = self.stoch_scenario_names.index(stoch_scenario_name)
        node_name = parent.name + '_' + str(num_scen)

        s._mpisppy_node_list.append(scenario_tree.ScenarioNode(
            node_name,
            1 / self.number_admm_subproblems,
            parent.stage + 1,
            pyo.Expression(expr=0),
            varlist[depth - 1],
            s,
        ))
        s._mpisppy_probability /= self.number_admm_subproblems

        # Update existing nodes with consensus var lists and scaled costs
        for stage in range(1, depth):
            old_node = s._mpisppy_node_list[stage - 1]
            s._mpisppy_node_list[stage - 1] = scenario_tree.ScenarioNode(
                old_node.name,
                old_node.cond_prob,
                old_node.stage,
                old_node.cost_expression * self.number_admm_subproblems,
                varlist[stage - 1],
                s,
            )

        return varprob

    def scenario_creator(self, bundle_name, **kwargs):
        """Create an EF bundle from same-subproblem stochastic scenarios.

        Creates constituent scenarios on-the-fly, processes them (adds
        dummy vars, computes variable probs, augments tree), then builds
        an EF bundle with all consensus vars flattened to ROOT.

        Args:
            bundle_name (str): Name of the bundle to create.

        Returns:
            Pyomo ConcreteModel: The bundled EF model.
        """
        constituents = self._bundle_to_scenarios[bundle_name]
        scen_dict = {}
        varprob_by_scenario = {}

        for sub_name, stoch_name in constituents:
            vsname = self.combining_fn(sub_name, stoch_name)
            s = self.module.scenario_creator(vsname, **self.scenario_creator_kwargs)
            varprob = self._process_scenario(vsname, s, sub_name)
            scen_dict[vsname] = s
            varprob_by_scenario[vsname] = varprob

        # Create EF with nonant_for_fixed_vars=True so all bundles have
        # identical nonant structure (fixed dummy vars get trivial 0==0)
        bundle = sputils._create_EF_from_scen_dict(
            scen_dict,
            EF_name=bundle_name,
            nonant_for_fixed_vars=True,
        )

        # Flatten ALL ref_vars from all node levels into ROOT.
        # Sort by (node_name, var_index) for consistent ordering across bundles.
        nonantlist = []
        for idx in sorted(bundle.ref_vars.keys()):
            nonantlist.append(bundle.ref_vars[idx])

        sputils.attach_root_node(bundle, 0, nonantlist)

        # Scale the bundle objective by num_admm_subproblems
        bundle.EF_Obj.expr = bundle.EF_Obj.expr * self.number_admm_subproblems

        # Build variable probability list for this bundle.
        # For shared nodes (e.g., ROOT), multiple scenarios have the same
        # (node_name, var_index) and the ref_var points to the first scenario's
        # variable. The probability should be the SUM across all scenarios
        # that share that ref_var (NA constraints make their values equal).
        # For unique nodes (leaf-level), only one scenario contributes.
        idx_to_prob_sum = {}
        for vsname in scen_dict:
            s = scen_dict[vsname]
            varprob = varprob_by_scenario[vsname]
            # Build mapping: var_id → (node_name, var_index) for this scenario
            var_id_to_idx = {}
            for node in s._mpisppy_node_list:
                for i, v in enumerate(node.nonant_vardata_list):
                    var_id_to_idx[id(v)] = (node.name, i)
            # Accumulate probabilities by (node_name, var_index)
            for vid, prob in varprob:
                idx = var_id_to_idx[vid]
                idx_to_prob_sum[idx] = idx_to_prob_sum.get(idx, 0) + prob

        # Map ref_vars (keyed by EF's (node_name, var_index)) to accumulated probs
        bundle_vpl = []
        for idx in sorted(bundle.ref_vars.keys()):
            ref_var = bundle.ref_vars[idx]
            if idx in idx_to_prob_sum:
                bundle_vpl.append((id(ref_var), idx_to_prob_sum[idx]))
            else:
                raise RuntimeError(
                    f"Bundle {bundle_name}: ref_var {ref_var.name} at {idx} "
                    f"not found in accumulated variable probabilities"
                )

        self._bundle_varprob[bundle] = bundle_vpl

        # Set probability: sum of constituent scenario probabilities
        bundle._mpisppy_probability = sum(
            scen_dict[self.combining_fn(sub, sn)]._mpisppy_probability
            for sub, sn in constituents
        )

        return bundle

    def var_prob_list(self, bundle_model):
        """Return per-variable probabilities for a bundle model.

        Args:
            bundle_model: The bundle's Pyomo ConcreteModel.

        Returns:
            list of (int, float): (variable_id, probability) pairs.
        """
        return self._bundle_varprob[bundle_model]
