###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Utilities to support formation and use of "proper" bundles
# TBD: we should consider restructuring this and moving the capability to spbase

import os
import numpy as np
import mpisppy.utils.sputils as sputils
import mpisppy.utils.pickle_bundle as pickle_bundle

# Development notes:
# 2 stage Cases:
#  - ordinary scenario
#  - read a bundle
#  - make a bundle 
#  - make a bundle and write it
# Multi-stage
#  Is not very special because all we do is make sure that
#  bundles cover entire second-stage nodes so the new bundled
#  problem is a two-stage problem no matter how many stages
#  were in the original problem. As of Dec 2024, support
#  is provided only when there are branching factors.
# NOTE:: the caller needs to make sure it is two stage
#        the caller needs to worry about what is in what rank
#        (local_scenarios might have bundle names, e.g.)

class ProperBundler():
    """ Wrap model file functions so as to create proper bundles 
    it might pickle them or read them from a pickle file

    Args:
        module (Python module): the model file with required functions
        comm: (MPI comm): might be the global comm when pickling and writing
    """

    def __init__(self, module, comm=None):
        self.module = module
        self.comm = comm

    def inparser_adder(self, cfg):
        # no need to wrap?
        self.model.inparser_adder(cfg)
        return cfg

    def scenario_names_creator(self, num_scens, start=None, cfg=None):
        # no need to wrap?
        assert cfg is not None, "ProperBundler needs cfg for scenario names"
        return cfg.model.scenario_names_creator(num_scens, start=start)

    def set_bunBFs(self, cfg):
        # utility for bundle objects. Might throw if it doesn't like the branching factors.
        if cfg.get("branching_factors") is None:
            self.bunBFs = None
        else:
            BFs = cfg.branching_factors
            beyond2size = np.prod(BFs[1:])
            bunsize = cfg.scenarios_per_bundle
            if bunsize % beyond2size!= 0:
                raise RuntimeError(f"Bundles must consume the same number of entire second stage nodes: {beyond2size=} {bunsize=}")
            # we need bunBFs for EF formulation
            self.bunBFs = [bunsize // beyond2size] + BFs[1:]
    

    def bundle_names_creator(self, num_buns, start=None, cfg=None):
        # Sets self.bunBFs, which is needed by scenario_creator, as a side effect.

        # start refers to the bundle number; bundles are always zero-based
        if start is None:
            start = 0
        assert cfg is not None, "ProperBundler needs cfg for bundle names"
        assert cfg.get("num_scens") is not None
        assert cfg.get("scenarios_per_bundle") is not None
        assert cfg.num_scens % cfg.scenarios_per_bundle == 0, "Bundles must consume the same number of entire second stage nodes: {cfg.num_scens=} {cfg.scenarios_per_bundle=}"
        bsize = cfg.scenarios_per_bundle  # typing aid
        self.set_bunBFs(cfg)
        # We need to know if scenarios (not bundles) are one-based.
        inum = sputils.extract_num(self.module.scenario_names_creator(1)[0])
        names = [f"Bundle_{bn*bsize+inum}_{(bn+1)*bsize-1+inum}" for bn in range(start+num_buns)]
        return names

    def kw_creator(self, cfg):
        sc_kwargs = self.module.kw_creator(cfg)
        self.original_kwargs = sc_kwargs.copy()
        # Add cfg in case it is not already there
        sc_kwargs["cfg"] = cfg
        return sc_kwargs
    
    def scenario_creator(self, sname, **kwargs):
        """
        Wraps the module scenario_creator to return a bundle if the name
        is Bundle_firstnum_lastnum (e.g. Bundle_14_28)
        This returns a Pyomo model (either for scenario, or the EF of a bundle)
        NOTE: has early returns
        """
        cfg = kwargs["cfg"]
        if "scen" in sname or "Scen" in sname:
            # In case the user passes in kwargs from scenario_creator_kwargs.
            return self.module.scenario_creator(sname, **{**self.original_kwargs, **kwargs})

        elif "Bundle" in sname and cfg.get("unpickle_bundles_dir") is not None:
            fname = os.path.join(cfg.unpickle_bundles_dir, sname+".pkl")
            bundle = pickle_bundle.dill_unpickle(fname)
            return bundle
        elif "Bundle" in sname and cfg.get("unpickle_bundles_dir") is None:
            # If we are still here, we have to create the bundle.
            firstnum = int(sname.split("_")[1])  # sname is a bundle name
            lastnum = int(sname.split("_")[2])
            # snames are scenario names
            snames = self.module.scenario_names_creator(lastnum-firstnum+1,
                                                        firstnum)
            kws = self.original_kwargs
            if self.bunBFs is not None:
                # The original scenario creator needs to handle these
                kws["branching_factors"] = self.bunBFs

            # We are assuming seeds are managed by the *scenario* creator.
            bundle = sputils.create_EF(snames, self.module.scenario_creator,
                                       scenario_creator_kwargs=kws,
                                       EF_name=sname,
                                       suppress_warnings=True,                                       
                                       nonant_for_fixed_vars = False,
                                       total_number_of_scenarios = cfg.num_scens,
            )

            nonantlist = []
            nonant_ef_suppl_list = []
            surrogate_nonant_list = []
            for idx, v in bundle.ref_vars.items():
                # surrogate nonants are added back to the nonant_list by attach_root_node,
                # after they have been noted in surrogate_vardatas
                if idx[0] == "ROOT" and idx not in bundle.ref_surrogate_vars:
                    nonantlist.append(v)
            for idx, v in bundle.ref_suppl_vars.items():
                if idx[0] == "ROOT" and idx not in bundle.ref_surrogate_vars:
                    nonant_ef_suppl_list.append(v)
            surrogate_nonant_list = [v for idx, v in bundle.ref_surrogate_vars.items() if idx[0] =="ROOT"]
            sputils.attach_root_node(bundle, 0, nonantlist, nonant_ef_suppl_list, surrogate_nonant_list)

            # Get an arbitrary scenario.
            scen = bundle.component(snames[0])
            if len(scen._mpisppy_node_list) > 1 and self.bunBFs is None:
                raise RuntimeError("You are creating proper bundles for a\n"
                      "multi-stage problem, but without cfg.branching_factors.\n"
                      "We need branching factors and all bundles must cover\n"
                      "the same number of entire second stage nodes.\n"
                      )
            return bundle
        else:
            raise RuntimeError (f"Scenario name does not have scen or Bundle: {sname}")
