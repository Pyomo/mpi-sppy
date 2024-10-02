###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# Utilities to support formation and use of "proper" bundles

import os
import dill  
from mpisppy import global_toc
import mpisppy.utils.sputils as sputils
import mpisppy.utils.pickle_bundle as pickle_bundle

# Development notes:
# 2 stage Cases:
#  - ordinary scenario
#  - read a bundle
#  - make a bundle 
#  - make a bundle and write it
# Multi-stage (as of Sept 2024) not supported in class or generic_cylinders
#  You need to do something clever like in aircondB
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

    def bundle_names_creator(self, num_buns, start=None, cfg=None):
        # start refers to the bundle number
        if start is None:
            start = 0
        assert cfg is not None, "ProperBundler needs cfg for bundle names"
        assert cfg.get("num_scens") is not None
        assert cfg.get("scenarios_per_bundle") is not None
        assert cfg.num_scens % cfg.scenarios_per_bundle == 0
        bsize = cfg.scenarios_per_bundle  # typing aid
        names = [f"Bundle_{bn*bsize}_{(bn+1)*bsize-1}" for bn in range(start+num_buns)]
        return names

    def kw_creator(self, cfg):
        sc_kwargs = self.module.kw_creator(cfg)
        self.oringal_kwargs = sc_kwargs.copy()
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
        if "scen" in sname or "Scen" in sname:
            return self.module.scenario_creator(sname, **self.oringal_kwargs)

        elif "Bundle" in sname and cfg.get("unpickle_bundles_dir") is not None:
            fname = os.path.join(cfg.unpickle_bundles_dir, sname+".pkl")
            bundle = pickle_bundle.dill_unpickle(fname)
            return bundle
        elif "Bundle" in sname and cfg.get("unpickle_bundles_dir") is None:
            
            # if we are still here, we have to create the bundle
            firstnum = int(sname.split("_")[1])
            lastnum = int(sname.split("_")[2])
            snames = self.module.scenario_names_creator(firstnum, lastnum)

            print("\nHEY!!! seeds are big trouble\n")
            bundle = sputils.create_EF(snames, self.module.scenario_creator,
                                       scenario_creator_kwargs=self.oringal_kwargs,
                                       EF_name=sname,
                                       nonant_for_fixed_vars = False)
            # every scenario has the same nonants, so create one and grab them
            scen = self.module.scenario_creator(snames[0], **self.oringal_kwargs)
            nonantlist = scen._mpisppy_node_list[0].nonant_vardata_list
            bprob = bundle._mpisppy_probability  # EF instance has probability
            sputils.attach_root_node(bundle, 0, nonantlist)
            # I think attach_root_node messes up the probability
            bundle._mpisppy_probability = bprob

            if kwargs.get("pickle_bundles_dir") is not None:
                # note that sname is a bundle name
                fname = os.path.join(kwargs["pickle_bundles_dir"], sname+".pkl")
                pickle_bundle.dill_pickle(bundle, fname)
            return bundle
        else:
            raise RuntimeError (f"Scenario name does not have scen or Bundle: {sname}")

