###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
"""Hub dict construction for generic_cylinders."""

import mpisppy.utils.cfg_vanilla as vanilla


def build_hub_dict(cfg, beans, scenario_creator_kwargs,
                   rho_setter, all_nodenames, ph_converger,
                   variable_probability=None):
    """Build and return the hub_dict for WheelSpinner.

    Args:
        cfg: Config object
        beans: tuple of (cfg, scenario_creator, scenario_denouement, all_scenario_names)
        scenario_creator_kwargs: dict for scenario creator
        rho_setter: rho setter function or None
        all_nodenames: list of node names or None
        ph_converger: converger class or None
        variable_probability: variable probability list or None (used by ADMM)

    Returns:
        dict: hub_dict for WheelSpinner
    """
    if cfg.APH:
        hub_dict = vanilla.aph_hub(*beans,
                                   scenario_creator_kwargs=scenario_creator_kwargs,
                                   ph_extensions=None,
                                   rho_setter=rho_setter,
                                   variable_probability=variable_probability,
                                   all_nodenames=all_nodenames,
                                   )
    elif cfg.subgradient_hub:
        hub_dict = vanilla.subgradient_hub(
                       *beans,
                       scenario_creator_kwargs=scenario_creator_kwargs,
                       ph_extensions=None,
                       ph_converger=ph_converger,
                       rho_setter=rho_setter,
                       variable_probability=variable_probability,
                       all_nodenames=all_nodenames,
                   )
    elif cfg.fwph_hub:
        hub_dict = vanilla.fwph_hub(
                       *beans,
                       scenario_creator_kwargs=scenario_creator_kwargs,
                       ph_extensions=None,
                       ph_converger=ph_converger,
                       rho_setter=rho_setter,
                       variable_probability=variable_probability,
                       all_nodenames=all_nodenames,
                   )
    elif cfg.ph_primal_hub:
        hub_dict = vanilla.ph_primal_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter=rho_setter,
                                  variable_probability=variable_probability,
                                  all_nodenames=all_nodenames,
                                  )
    elif cfg.cg_hub:
        #Vanilla CG Hub
        hub_dict = vanilla.cg_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  all_nodenames = all_nodenames,
                                  )
    elif cfg.dualcg_hub:
        #Dual Stabilized CG Hub
        hub_dict = vanilla.dualcg_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  all_nodenames = all_nodenames,
                                  )
    else:
        hub_dict = vanilla.ph_hub(*beans,
                                  scenario_creator_kwargs=scenario_creator_kwargs,
                                  ph_extensions=None,
                                  ph_converger=ph_converger,
                                  rho_setter=rho_setter,
                                  variable_probability=variable_probability,
                                  all_nodenames=all_nodenames,
                                  )

    # transition to strictly cfg-based option passing
    hub_dict['opt_kwargs']['options']['cfg'] = cfg

    return hub_dict
