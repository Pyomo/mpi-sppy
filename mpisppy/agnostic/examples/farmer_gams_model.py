LINEARIZED = True

import os
import gams
import gamspy_base

this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

# for debugging
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

from mpisppy.agnostic import gams_guest
import mpisppy.agnostic.farmer4agnostic as farmer

def nonants_name_pairs_creator():
    return [("crop", "x")]


def scenario_creator(scenario_name, new_file_name, nonants_name_pairs, cfg=None):
    """ Create a scenario for the (scalable) farmer example.
    
    Args:
        scenario_name (str):
            Name of the scenario to construct.
        use_integer (bool, optional):
            If True, restricts variables to be integer. Default is False.
        sense (int, optional):
            Model sense (minimization or maximization). Must be either
            pyo.minimize or pyo.maximize. Default is pyo.minimize.
        crops_multiplier (int, optional):
            Factor to control scaling. There will be three times this many
            crops. Default is 1.
        num_scens (int, optional):
            Number of scenarios. We use it to compute _mpisppy_probability. 
            Default is None.
        seedoffset (int): used by confidence interval code

    """
    assert new_file_name is not None

    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)
    job = ws.add_job_from_file(new_file_name)

    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    job.run(checkpoint=cp) # at this point the model with bad values is solved, it creates the file _gams_py_gjo0.lst

    # Extract the elements (names) of the set into a list
    nonants_support_sets_out = [job.out_db.get_set(nonants_support_set_name) for nonants_support_set_name, _ in nonants_name_pairs]
    set_element_names_dict = {nonant_set.name: [record.keys[0] for record in nonant_set] for nonant_set in nonants_support_sets_out}

    nonants_support_sets = [mi.sync_db.add_set(nonant_set.name, nonant_set._dim, nonant_set.text) for nonant_set in nonants_support_sets_out]

    ### This part is somehow specific to the model
    crop = nonants_support_sets[0]
    y = mi.sync_db.add_parameter_dc("yield", [crop,], "tons per acre")
    glist = [gams.GamsModifier(y)] # will be completed later
    ### End of the specific part

    glist, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict = gams_guest.gamsmodifiers_for_PH(glist, mi, job, nonants_name_pairs)

    ### This part is specific to the model
    opt = ws.add_options()
    opt.all_model_types = cfg.solver_name
    if LINEARIZED:
        mi.instantiate("simple using lp minimizing objective_ph", glist, opt)
    else:
        mi.instantiate("simple using qcp minimizing objective_ph", glist, opt)
    ### End of the specific part

    gams_guest.adding_record_for_PH(nonants_name_pairs, set_element_names_dict, cfg, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict)

    ### This part is specific to the model
    scennum = sputils.extract_num(scenario_name)
    assert scennum < 3, "three scenarios hardwired for now"
    if scennum == 0:  # below
        y.add_record("wheat").value = 2.0
        y.add_record("corn").value = 2.4
        y.add_record("sugarbeets").value = 16.0
    elif scennum == 1: # average
        y.add_record("wheat").value = 2.5
        y.add_record("corn").value = 3.0
        y.add_record("sugarbeets").value = 20.0
    elif scennum == 2: # above
        y.add_record("wheat").value = 3.0
        y.add_record("corn").value = 3.6
        y.add_record("sugarbeets").value = 24.0    
    ### End of the specific part

    return mi, nonants_name_pairs, set_element_names_dict

        
#=========
def scenario_names_creator(num_scens,start=None):
    return farmer.scenario_names_creator(num_scens,start)


#=========
def inparser_adder(cfg):
    farmer.inparser_adder(cfg)

    
#=========
def kw_creator(cfg):
    # creates keywords for scenario creator
    #kwargs = farmer.kw_creator(cfg)
    kwargs = {}
    kwargs["cfg"] = cfg
    #kwargs["nonants_name_pairs"] = nonants_name_pairs_creator()
    return kwargs

# This is not needed for PH
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return farmer.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                           given_scenario, **scenario_creator_kwargs)

#============================
def scenario_denouement(rank, scenario_name, scenario):
    # doesn't seem to be called
    if global_rank == 1:
        x_dict = {}
        for x_record in scenario._agnostic_dict["scenario"].sync_db.get_variable('x'):
            x_dict[x_record.get_keys()[0]] = x_record.get_level()
        print(f"In {scenario_name}: {x_dict}")
    pass
    # (the fct in farmer won't work because the Var names don't match)
    #farmer.scenario_denouement(rank, scenario_name, scenario)
