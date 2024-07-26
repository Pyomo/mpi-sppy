# <special for agnostic debugging DLW Aug 2023>
# In this example, GAMS is the guest language.
# NOTE: unlike everywhere else, we are using xbar instead of xbars (no biggy)

"""
This file tries to show many ways to do things in gams,
but not necessarily the best ways in any case.
"""
import sys
import os
import time
import gams
import gamspy_base
import shutil

LINEARIZED = True   # False means quadratic prox (hack) which is not accessible with community license
VERBOSE = -1

this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils
import examples.farmer.farmer as farmer
import numpy as np

# If you need random numbers, use this random stream:
farmerstream = np.random.RandomState()


# for debugging
from mpisppy import MPI
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

# For now nonants_name_pairs is a default otherwise things get tricky as we need to ask a list of pairs to cfg
def scenario_creator(
    scenario_name, nonants_name_pairs=[("crop","x")], cfg=None):
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
    assert cfg is not None, "cfg needs to be transmitted"
    assert cfg.crops_multiplier == 1, "just getting started with 3 crops"

    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

    if LINEARIZED:
        new_file_name = "GAMS/farmer_average_ph_linearized"
    else:
        new_file_name = "GAMS/farmer_average_ph_quadratic"

    job = ws.add_job_from_file(new_file_name)

    #job.run() # at this point the model is solved, it creates the file _gams_py_gjo0.lst

    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    job.run(checkpoint=cp) # at this point the model with bad values is solved, it creates the file _gams_py_gjo0.lst

    # Extract the elements (names) of the set into a list
    crop_elements = [record.keys[0] for record in job.out_db.get_set("crop")]

    crop = mi.sync_db.add_set("crop", 1, "crop type")

    y = mi.sync_db.add_parameter_dc("yield", [crop,], "tons per acre")
    ### Could be done with dict comprehension
    ph_W_dict = {(nonants_support_set_name, nonant_variables_name): mi.sync_db.add_parameter_dc(f"ph_W_{nonant_variables_name}", [nonants_support_set_name,], "ph weight") for nonants_support_set_name, nonant_variables_name in nonants_name_pairs}
    xbar_dict = {}
    rho_dict = {}

    for nonants_name_pair in nonants_name_pairs:
        nonants_support_set_name, nonant_variables_name = nonants_name_pair
        #ph_W_dict[nonants_name_pair] = mi.sync_db.add_parameter_dc(f"ph_W_{nonant_variables_name}", [nonants_support_set_name,], "ph weight")
        xbar_dict[nonants_name_pair] = mi.sync_db.add_parameter_dc(f"{nonant_variables_name}bar", [nonants_support_set_name,], "ph average")
        rho_dict[nonants_name_pair] = mi.sync_db.add_parameter_dc(f"rho_{nonant_variables_name}", [nonants_support_set_name,], "ph rho")

    W_on = mi.sync_db.add_parameter(f"W_on", 0, "activate w term")
    prox_on = mi.sync_db.add_parameter(f"prox_on", 0, "activate prox term")

    x = mi.sync_db.add_variable("x", 1, gams.VarType.Positive)
    xlo = mi.sync_db.add_parameter("xlo", 1, "lower bound on x")
    xup = mi.sync_db.add_parameter("xup", 1, "upper bound on x")

    glist = [gams.GamsModifier(y)] \
        + [gams.GamsModifier(ph_W_dict[nonants_name_pair]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(xbar_dict[nonants_name_pair]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(rho_dict[nonants_name_pair]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(W_on)] \
        + [gams.GamsModifier(prox_on)] \
        + [gams.GamsModifier(x, gams.UpdateAction.Lower, xlo)] \
        + [gams.GamsModifier(x, gams.UpdateAction.Upper, xup)]

    opt = ws.add_options()

    opt.all_model_types = cfg.solver_name
    if LINEARIZED:
        mi.instantiate("simple using lp minimizing objective_ph", glist, opt)
    else:
        mi.instantiate("simple using qcp minimizing objective_ph", glist, opt)

    # initialize W, rho, xbar, W_on, prox_on
    crops = ["wheat", "corn", "sugarbeets"]

    for nonants_name_pair in nonants_name_pairs:
        for c in crop_elements:
            ph_W_dict[nonants_name_pair].add_record(c).value = 0
            xbar_dict[nonants_name_pair].add_record(c).value = 0
            if cfg is None:
                rho_dict[nonants_name_pair].add_record(c).value = 1
            else:
                rho_dict[nonants_name_pair].add_record(c).value = cfg.default_rho
    W_on.add_record().value = 0
    prox_on.add_record().value = 0

    for c in crop_elements:
        xlo.add_record(c).value = 0
        xup.add_record(c).value = 500

    # scenario specific data applied
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

    mi.solve()
    nonant_variable_list = list( mi.sync_db[f"{nonant_variables_name}"] )
    nonant_names_dict = {("ROOT",i): (f"{nonant_variables_name}", v.key(0)) for i, v in enumerate(nonant_variable_list)}

    gd = {
        "scenario": mi,
        "nonants": {("ROOT",i): 0 for i,v in enumerate(nonant_variable_list)},
        "nonant_fixedness": {("ROOT",i): v.get_lower() == v.get_upper() for i,v in enumerate(nonant_variable_list)},
        "nonant_start": {("ROOT",i): v.get_level() for i,v in enumerate(nonant_variable_list)},
        #"nonant_names": nonant_names_dict,
        #"nameset": {nt[0] for nt in nonant_names_dict.values()}, ### TBD should be modified to carry nonants_name_pairs
        "probability": "uniform",
        "sense": pyo.minimize,
        "BFs": None,
        "crop": crop_elements,

        ### Everything added in ph is records. The problem is that the GamsSymbolRecord are only a snapshot of
        # the model. Moreover, once the record is detached from the model, setting the record to have a value
        # won't change the database. Therefore, I have chosen to obtain at each iteration the parameter directly
        # from the synchronized database.
        # The other option might have been to redefine gd at each iteration. But even if this is done, I doubt
        # that some functions such as _reenable_W could be done easily.
    }
    return gd
    
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



##################################################################################################
# begin callouts
# NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
# the function names correspond to function names in mpisppy

def attach_Ws_and_prox(Ag, sname, scenario):
    # Done in create_ph_model
    pass


def _disable_prox(Ag, scenario):
    #print(f"In {global_rank=} for {scenario.name}: disabling prox")
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("prox_on").first_record().value = 0

    
def _disable_W(Ag, scenario):
    #print(f"In {global_rank=} for {scenario.name}: disabling W")
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("W_on").first_record().value = 0

    
def _reenable_prox(Ag, scenario):
    #print(f"In {global_rank=} for {scenario.name}: reenabling prox")
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("prox_on").first_record().value = 1

    
def _reenable_W(Ag, scenario):
    #print(f"In {global_rank=} for {scenario.name}: reenabling W")
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("W_on").first_record().value = 1
    
    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # Done in create_ph_model
    pass


def create_ph_model(original_file, nonants_name_pairs):
    # Get the directory and filename
    directory, filename = os.path.split(original_file)
    name, ext = os.path.splitext(filename)

    assert ext == ".gms", "the original data file should be a gms file"
    
    # Create the new filename
    if LINEARIZED:
        new_filename = f"{name}_ph_linearized{ext}"
    else:
        print("WARNING: the normal quadratic PH has not been tested")
        new_filename = f"{name}_ph_quadratic{ext}"
    new_file_path = os.path.join(directory, new_filename)
    
    # Copy the original file
    shutil.copy2(original_file, new_file_path)
    
    # Read the content of the new file
    with open(new_file_path, 'r') as file:
        lines = file.readlines()
    
    keyword = "__InsertPH__here_Model_defined_three_lines_later"
    line_number = None

    # Insert the new text 3 lines before the end
    for i in range(len(lines)):
        index = len(lines)-1-i
        line = lines[index]
        if keyword in line:
            line_number = index

    assert line_number is not None, "the keyword is not used"

    insert_position = line_number + 2

    #First modify the model to include the new equations and assert that the model is defined at the good position
    model_line = lines[insert_position + 1]
    model_line_stripped = model_line.strip().lower()

    model_line_text = ""
    if LINEARIZED:
        for nonants_name_pair in nonants_name_pairs:
            nonants_support_set, nonant_variables = nonants_name_pair
            model_line_text += f", PenLeft_{nonant_variables}, PenRight_{nonant_variables}"

    assert "model" in model_line_stripped and "/" in model_line_stripped and model_line_stripped.endswith("/;"), "this is not "
    lines[insert_position + 1] = model_line[:-4] + model_line_text + ", objective_ph_def" + model_line[-4:]

    ### TBD differenciate if there is written "/ all /" in the gams model

    parameter_definition = ""
    scalar_definition = f"""
   W_on      'activate w term'    /    0 /
   prox_on   'activate prox term' /    0 /"""
    variable_definition = ""
    linearized_inequation_definition = ""
    objective_ph_excess = ""
    linearized_equation_expression = ""

    for nonant_name_pair in nonants_name_pairs:
        nonants_support_set, nonant_variables = nonant_name_pair

        parameter_definition += f"""
   ph_W_{nonant_variables}({nonants_support_set})        'ph weight'                   /set.{nonants_support_set} 1/
   {nonant_variables}bar({nonants_support_set})        'ph average'                  /set.{nonants_support_set} 0/
   rho_{nonant_variables}({nonants_support_set})         'ph rho'                      /set.{nonants_support_set} 1/"""
        
        parameter_definition += f"""
   {nonant_variables}up(crop)          'upper bound on {nonant_variables}'           /set.{nonants_support_set} 500/
   {nonant_variables}lo(crop)          'lower bound on {nonant_variables}'           /set.{nonants_support_set} 0/"""
        
        variable_definition += f"""
   PHpenalty_{nonant_variables}({nonants_support_set}) 'linearized prox penalty'"""
        
        if LINEARIZED:
            linearized_inequation_definition += f"""
   PenLeft_{nonant_variables}({nonants_support_set}) 'left side of linearized PH penalty'
   PenRight_{nonant_variables}({nonants_support_set}) 'right side of linearized PH penalty'"""

        if LINEARIZED:
            PHpenalty = f"PHpenalty_{nonant_variables}({nonants_support_set})"
        else:
            PHpenalty = f"({nonant_variables}({nonants_support_set}) - {nonant_variables}bar({nonants_support_set}))*({nonant_variables}({nonants_support_set}) - {nonant_variables}bar({nonants_support_set}))"
        objective_ph_excess += f"""
                +  W_on * sum({nonants_support_set}, ph_W_{nonant_variables}({nonants_support_set})*{nonant_variables}({nonants_support_set}))
                +  prox_on * sum({nonants_support_set}, 0.5 * rho_{nonant_variables}({nonants_support_set}) * {PHpenalty})"""
        
        if LINEARIZED:
            linearized_equation_expression += f"""
PenLeft_{nonant_variables}({nonants_support_set}).. PHpenalty_{nonant_variables}({nonants_support_set}) =g= ({nonant_variables}.up({nonants_support_set}) - {nonant_variables}bar({nonants_support_set})) * ({nonant_variables}({nonants_support_set}) - {nonant_variables}bar({nonants_support_set}));
PenRight_{nonant_variables}({nonants_support_set}).. PHpenalty_{nonant_variables}({nonants_support_set}) =g= ({nonant_variables}bar({nonants_support_set}) - {nonant_variables}.lo({nonants_support_set})) * ({nonant_variables}bar({nonants_support_set}) - {nonant_variables}(crop));
"""

    my_text = f"""

Parameter{parameter_definition};

Scalar{scalar_definition};

Variable{variable_definition}
   objective_ph 'final objective augmented with ph cost';

Equation{linearized_inequation_definition}
   objective_ph_def 'defines objective_ph';

objective_ph_def..    objective_ph =e= - profit {objective_ph_excess};

{linearized_equation_expression}
"""

    lines.insert(insert_position, my_text)

    lines[-1] = "solve simple using lp minimizing objective_ph;"

    # Write the modified content back to the new file
    with open(new_file_path, 'w') as file:
        file.writelines(lines)
    
    print(f"Modified file saved as: {new_filename}")
    return f"{name}_ph"


def solve_one(Ag, s, solve_keyword_args, gripe, tee):
    # s is the host scenario
    # This needs to attach stuff to s (see solve_one in spopt.py)
    # Solve the guest language version, then copy values to the host scenario

    # This function needs to put W on the guest right before the solve

    # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
    # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

    # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s
    _copy_Ws_xbar_rho_from_host(s)
    gd = s._agnostic_dict
    gs = gd["scenario"]  # guest scenario handle

    x_dict = {}
    for x_record in s._agnostic_dict["scenario"].sync_db.get_variable('x'):
        x_dict[x_record.get_keys()[0]] = (x_record.get_level(), x_record.get_lower(), x_record.get_upper())
    #print(f"In {global_rank=}, for {s.name}, before solve: {x_dict}")


    solver_name = s._solver_plugin.name   # not used?

    solver_exception = None

    try:
        if VERBOSE==2:
            gs.solve(output=sys.stdout)
        else:
            gs.solve()#update_type=2)
    except Exception as e:
        results = None
        solver_exception = e
        print(f"{solver_exception=}")
    
    solve_ok = (1, 2, 7, 8, 15, 16, 17)

    #print(f"{gs.model_status=}, {gs.solver_status=}")
    if gs.model_status not in solve_ok:
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            print(f"{gs.model_status =}")
            #raise RuntimeError
            
    if solver_exception is not None:
        raise solver_exception

    s._mpisppy_data.scenario_feasible = True

    ## TODO: how to get lower bound??
    objval = gs.sync_db.get_variable('objective_ph').find_record().get_level()

    ### For debugging
    #x_dict = {}
    #for x_record in s._agnostic_dict["scenario"].sync_db.get_variable('x'):
    #    x_dict[x_record.get_keys()[0]] = (x_record.get_level(), x_record.get_lower(), x_record.get_upper())
    #print(f"In {global_rank=}, for {s.name}, after solve: {objval=}, {x_dict}")

    if gd["sense"] == pyo.minimize:
        s._mpisppy_data.outer_bound = objval
    else:
        s._mpisppy_data.outer_bound = objval

    # copy the nonant x values from gs to s so mpisppy can use them in s
    # in general, we need more checks (see the pyomo agnostic guest example)
    
    x = gs.sync_db.get_variable('x')
    i = 0
    for record in x:
        ndn_i = ('ROOT', i)
        #print(f"BEFORE in {global_rank=} {s._mpisppy_data.nonant_indices[ndn_i]._value =}")
        s._mpisppy_data.nonant_indices[ndn_i]._value = record.get_level()
        #print(f"AFTER in {global_rank=} {s._mpisppy_data.nonant_indices[ndn_i]._value =}")
        i += 1

    # the next line ignores bundling
    s._mpisppy_data._obj_from_agnostic = objval

    # TBD: deal with other aspects of bundling (see solve_one in spopt.py)


# local helper
def _copy_Ws_xbar_rho_from_host(s):
    # special for farmer
    # print(f"   debug copy_Ws {s.name =}, {global_rank =}")

    gd = s._agnostic_dict
    # could/should use set values, but then use a dict to get the indexes right
    gs = gd["scenario"]
    if hasattr(s._mpisppy_model, "W"):
        i = 0
        for record in gs.sync_db["ph_W_x"]:
            ndn_i = ('ROOT', i)
            record.set_value(s._mpisppy_model.W[ndn_i].value)
            i += 1
        i = 0
        for record in gs.sync_db["rho_x"]:
            ndn_i = ('ROOT', i)
            record.set_value(s._mpisppy_model.rho[ndn_i].value)
            i += 1
        i = 0
        for record in gs.sync_db["xbar"]:
            ndn_i = ('ROOT', i)
            record.set_value(s._mpisppy_model.xbars[ndn_i].value)
            i += 1
        

# local helper
def _copy_nonants_from_host(s):
    # values and fixedness; 
    #print(f"copiing nonants from host in {s.name}")
    gs = s._agnostic_dict["scenario"]
    crop_elements = s._agnostic_dict["crop"]

    i = 0
    gs.sync_db.get_parameter("xlo").clear()
    gs.sync_db.get_parameter("xup").clear()
    for element in crop_elements:
        ndn_i = ("ROOT", i)
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        if hostVar.is_fixed():
            #print(f"FIXING for {global_rank=}, in {s.name}, for {rec.keys=}: {hostVar._value=}")
            gs.sync_db.get_parameter("xlo").add_record(element).set_value(hostVar._value)
            gs.sync_db.get_parameter("xup").add_record(element).set_value(hostVar._value)
        else:
            gs.sync_db.get_variable("x").find_record(element).set_level(hostVar._value)
        i += 1


def _restore_nonants(Ag, s):
    # the host has already restored
    _copy_nonants_from_host(s)

    
def _restore_original_fixedness(Ag, s):
    # the host has already restored
    #This doesn't seem to be used and may not work correctly
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    # the host has already fixed?
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    #This doesn't seem to be used and may not work correctly
    _copy_nonants_from_host(s)