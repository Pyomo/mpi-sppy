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
    scenario_name,  nonants_name_pairs=[("crop","x")], use_integer=False, sense=pyo.minimize, crops_multiplier=1,
        num_scens=None, seedoffset=0,
):
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

    assert crops_multiplier == 1, "just getting started with 3 crops"

    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

    if LINEARIZED:
        new_file_name = "GAMS/farmer_average_ph_linearized"
    else:
        new_file_name = "GAMS/farmer_average_ph_quadratic"

    job = ws.add_job_from_file(new_file_name)

    #job.run() # at this point the model is solved, it creates the file _gams_py_gjo0.lst

    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    if VERBOSE == 3:
        db = ws.add_database()
        # Step 3: Set up options for LST file generation
        opt = ws.add_options()
        opt.output = "custom_name0.lst" # This ensures LST file is generated

        # Step 4: Run the new job with the updated database and LST option
        job.run(opt, checkpoint=cp, databases=db)

    job.run(checkpoint=cp) # at this point the model with bad values is solved, it creates the file _gams_py_gjo0.lst

    crop = mi.sync_db.add_set("crop", 1, "crop type")

    y = mi.sync_db.add_parameter_dc("yield", [crop,], "tons per acre")
    ### Could be done with dict comprehension
    ph_W_dict = {}
    xbar_dict = {}
    rho_dict = {}

    for nonants_name_pair in nonants_name_pairs:
        nonants_support_set_name, nonant_variables_name = nonants_name_pair
        ph_W_dict[nonants_name_pair] = mi.sync_db.add_parameter_dc(f"ph_W_{nonants_support_set_name}", [nonants_support_set_name,], "ph weight")
        xbar_dict[nonants_name_pair] = mi.sync_db.add_parameter_dc(f"{nonant_variables_name}bar_{nonants_support_set_name}", [nonants_support_set_name,], "ph average")
        rho_dict[nonants_name_pair] = mi.sync_db.add_parameter_dc(f"rho_{nonants_support_set_name}", [nonants_support_set_name,], "ph rho")

    W_on = mi.sync_db.add_parameter(f"W_on", 0, "activate w term")
    prox_on = mi.sync_db.add_parameter(f"prox_on", 0, "activate prox term")

    glist = [gams.GamsModifier(y)] \
        + [gams.GamsModifier(ph_W_dict[nonants_name_pair]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(xbar_dict[nonants_name_pair]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(rho_dict[nonants_name_pair]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(W_on)] \
        + [gams.GamsModifier(prox_on)]

    if LINEARIZED:
        mi.instantiate("simple using lp minimizing objective_ph", glist)
    else:
        mi.instantiate("simple using qcp minimizing objective_ph", glist)

    # initialize W, rho, xbar, W_on, prox_on
    crops = ["wheat", "corn", "sugarbeets"]
    variable_x = job.out_db["x"]
    for nonants_name_pair in nonants_name_pairs:
        for c in crops:
            ph_W_dict[nonants_name_pair].add_record(c).value = 0
            xbar_dict[nonants_name_pair].add_record(c).value = 0
            rho_dict[nonants_name_pair].add_record(c).value = 1
    W_on.add_record().value = 0
    prox_on.add_record().value = 0

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
    #print(f'{mi.sync_db[f"{nonant_variables_name}"]=}')
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

        ### Everything added in ph is records. The problem is that the GamsSymbolRecord are only a snapshot of
        # the model. Moreover, once the record is detached from the model, setting the record to have a value
        # won't change the database. Therefore, I have chosen to obtain at each iteration the parameter directly
        # from the synchronized database.
        # The other option might have been to redefine gd at each iteration. But even if this is done, I doubt
        # that some functions such as _reenable_W could be done easily.

        #"ph" : {
            #"ph_W" : {("ROOT",i): p for nonants_name_pair in nonants_name_pairs for i,p in enumerate(ph_W_dict[nonants_name_pair])},
            #"xbar" : {("ROOT",i): p for nonants_name_pair in nonants_name_pairs for i,p in enumerate(xbar_dict[nonants_name_pair])},
            #"rho" : {("ROOT",i): p for nonants_name_pair in nonants_name_pairs for i,p in enumerate(rho_dict[nonants_name_pair])},
            #"W_on" : W_on.first_record(),
            #"prox_on" : prox_on.first_record(),
            #"obj" : mi.sync_db["objective_ph"].find_record(),
            #"nonant_lbs" : {("ROOT",i): v.get_lower() for i,v in enumerate(nonant_variable_list)},
            #"nonant_ubs" : {("ROOT",i): v.get_upper() for i,v in enumerate(nonant_variable_list)},
        #},
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
    kwargs = farmer.kw_creator(cfg)
    #kwargs["nonants_name_pairs"] = cfg.nonants_name_pairs
    return kwargs

# This is not needed for PH
def sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                             given_scenario=None, **scenario_creator_kwargs):
    return farmer.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                           given_scenario, **scenario_creator_kwargs)

#============================
def scenario_denouement(rank, scenario_name, scenario):
    pass
    # (the fct in farmer won't work because the Var names don't match)
    #farmer.scenario_denouement(rank, scenario_name, scenario)



##################################################################################################
# begin callouts
# NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
# the function names correspond to function names in mpisppy

def attach_Ws_and_prox(Ag, sname, scenario):
    # TODO: the current version has this hardcoded in the GAMS model
    # (W, rho, and xbar all get values right before the solve)
    pass


def _disable_prox(Ag, scenario):
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("prox_on").first_record().value = 0

    
def _disable_W(Ag, scenario):
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("W_on").first_record().value = 0

    
def _reenable_prox(Ag, scenario):
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("prox_on").first_record().value = 1

    
def _reenable_W(Ag, scenario):
    scenario._agnostic_dict["scenario"].sync_db.get_parameter("W_on").first_record().value = 1
    
    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # TODO: hard coded in GAMS model
    pass


def create_ph_model(original_file, nonants_name_pairs):
    #nonants_support_set_list = [nonants_support_set] # should be given

    # Get the directory and filename
    directory, filename = os.path.split(original_file)
    name, ext = os.path.splitext(filename)

    assert ext == ".gms", "the original data file should be a gms file"
    
    # Create the new filename
    if LINEARIZED:
        new_filename = f"{name}_ph_linearized{ext}"
    else:
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
            nonants_support_set, _ = nonants_name_pair
            model_line_text += f", PenLeft_{nonants_support_set}, PenRight_{nonants_support_set}"

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

    #for i in range(len(nonants_support_set_list)):
    for nonant_name_pair in nonants_name_pairs:
        nonants_support_set, nonant_variables = nonant_name_pair
        #nonants_support_set = nonants_support_set_list[i]
        #print(f"{nonants_support_set}")
        #nonant_variables = nonant_variables_list[i]

        parameter_definition += f"""
   ph_W_{nonants_support_set}({nonants_support_set})        'ph weight'                   /set.{nonants_support_set} 1/
   {nonant_variables}bar_{nonants_support_set}({nonants_support_set})        'ph average'                  /set.{nonants_support_set} 0/
   rho_{nonants_support_set}({nonants_support_set})         'ph rho'                      /set.{nonants_support_set} 1/"""
        
        #if LINEARIZED:
        #    parameter_definition += f"""
   #x_upper(crop)          'upper bound for x'           /set.crop 500/
   #x_lower(crop)          'lower bound for x'           /set.crop 0/"""
        
        variable_definition += f"""
   PHpenalty_{nonants_support_set}({nonants_support_set}) 'linearized prox penalty'"""
        
        if LINEARIZED:
            linearized_inequation_definition += f"""
   PenLeft_{nonants_support_set}({nonants_support_set}) 'left side of linearized PH penalty'
   PenRight_{nonants_support_set}({nonants_support_set}) 'right side of linearized PH penalty'"""

        if LINEARIZED:
            PHpenalty = f"PHpenalty_{nonants_support_set}({nonants_support_set})"
        else:
            PHpenalty = f"({nonant_variables}({nonants_support_set}) - {nonant_variables}bar_{nonants_support_set}({nonants_support_set}))*({nonant_variables}({nonants_support_set}) - {nonant_variables}bar_{nonants_support_set}({nonants_support_set}))"
        objective_ph_excess += f"""
                +  W_on * sum({nonants_support_set}, ph_W_{nonants_support_set}({nonants_support_set})*{nonant_variables}({nonants_support_set}))
                +  prox_on * sum({nonants_support_set}, 0.5 * rho_{nonants_support_set}({nonants_support_set}) * {PHpenalty})"""
        
        if LINEARIZED:
            linearized_equation_expression += f"""
PenLeft_{nonants_support_set}({nonants_support_set}).. PHpenalty_{nonants_support_set}({nonants_support_set}) =g= ({nonant_variables}.up({nonants_support_set}) - {nonant_variables}bar_{nonants_support_set}({nonants_support_set})) * ({nonant_variables}({nonants_support_set}) - {nonant_variables}bar_{nonants_support_set}({nonants_support_set}));
PenRight_{nonants_support_set}({nonants_support_set}).. PHpenalty_{nonants_support_set}({nonants_support_set}) =g= ({nonant_variables}bar_{nonants_support_set}({nonants_support_set}) - {nonant_variables}.lo({nonants_support_set})) * ({nonant_variables}bar_{nonants_support_set}({nonants_support_set}) - {nonant_variables}(crop));
"""
            
            unuseful_text = f"""
PenLeft_{nonants_support_set}({nonants_support_set}).. sqr({nonant_variables}bar_{nonants_support_set}({nonants_support_set})) + {nonant_variables}bar_{nonants_support_set}({nonants_support_set})*0 + {nonant_variables}bar_{nonants_support_set}({nonants_support_set}) * {nonant_variables}({nonants_support_set}) + land * {nonant_variables}({nonants_support_set}) =g= PHpenalty_{nonants_support_set}({nonants_support_set});
PenRight_{nonants_support_set}({nonants_support_set}).. sqr({nonant_variables}bar_{nonants_support_set}({nonants_support_set})) - {nonant_variables}bar_{nonants_support_set}({nonants_support_set})*land - {nonant_variables}bar_{nonants_support_set}({nonants_support_set}) * {nonant_variables}({nonants_support_set}) + land * {nonant_variables}({nonants_support_set}) =g= PHpenalty_{nonants_support_set}({nonants_support_set});

PHpenalty_{nonants_support_set}.lo({nonants_support_set}) = 0;
PHpenalty_{nonants_support_set}.up({nonants_support_set}) = max(sqr({nonant_variables}bar_{nonants_support_set}({nonants_support_set}) - 0), sqr(land - {nonant_variables}bar_{nonants_support_set}({nonants_support_set})));
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

    solver_name = s._solver_plugin.name   # not used?

    solver_exception = None

    try:
        if VERBOSE==2:
            gs.solve(output=sys.stdout)
        else:
            gs.solve()
    except Exception as e:
        results = None
        solver_exception = e
        print(f"{solver_exception=}")

    #print(f"debug {gs.model_status =}")
    #print(f"debug {gs.solver_status =}")
    #time.sleep(0.5)  # just hoping this helps...
    
    solve_ok = (1, 2, 7, 8, 15, 16, 17)

    if gs.model_status not in solve_ok:
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            print(f"{gs.model_status =}")
            #raise RuntimeError
            
    if solver_exception is not None:
        raise solver_exception

    s._mpisppy_data.scenario_feasible = True

    if VERBOSE == 0:
        x = gs.sync_db.get_variable('x')
        print(f"\n For scenario {s.name} \n")
        for record in x:
            print(f"{s.name=}, {record.get_keys()=}, {record.get_level()=}")



    ## TODO: how to get lower bound??
    objval = gs.sync_db.get_variable('objective_ph').find_record().get_level()

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
        s._mpisppy_data.nonant_indices[ndn_i]._value = record.get_level()
        i += 1
    
    if s.name == "scen0" and global_rank == 0 and VERBOSE == 0:
        printing_penalties = f"solve_one: {s.name =}, linearized_penalty_crop.get_level()... "
        if global_rank == 0:  # debugging
            for penalty_crop in gs.sync_db.get_variable('PHpenalty_crop'):
                printing_penalties +=f" {penalty_crop.get_level()}, "
            print(printing_penalties)
        xbar_dict = {}
        x_dict = {}
        x_up_dict = {}
        ph_W_dict = {}
        for xbar_record in gs.sync_db.get_parameter('xbar_crop'):
            xbar_dict[xbar_record.get_keys()[0]] = xbar_record.get_value()
        for x_record in gs.sync_db.get_variable('x'):
            x_dict[x_record.get_keys()[0]] = x_record.get_level()
            x_up_dict[x_record.get_keys()[0]] = x_record.get_upper()
        for ph_W_record in gs.sync_db.get_parameter('ph_W_crop'):
            ph_W_dict[ph_W_record.get_keys()[0]] = ph_W_record.get_value()
        print(f"{xbar_dict =}\n {x_dict =} \n {x_up_dict=}")

        linearized_penalty = 0
        squared_penalty = 0
        sum_W = 0
        for rho_record in gs.sync_db.get_parameter('xbar_crop'):
            crop_name = rho_record.get_keys()[0]
            linearized_penalty += 0.5 * rho_record.get_value() * (x_up_dict[crop_name] - xbar_dict[crop_name]) * (x_dict[crop_name] - xbar_dict[crop_name])
            squared_penalty += 0.5 * rho_record.get_value() * (x_dict[crop_name] - xbar_dict[crop_name]) * (x_dict[crop_name] - xbar_dict[crop_name])
            sum_W += ph_W_dict[crop_name]*x_dict[crop_name]
        print(f"{linearized_penalty=}")
        print(f"{squared_penalty=}")
        print(f"{sum_W=}")
        #print(f"{dir(gs.sync_db.get_variable('profit'))=}")

        profit = gs.sync_db.get_variable('profit').first_record().get_level()
        print(f"{profit=}")
        print(f"{-profit + sum_W + linearized_penalty =}")

        print(f"   {objval =}")

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
        for record in gs.sync_db["ph_W_crop"]:
            ndn_i = ('ROOT', i)
            #print(f"{record=}")
            #print(f"{s._mpisppy_model.W[ndn_i].value=}")
            record.set_value(s._mpisppy_model.W[ndn_i].value)
            i += 1
        i = 0
        for record in gs.sync_db["rho_crop"]:
            ndn_i = ('ROOT', i)
            record.set_value(s._mpisppy_model.rho[ndn_i].value)
            i += 1
        i = 0
        for record in gs.sync_db["xbar_crop"]:
            ndn_i = ('ROOT', i)
            record.set_value(s._mpisppy_model.xbars[ndn_i].value)
            i += 1
        

# local helper
def _copy_nonants_from_host(s):
    # values and fixedness; 
    gd = s._agnostic_dict
    guestVar = gd["scenario"].sync_db.get_variable("x")
    i = 0
    for record_guestVar in guestVar:
        ndn_i = ("ROOT", i)
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        record_guestVar.set_level(hostVar._value)
        if hostVar.is_fixed():
            record_guestVar.set_lower(hostVar._value)
            record_guestVar.set_upper(hostVar._value)
        else:
            record_guestVar.set_level(hostVar._value)
            #record_guestVar.set_lower(hostVar.lb)
            #record_guestVar.set_lower(hostVar.ub)


def _restore_nonants(Ag, s):
    # the host has already restored
    _copy_nonants_from_host(s)

    
def _restore_original_fixedness(Ag, s):
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    _copy_nonants_from_host(s)