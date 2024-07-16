# <special for agnostic debugging DLW Aug 2023>
# In this example, GAMS is the guest language.
# NOTE: unlike everywhere else, we are using xbar instead of xbars (no biggy)

"""
This file tries to show many ways to do things in gams,
but not necessarily the best ways in any case.
"""
import os
import time
import gams
import gamspy_base
import shutil

LINEARIZED = False   # False means quadratic prox (hack)

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

def scenario_creator(
    scenario_name,  nonants_support_set_name="crop", nonant_variables_name="x", use_integer=False, sense=pyo.minimize, crops_multiplier=1,
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
    print(f"******************************** {scenario_name=}")
    nonants_support_set_list = [nonants_support_set_name] # should be given
    nonant_variables_list = [nonant_variables_name]

    assert crops_multiplier == 1, "just getting started with 3 crops"

    ws = gams.GamsWorkspace(working_directory=this_dir, system_directory=gamspy_base_dir)

    if LINEARIZED:
        new_file_name = "GAMS/farmer_average_ph_linearized"
    else:
        new_file_name = "GAMS/farmer_average_ph_quadratic"

    job = ws.add_job_from_file(new_file_name)

    job.run() # at this point the model is solved, it creates the file _gams_py_gjo0.lst
    print("!!!!!!!!!!!!! first unuseful solve done")

    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    job.run(checkpoint=cp)

    crop = mi.sync_db.add_set("crop", 1, "crop type")

    y = mi.sync_db.add_parameter_dc("yield", [crop,], "tons per acre")
    ### Could be done with dict comprehension
    ph_W_dict = {}
    xbar_dict = {}
    rho_dict = {}

    for nonants_support_set_name in nonants_support_set_list:
        ph_W_dict[nonants_support_set_name] = mi.sync_db.add_parameter_dc(f"ph_W_{nonants_support_set_name}", [nonants_support_set_name,], "ph weight")
        xbar_dict[nonants_support_set_name] = mi.sync_db.add_parameter_dc(f"xbar_{nonants_support_set_name}", [nonants_support_set_name,], "ph average")
        rho_dict[nonants_support_set_name] = mi.sync_db.add_parameter_dc(f"rho_{nonants_support_set_name}", [nonants_support_set_name,], "ph rho")

    W_on = mi.sync_db.add_parameter(f"W_on", 0, "activate w term")
    prox_on = mi.sync_db.add_parameter(f"prox_on", 0, "activate prox term")

    """nonant_variables_def_eq = mi.sync_db.add_equation("nonant_variables_def")
    PenLeft_eq = mi.sync_db.add_equation("PenLeft")
    PenRight_eq = mi.sync_db.add_equation("PenRight")
    objective_ph_def_eq = mi.sync_db.add_equation("objective_ph_def")"""

    glist = [gams.GamsModifier(y)] \
        + [gams.GamsModifier(ph_W_dict[nonants_support_set_name]) for nonants_support_set_name in nonants_support_set_list] \
        + [gams.GamsModifier(xbar_dict[nonants_support_set_name]) for nonants_support_set_name in nonants_support_set_list] \
        + [gams.GamsModifier(rho_dict[nonants_support_set_name]) for nonants_support_set_name in nonants_support_set_list] \
        + [gams.GamsModifier(W_on)] \
        + [gams.GamsModifier(prox_on)]
        
    """gams.GamsModifier(nonant_variables_def_eq),
            gams.GamsModifier(PenLeft_eq),
            gams.GamsModifier(PenRight_eq),
            gams.GamsModifier(objective_ph_def_eq),"""

    if LINEARIZED:
        mi.instantiate("simple using lp minimizing objective_ph", glist)
    else:
        mi.instantiate("simple using qcp minimizing objective_ph", glist)
        #mi.instantiate("simple using cplex minimizing objective_ph", glist)

    mi.solve()

    # initialize W, rho, xbar, W_on, prox_on
    """for param in [ph_W, xbar, rho]:
        for rec in param:
            print("1111111111111111111111111111111111111111")
            value = rec.value  # This reads the value from the GAMS file
            param.add_record(rec.keys).value = value
    for scalar in [W_on, prox_on]:
        scalar.add_record().value = 0"""
    crops = ["wheat", "corn", "sugarbeets"]
    for nonants_support_set_name in nonants_support_set_list:
        for c in crops:
            ph_W_dict[nonants_support_set_name].add_record(c).value = 0
            xbar_dict[nonants_support_set_name].add_record(c).value = 0
            rho_dict[nonants_support_set_name].add_record(c).value = 0
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
    nonant_variable_list = list( mi.sync_db[f"{nonant_variables_name}"] ) 
    my_dict = {("ROOT",i): p for nonants_support_set_name in nonants_support_set_list for i,p in enumerate(ph_W_dict[nonants_support_set_name]) }
    #my_dict = {("ROOT",i): p for i,p in enumerate(ph_W)}
    # In general, be sure to process variables in the same order has the guest does (so indexes match)
    nonant_names_dict = {("ROOT",i): (f"{nonant_variables_name}", v.key(0)) for i, v in enumerate(nonant_variable_list)}    
    gd = {
        "scenario": mi,
        "nonants": {("ROOT",i): v for i,v in enumerate(nonant_variable_list)},
        "nonant_fixedness": {("ROOT",i): v.get_lower() == v.get_upper() for i,v in enumerate(nonant_variable_list)},
        "nonant_start": {("ROOT",i): v.get_level() for i,v in enumerate(nonant_variable_list)},
        "nonant_names": nonant_names_dict,
        "nameset": {nt[0] for nt in nonant_names_dict.values()},
        "probability": "uniform",
        "sense": pyo.minimize,
        "BFs": None,
        "ph" : {
            "ph_W" : {("ROOT",i): p for nonants_support_set_name in nonants_support_set_list for i,p in enumerate(ph_W_dict[nonants_support_set_name])},
            "xbar" : {("ROOT",i): p for nonants_support_set_name in nonants_support_set_list for i,p in enumerate(xbar_dict[nonants_support_set_name])},
            "rho" : {("ROOT",i): p for nonants_support_set_name in nonants_support_set_list for i,p in enumerate(rho_dict[nonants_support_set_name])},
            "W_on" : W_on.first_record(),
            "prox_on" : prox_on.first_record(),
            #"obj" : mi.sync_db["negprofit"].find_record(),
            "obj" : mi.sync_db["objective_ph"].find_record(),
            "nonant_lbs" : {("ROOT",i): v.get_lower() for i,v in enumerate(nonant_variable_list)},
            "nonant_ubs" : {("ROOT",i): v.get_upper() for i,v in enumerate(nonant_variable_list)},
        },
    }

    """
    # Read and print the contents of the Gurobi log file
    with open(f"gurobi_{scenario_name}.log", "r") as log_file:
        gurobi_log = log_file.read()
        print("Gurobi Log File Content:\n")
        print(gurobi_log)

    # Read and print the contents of the Gurobi LP model file
    with open(f"my_model_{scenario_name}.lp", "r") as lp_file:
        lp_model = lp_file.read()
        print("\nGurobi LP Model File Content:\n")
        print(lp_model)"""

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
    return farmer.kw_creator(cfg)

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
    scenario._agnostic_dict["ph"]["prox_on"].set_value(0)

    
def _disable_W(Ag, scenario):
    scenario._agnostic_dict["ph"]["W_on"].set_value(0)

    
def _reenable_prox(Ag, scenario):
    scenario._agnostic_dict["ph"]["prox_on"].set_value(1)

    
def _reenable_W(Ag, scenario):
    scenario._agnostic_dict["ph"]["W_on"].set_value(1)
    
    
def attach_PH_to_objective(Ag, sname, scenario, add_duals, add_prox):
    # TODO: hard coded in GAMS model
    pass


def create_ph_model(original_file, nonants_support_set, nonant_variables):
    nonants_support_set_list = [nonants_support_set] # should be given
    nonant_variables_list = [nonant_variables]

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
        for nonants_support_set in nonants_support_set_list:
            model_line_text += f", PenLeft_{nonants_support_set}, PenRight_{nonants_support_set}"

    assert "model" in model_line_stripped and "/" in model_line_stripped and model_line_stripped.endswith("/;"), "this is not "
    lines[insert_position + 1] = model_line[:-4] + model_line_text + ", objective_ph_def" + model_line[-4:]
    
    my_old_text_linearized = f"""

Alias(nonants_support_set,{nonants_support_set});

Parameter
   ph_W(nonants_support_set)        'ph weight'                   /set.nonants_support_set 0/
   xbar(nonants_support_set)        'ph average'                  /set.nonants_support_set 0/
   rho(nonants_support_set)         'ph rho'                      /set.nonants_support_set 0/;

Scalar
   W_on      'activate w term'    /    0 /
   prox_on   'activate prox term' /    0 /;

Variable
   nonant_variables(nonants_support_set) 'the nonant_variables'
   PHpenalty(nonants_support_set) 'linearized prox penalty'
   objective_ph 'objective variable with ph included';

Equation
   nonant_variables_def(nonants_support_set) 'getting the values of the nonant_variables'
   PenLeft(nonants_support_set) 'left side of linearized PH penalty'
   PenRight(nonants_support_set) 'right side of linearized PH penalty'
   objective_ph_def 'objective augmented with ph cost';

nonant_variables_def(nonants_support_set).. nonant_variables(nonants_support_set) =e= {nonant_variables}(nonants_support_set);

objective_ph_def..    objective_ph =e= - profit
                       +  W_on * sum(nonants_support_set, ph_W(nonants_support_set)*nonant_variables(nonants_support_set))
                       +  prox_on * sum(nonants_support_set, 0.5 * rho(nonants_support_set) * PHpenalty(nonants_support_set));

PenLeft(nonants_support_set).. sqr(xbar(nonants_support_set)) + xbar(nonants_support_set)*0 + xbar(nonants_support_set) * nonant_variables(nonants_support_set) + land * nonant_variables(nonants_support_set) =g= PHpenalty(nonants_support_set);
PenRight(nonants_support_set).. sqr(xbar(nonants_support_set)) - xbar(nonants_support_set)*land - xbar(nonants_support_set)*nonant_variables(nonants_support_set) + land * nonant_variables(nonants_support_set) =g= PHpenalty(nonants_support_set);

    """

    parameter_definition = ""
    scalar_definition = f"""
   W_on      'activate w term'    /    0 /
   prox_on   'activate prox term' /    0 /"""
    variable_definition = ""
    linearized_inequation_definition = ""
    objective_ph_excess = ""
    linearized_equation_expression = ""

    for i in range(len(nonants_support_set_list)):
        nonants_support_set = nonants_support_set_list[i]
        print(f"{nonants_support_set}")
        nonant_variables = nonant_variables_list[i]

        parameter_definition += f"""
   ph_W_{nonants_support_set}({nonants_support_set})        'ph weight'                   /set.{nonants_support_set} 0/
   xbar_{nonants_support_set}({nonants_support_set})        'ph average'                  /set.{nonants_support_set} 0/
   rho_{nonants_support_set}({nonants_support_set})         'ph rho'                      /set.{nonants_support_set} 0/"""
        
        variable_definition += f"""
   PHpenalty_{nonants_support_set}({nonants_support_set}) 'linearized prox penalty'"""
        
        if LINEARIZED:
            linearized_inequation_definition += f"""
   PenLeft_{nonants_support_set}({nonants_support_set}) 'left side of linearized PH penalty'
   PenRight_{nonants_support_set}({nonants_support_set}) 'right side of linearized PH penalty'"""

        if LINEARIZED:
            PHpenalty = f"PHpenalty_{nonants_support_set}({nonants_support_set})"
        else:
            PHpenalty = f"(x({nonants_support_set}) - xbar_{nonants_support_set}({nonants_support_set}))*(x({nonants_support_set}) - xbar_{nonants_support_set}({nonants_support_set}))"
        objective_ph_excess += f"""
                +  W_on * sum({nonants_support_set}, ph_W_{nonants_support_set}({nonants_support_set})*{nonant_variables}({nonants_support_set}))
                +  prox_on * sum({nonants_support_set}, 0.5 * rho_{nonants_support_set}({nonants_support_set}) * {PHpenalty})"""

        if LINEARIZED:
            linearized_equation_expression += f"""
PenLeft_{nonants_support_set}({nonants_support_set}).. sqr(xbar_{nonants_support_set}({nonants_support_set})) + xbar_{nonants_support_set}({nonants_support_set})*0 + xbar_{nonants_support_set}({nonants_support_set}) * {nonant_variables}({nonants_support_set}) + land * {nonant_variables}({nonants_support_set}) =g= PHpenalty_{nonants_support_set}({nonants_support_set});
PenRight_{nonants_support_set}({nonants_support_set}).. sqr(xbar_{nonants_support_set}({nonants_support_set})) - xbar_{nonants_support_set}({nonants_support_set})*land - xbar_{nonants_support_set}({nonants_support_set}) * {nonant_variables}({nonants_support_set}) + land * {nonant_variables}({nonants_support_set}) =g= PHpenalty_{nonants_support_set}({nonants_support_set});

PHpenalty_{nonants_support_set}.lo({nonants_support_set}) = 0;
PHpenalty_{nonants_support_set}.up({nonants_support_set}) = max(sqr(xbar_{nonants_support_set}({nonants_support_set}) - 0), sqr(land - xbar_{nonants_support_set}({nonants_support_set})));
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
    #lines[-3] = ""
    #lines[-3] = "Model simple / profitdef, landuse, req, beets, ylddef, PenLeft, PenRight, objective_ph_def /;"

    #lines[-1] = "solve simple using lp minimizing objective_ph;"
    #lines[-1] = ""

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
        gs.solve()
    except Exception as e:
        results = None
        solver_exception = e
    print(f"debug {gs.model_status =}")
    time.sleep(1)  # just hoping this helps...
    
    solve_ok = (1, 2, 7, 8, 15, 16, 17)

    if gs.model_status not in solve_ok:
        s._mpisppy_data.scenario_feasible = False
        if gripe:
            print (f"Solve failed for scenario {s.name} on rank {global_rank}")
            print(f"{gs.model_status =}")
            raise RuntimeError
            
    if solver_exception is not None:
        raise solver_exception

    s._mpisppy_data.scenario_feasible = True

    ## TODO: how to get lower bound??
    objval = gd["ph"]["obj"].get_level()  # use this?
    ###phobjval = gs.get_objective("phobj").value()   # use this???
    s._mpisppy_data.outer_bound = objval

    # copy the nonant x values from gs to s so mpisppy can use them in s
    # in general, we need more checks (see the pyomo agnostic guest example)
    for n in gd["nameset"]:    
        list(gs.sync_db[n])   # needed to get current solution, I guess (the iterator seems to have a side-effect because list is needed)
    for ndn_i, gxvar in gd["nonants"].items():
        try:   # not sure this is needed
            float(gxvar.get_level())
        except:
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "had no value. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")
        if False: # needed?
            raise RuntimeError(
                f"Non-anticipative variable {gxvar.name} on scenario {s.name} "
                "was presolved out. This usually means this variable "
                "did not appear in any (active) components, and hence "
                "was not communicated to the subproblem solver. ")

        s._mpisppy_data.nonant_indices[ndn_i]._value = gxvar.get_level()
        if global_rank == 0:  # debugging
            print(f"solve_one: {s.name =}, {ndn_i =}, {gxvar.get_level() =}")

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
    for ndn_i, gxvar in gd["nonants"].items():
        if hasattr(s._mpisppy_model, "W"):
            gd["ph"]["ph_W"][ndn_i].set_value(s._mpisppy_model.W[ndn_i].value)
            gd["ph"]["rho"][ndn_i].set_value(s._mpisppy_model.rho[ndn_i].value)
            gd["ph"]["xbar"][ndn_i].set_value(s._mpisppy_model.xbars[ndn_i].value)
        else:
            # presumably an xhatter; we should check, I suppose
            pass


# local helper
def _copy_nonants_from_host(s):
    # values and fixedness; 
    gd = s._agnostic_dict
    for ndn_i, gxvar in gd["nonants"].items():
        hostVar = s._mpisppy_data.nonant_indices[ndn_i]
        guestVar = gd["nonants"][ndn_i]
        guestVar.set_level(hostVar._value)
        if hostVar.is_fixed():
            guestVar.set_lower(hostVar._value)
            guestVar.set_upper(hostVar._value)
        else:
            guestVar.set_lower(gd["ph"]["nonant_lbs"][ndn_i])
            guestVar.set_upper(gd["ph"]["nonant_ubs"][ndn_i])


def _restore_nonants(Ag, s):
    # the host has already restored
    _copy_nonants_from_host(s)

    
def _restore_original_fixedness(Ag, s):
    _copy_nonants_from_host(s)


def _fix_nonants(Ag, s):
    _copy_nonants_from_host(s)


def _fix_root_nonants(Ag, s):
    _copy_nonants_from_host(s)