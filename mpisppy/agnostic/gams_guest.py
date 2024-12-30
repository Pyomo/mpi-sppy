###############################################################################
# mpi-sppy: MPI-based Stochastic Programming in PYthon
#
# Copyright (c) 2024, Lawrence Livermore National Security, LLC, Alliance for
# Sustainable Energy, LLC, The Regents of the University of California, et al.
# All rights reserved. Please see the files COPYRIGHT.md and LICENSE.md for
# full copyright and license information.
###############################################################################
# In this example, GAMS is the guest language.
# NOTE: unlike everywhere else, we are using xbar instead of xbars (no biggy)

"""
This file tries to show many ways to do things in gams,
but not necessarily the best ways in any case.
"""

import itertools
import os
import gams
import gamspy_base
import shutil
import tempfile
import re

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

from mpisppy import MPI  # for debugging
fullcomm = MPI.COMM_WORLD
global_rank = fullcomm.Get_rank()

LINEARIZED = True
GM_NAME = "PH___Model"  # to put in the new GAMS files
this_dir = os.path.dirname(os.path.abspath(__file__))
gamspy_base_dir = gamspy_base.__path__[0]

class GAMS_guest():
    """
    Provide an interface to a model file for a GAMS guest.
    
    Args:
        model_file_name (str): name of Python file that has functions like scenario_creator
        gams_file_name (str): name of GAMS file that is passed to the model file
        nonants_name_pairs (list of (str,str)): list of (non_ant_support_set_name, non_ant_variable_name)
        cfg (pyomo config object)
    """
    def __init__(self, model_file_name, new_file_name, nonants_name_pairs, cfg):
        self.model_file_name = model_file_name
        self.model_module = sputils.module_name_to_module(model_file_name)
        self.new_file_name = new_file_name
        self.nonants_name_pairs = nonants_name_pairs
        self.cfg = cfg
        self.temp_dir = tempfile.TemporaryDirectory()

    def __del__(self):
        self.temp_dir.cleanup()

    def scenario_creator(self, scenario_name, **kwargs):
        """ Wrap the guest (GAMS in this case) scenario creator

        Args:
            scenario_name (str):
                Name of the scenario to construct.

        """
        new_file_name = self.new_file_name
        assert new_file_name is not None
        stoch_param_name_pairs = self.model_module.stoch_param_name_pairs_creator()

        ws = gams.GamsWorkspace(working_directory=self.temp_dir.name, system_directory=gamspy_base_dir)

        ### Calling this function is required regardless of the model
        # This function creates a model instance not instantiated yet, and gathers in glist all the parameters and variables that need to be modifiable
        mi, job, glist, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict = pre_instantiation_for_PH(ws, new_file_name, self.nonants_name_pairs, stoch_param_name_pairs)

        opt = ws.add_options()
        opt.all_model_types = self.cfg.solver_name
        if LINEARIZED:
            mi.instantiate(f"{GM_NAME} using lp minimizing objective_ph", glist, opt)
        else:
            mi.instantiate(f"{GM_NAME} using qcp minimizing objective_ph", glist, opt)

        ### Calling this function is required regardless of the model
        # This functions initializes, by adding records (and values), all the parameters that appear due to PH
        nonant_set_sync_dict = adding_record_for_PH(self.nonants_name_pairs, self.cfg, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict, job)

        mi = self.model_module.scenario_creator(scenario_name, mi, job, **kwargs)
        mi.solve()
        nonant_variable_list = [nonant_var  for (_, nonant_variables_name) in self.nonants_name_pairs for nonant_var in mi.sync_db.get_variable(nonant_variables_name)]

        gd = {
            "scenario": mi,
            "nonants": {("ROOT",i): 0 for i,v in enumerate(nonant_variable_list)},
            "nonant_fixedness": {("ROOT",i): v.get_lower() == v.get_upper() for i,v in enumerate(nonant_variable_list)},
            "nonant_start": {("ROOT",i): v.get_level() for i,v in enumerate(nonant_variable_list)},
            "probability": "uniform",
            "sense": pyo.minimize,
            "BFs": None,
            "nonants_name_pairs": self.nonants_name_pairs,
            "nonant_set_sync_dict": nonant_set_sync_dict
        }
        return gd


    #=========
    def scenario_names_creator(self, num_scens,start=None):
        return self.model_module.scenario_names_creator(num_scens,start)


    #=========
    def inparser_adder(self, cfg):
        self.model_module.inparser_adder(cfg)


    #=========
    def kw_creator(self, cfg):
        # creates keywords for scenario creator
        return self.model_module.kw_creator(cfg)

    
    # This is not needed for PH
    def sample_tree_scen_creator(self, sname, stage, sample_branching_factors, seed,
                                 given_scenario=None, **scenario_creator_kwargs):
        return self.model_module.sample_tree_scen_creator(sname, stage, sample_branching_factors, seed,
                                               given_scenario, **scenario_creator_kwargs)

    #============================
    def scenario_denouement(self, rank, scenario_name, scenario):
        pass
        # (the fct in farmer won't work because the Var names don't match)
        #self.model_module.scenario_denouement(rank, scenario_name, scenario)

    ##################################################################################################
    # begin callouts
    # NOTE: the callouts all take the Ag object as their first argument, mainly to see cfg if needed
    # the function names correspond to function names in mpisppy

    def attach_Ws_and_prox(self, Ag, sname, scenario):
        # Done in create_ph_model
        pass


    def _disable_prox(self, Ag, scenario):
        #print(f"In {global_rank=} for {scenario.name}: disabling prox")
        scenario._agnostic_dict["scenario"].sync_db.get_parameter("prox_on").first_record().value = 0

        
    def _disable_W(self, Ag, scenario):
        #print(f"In {global_rank=} for {scenario.name}: disabling W")
        scenario._agnostic_dict["scenario"].sync_db.get_parameter("W_on").first_record().value = 0

        
    def _reenable_prox(self, Ag, scenario):
        #print(f"In {global_rank=} for {scenario.name}: reenabling prox")
        scenario._agnostic_dict["scenario"].sync_db.get_parameter("prox_on").first_record().value = 1

        
    def _reenable_W(self, Ag, scenario):
        #print(f"In {global_rank=} for {scenario.name}: reenabling W")
        scenario._agnostic_dict["scenario"].sync_db.get_parameter("W_on").first_record().value = 1
        
        
    def attach_PH_to_objective(self, Ag, sname, scenario, add_duals, add_prox):
        # Done in create_ph_model
        pass

    def solve_one(self, Ag, s, solve_keyword_args, gripe, tee, need_solution=True):
        # s is the host scenario
        # This needs to attach stuff to s (see solve_one in spopt.py)
        # Solve the guest language version, then copy values to the host scenario

        # This function needs to put W on the guest right before the solve

        # We need to operate on the guest scenario, not s; however, attach things to s (the host scenario)
        # and copy to s. If you are working on a new guest, you should not have to edit the s side of things

        # To acommdate the solve_one call from xhat_eval.py, we need to attach the obj fct value to s
        self._copy_Ws_xbar_rho_from_host(s)
        gd = s._agnostic_dict
        gs = gd["scenario"]  # guest scenario handle

        """before_solve_bounds = {}
        W_dict = {}
        for record in s._agnostic_dict["scenario"].sync_db.get_variable("x"):
            before_solve_bounds[tuple(record.keys)] = record.get_level(), record.get_lower(), record.get_upper()
            W_dict[tuple(record.keys)] = s._agnostic_dict["scenario"].sync_db.get_parameter("ph_W_x").find_record(record.keys).get_value()
        print(f"For {s.name} in {global_rank=}: {before_solve_bounds=} \n and {W_dict=}")"""

        solver_exception = None
        #import sys
        #print(f"SOLVING FOR {s.name}")
        try:
            gs.solve()
            #gs.solve(output=sys.stdout)#update_type=2)
        except Exception as e:
            solver_exception = e
            print(f"{solver_exception=}")
        
        solve_ok = (1, 2, 7, 8, 15, 16, 17)

        #print(f"{gs.model_status=}, {gs.solver_status=}")
        if gs.model_status not in solve_ok:
            s._mpisppy_data.scenario_feasible = False
            if gripe:
                print (f"Solve failed for scenario {s.name} on rank {global_rank}")
                print(f"{gs.model_status =}")
            s._mpisppy_data._obj_from_agnostic = None
            return

        if solver_exception is not None and need_solution:
            raise solver_exception
        
        s._mpisppy_data.scenario_feasible = True

        # For debugging, thisprints the weights of the PH model
        """W_dict = {}
        for record in s._agnostic_dict["scenario"].sync_db.get_variable("x"):
            #after_solve_bounds[tuple(record.keys)] = record.get_level(), record.get_lower(), record.get_upper()
            W_dict[tuple(record.keys)] = s._agnostic_dict["scenario"].sync_db.get_parameter("ph_W_x").find_record(record.keys).get_value()
        print(f"For {s.name} in {global_rank=}: {W_dict=}")"""

        # For debugging
        """if global_rank == 0:
            x_dict = {}
            for record in s._agnostic_dict["scenario"].sync_db.get_variable("x"):
                #xbar = s._agnostic_dict["scenario"].sync_db.get_parameter("xbar").find_record(record.keys).get_value()
                x_var_rec = s._agnostic_dict["scenario"].sync_db.get_variable("x").find_record(record.keys)
                #x = x_var_rec.get_level()
                x = x_var_rec.get_level() #, x_var_rec.get_lower() == x_var_rec.get_level(), x_var_rec.get_upper() == x_var_rec.get_level(), x_var_rec.get_lower()==s._agnostic_dict["scenario"].sync_db.get_parameter("xlo").find_record(record.keys).get_value(), x_var_rec.get_upper()==s._agnostic_dict["scenario"].sync_db.get_parameter("xup").find_record(record.keys).get_value() 
                #assert  xbar == x, f"for {s.name}, {tuple(record.keys)}: {xbar=} and {x=}"
                x_dict[tuple(record.keys)] = x
            print(f"for {s.name}: {x_dict=}")"""

        objval = gs.sync_db.get_variable('objective_ph').find_record().get_level()

        # copy the nonant x values from gs to s so mpisppy can use them in s
        # in general, we need more checks (see the pyomo agnostic guest example)
        i = 0
        for nonants_set, nonants_var in gd["nonants_name_pairs"]:
            for record in gs.sync_db.get_variable(nonants_var):
                ndn_i = ('ROOT', i)
                s._mpisppy_data.nonant_indices[ndn_i]._value = record.get_level()
                i += 1

        if gd["sense"] == pyo.minimize:
            s._mpisppy_data.outer_bound = objval
        else:
            s._mpisppy_data.outer_bound = objval

        # the next line ignores bundling
        s._mpisppy_data._obj_from_agnostic = objval

        # TBD: deal with other aspects of bundling (see solve_one in spopt.py)
        #print(f"For {s.name} in {global_rank=}: {objval=}")


    # local helper
    def _copy_Ws_xbar_rho_from_host(self, s):
        # special for farmer
        # print(f"   debug copy_Ws {s.name =}, {global_rank =}")

        gd = s._agnostic_dict
        # could/should use set values, but then use a dict to get the indexes right
        gs = gd["scenario"]
        if hasattr(s._mpisppy_model, "W"):
            i=0
            for nonants_set, nonants_var in gd["nonants_name_pairs"]:
                #print(f'{gd["nonant_set_sync_dict"]}')
                rho_dict = {}
                for element in gd["nonant_set_sync_dict"][nonants_set]:
                    ndn_i = ('ROOT', i)
                    rho_dict[element] = s._mpisppy_model.rho[ndn_i].value
                    gs.sync_db[f"ph_W_{nonants_var}"].find_record(element).set_value(s._mpisppy_model.W[ndn_i].value)
                    gs.sync_db[f"rho_{nonants_var}"].find_record(element).set_value(s._mpisppy_model.rho[ndn_i].value)
                    gs.sync_db[f"{nonants_var}bar"].find_record(element).set_value(s._mpisppy_model.xbars[ndn_i].value)
                    i += 1
                #print(f"for {s.name} in {global_rank=}, {rho_dict=}")
            

    # local helper
    def _copy_nonants_from_host(self, s):
        # values and fixedness;
        #print(f"copiing nonants from host in {s.name}")
        gs = s._agnostic_dict["scenario"]
        gd = s._agnostic_dict

        i = 0
        for nonants_set, nonants_var in gd["nonants_name_pairs"]:
            gs.sync_db.get_parameter(f"{nonants_var}lo").clear()
            gs.sync_db.get_parameter(f"{nonants_var}up").clear()
            for element in gd["nonant_set_sync_dict"][nonants_set]:
                ndn_i = ("ROOT", i)
                hostVar = s._mpisppy_data.nonant_indices[ndn_i]
                if hostVar.is_fixed():
                    #print(f"FIXING for {global_rank=}, in {s.name}, for {element=}: {hostVar._value=}")
                    gs.sync_db.get_parameter(f"{nonants_var}lo").add_record(element).set_value(hostVar._value)
                    gs.sync_db.get_parameter(f"{nonants_var}up").add_record(element).set_value(hostVar._value)
                else:
                    gs.sync_db.get_variable(nonants_var).find_record(element).set_level(hostVar._value)
                i += 1
        #print("LEAVING _copy_nonants")


    def _restore_nonants(self, Ag, s):
        # the host has already restored
        self._copy_nonants_from_host(s)

        
    def _restore_original_fixedness(self, Ag, s):
        # the host has already restored
        #This doesn't seem to be used and may not work correctly
        self._copy_nonants_from_host(self, s)


    def _fix_nonants(self, Ag, s):
        # the host has already fixed?
        self._copy_nonants_from_host(s)


    def _fix_root_nonants(self, Ag, s):
        #This doesn't seem to be used and may not work correctly
        self._copy_nonants_from_host(s)


### This function creates a new gams model file including PH before anything else happens

def create_ph_model(original_file_path, new_file_path, nonants_name_pairs):
    """Copies the original file in original_file_path and creates a new file in new_file_path including the PH penalty, weights...
    
    Args:
        original_file_path (str):
            The path (including the name of the gms file) of the original file
        new_file_path (str):
            The path (including the name of the gms file) in which is created the gams model with the ph_objective
        nonants_name_pairs (list of (str,str)):
            List of (non_ant_support_set_name, non_ant_variable_name)
    """
    # Copy the original file
    shutil.copy2(original_file_path, new_file_path)
    
    # Read the content of the new file
    with open(new_file_path, 'r') as file:
        lines = file.readlines()
    
    # The keyword is used to insert everything from PH.
    # For now the model was always defined three lines later to make life easier
    insert_keyword = "__InsertPH__here_Model_defined_three_lines_later"
    line_number = None

    # Searches through the lines for the keyword (from the end because its located in the end)
    # Also captures whether the problem is a minimization or maximization
    # problem and modifies the solve line (although it might not be necessary)
    for i in range(len(lines)):
        index = i #  len(lines)-1-i
        line = lines[index]
        if line.startswith("Model"):
            words = re.findall(r'\b\w+\b', line)
            gmodel_name = words[1]
            line = line.replace(gmodel_name, GM_NAME)
            lines[index] = line
            
        if line.startswith("solve"):
            # Should be in the last lines. This line 
            words = re.findall(r'\b\w+\b', line)
            # print(f"{words=}")
            if "minimizing" in words:
                sense = "minimizing"
                sign = "+"
            elif "maximizing" in words:
                print("WARNING: the objective function's sign has been changed")
                sense = "maximizing"
                sign = "-"
            else:
                raise RuntimeError(f"The line: {line}, doesn't include any sense")
            assert gmodel_name == words[1], f"Model line as model name {gmodel_name} but solve line has {words[1]}"
            # The word following the sense is the objective value
            index_word = words.index(sense)
            previous_objective = words[index_word + 1]
            line = line.replace(sense, "minimizing")
            line = line.replace(gmodel_name, GM_NAME)
            new_obj = "objective_ph"
            lines[index] = new_obj.join(line.rsplit(previous_objective, 1))
            
        if insert_keyword in line:
            line_number = index
    
    assert line_number is not None, "the insert_keyword is not used"
    
    # Where the text will be inserted. After the comment
    insert_position = line_number + 2

    #First modify the model to include the new equations and assert that the model is defined at the good position
    model_line = lines[insert_position + 1]
    model_line_stripped = model_line.strip().lower()

    ### Modify the line where the model is defined. If all the lines are included, nthing changes. Otherwise the new lines are added.
    model_line_text = ""
    if LINEARIZED:
        for nonants_name_pair in nonants_name_pairs:
            nonants_support_set, nonant_variables = nonants_name_pair
            model_line_text += f", PenLeft_{nonant_variables}, PenRight_{nonant_variables}"

    assert "model" in model_line_stripped and "/" in model_line_stripped and model_line_stripped.endswith("/;"), "this is not the model line"
    all_words = [" all ", "/all ", " all/", "/all/"]
    all_model = False
    for word in all_words:
        if word in model_line:
            all_model = True
    if all_model: # we still use all the equations
        lines[insert_position + 1] = model_line 
    else: # we have to specify which equations we use
        lines[insert_position + 1] = model_line[:-4] + model_line_text + ", objective_ph_def" + model_line[-4:]

    ### Adds all the lines in the end of the gms file (where insert here is written).
    # The lines contain the parameters, variables, equations... that are necessary for PH. Their value will be instanciated
    # later in adding_record_for_PH

    parameter_definition = ""
    scalar_definition = """
   W_on      'activate w term'    /    0 /
   prox_on   'activate prox term' /    0 /"""
    variable_definition = ""
    linearized_inequation_definition = ""
    objective_ph_excess = ""
    linearized_equation_expression = ""
    parameter_initialization = ""

    for nonant_name_pair in nonants_name_pairs:
        nonants_support_set, nonant_variables = nonant_name_pair
        if "(" in nonants_support_set:
            nonants_paranthesis_support_set = nonants_support_set
        else:
            nonants_paranthesis_support_set = f"({nonants_support_set})"

        parameter_definition += f"""
   ph_W_{nonant_variables}({nonants_support_set})        'ph weight'
   {nonant_variables}bar({nonants_support_set})        'ph average'
   rho_{nonant_variables}({nonants_support_set})         'ph rho'"""
        
        parameter_definition2 = f"""
   ph_W_{nonant_variables}({nonants_support_set})        'ph weight'                   /set.{nonants_support_set} 0/
   {nonant_variables}bar({nonants_support_set})        'ph average'                  /set.{nonants_support_set} 0/
   rho_{nonant_variables}({nonants_support_set})         'ph rho'                      /set.{nonants_support_set} 0/"""
        
        parameter_definition += f"""
   {nonant_variables}up({nonants_support_set})          'upper bound on {nonant_variables}'
   {nonant_variables}lo({nonants_support_set})          'lower bound on {nonant_variables}'"""

        parameter_definition2 += f"""
   {nonant_variables}up({nonants_support_set})          'upper bound on {nonant_variables}'           /set.{nonants_support_set} 500/
   {nonant_variables}lo({nonants_support_set})          'lower bound on {nonant_variables}'           /set.{nonants_support_set} 0/"""
        
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
        
        # This line is the only line where a cartesian set should appear with paranthesis for the sum. That is why we use nonants_paranthesis_support_set
        objective_ph_excess += f"""
                +  W_on * sum({nonants_paranthesis_support_set}, ph_W_{nonant_variables}({nonants_support_set})*{nonant_variables}({nonants_support_set}))
                +  prox_on * sum({nonants_paranthesis_support_set}, 0.5 * rho_{nonant_variables}({nonants_support_set}) * {PHpenalty})"""
        
        if LINEARIZED:
            linearized_equation_expression += f"""
PenLeft_{nonant_variables}({nonants_support_set}).. PHpenalty_{nonant_variables}({nonants_support_set}) =g= ({nonant_variables}.up({nonants_support_set}) - {nonant_variables}bar({nonants_support_set})) * ({nonant_variables}({nonants_support_set}) - {nonant_variables}bar({nonants_support_set}));
PenRight_{nonant_variables}({nonants_support_set}).. PHpenalty_{nonant_variables}({nonants_support_set}) =g= ({nonant_variables}bar({nonants_support_set}) - {nonant_variables}.lo({nonants_support_set})) * ({nonant_variables}bar({nonants_support_set}) - {nonant_variables}({nonants_support_set}));
"""
        parameter_initialization += f"""
ph_W_{nonant_variables}({nonants_support_set}) = 0;
{nonant_variables}bar({nonants_support_set}) = 0;
rho_{nonant_variables}({nonants_support_set}) = 0;
{nonant_variables}up({nonants_support_set}) = 0;
{nonant_variables}lo({nonants_support_set}) = 0;
"""

    my_text = f"""

Parameter{parameter_definition};

Scalar{scalar_definition};

{parameter_initialization}

Variable{variable_definition}
   objective_ph 'final objective augmented with ph cost';

Equation{linearized_inequation_definition}
   objective_ph_def 'defines objective_ph';

objective_ph_def..    objective_ph =e= {sign} {previous_objective} {objective_ph_excess};

{linearized_equation_expression}
"""

    lines.insert(insert_position, my_text)

    # Write the modified content back to the new file
    with open(new_file_path, 'w') as file:
        file.writelines(lines)
    
    #print(f"Modified file saved as: {new_file_path}")


def file_name_creator(original_file_path):
    """ Returns the path and name of the gms file where ph will be added
    Args:
        original_file_path (str): the path (including the name) of the original gms path
    """
    # Get the directory and filename
        
    directory, filename = os.path.split(os.path.abspath(original_file_path))
    name, ext = os.path.splitext(filename)

    assert ext == ".gms", "the original data file should be a gms file"
    
    # Create the new filename
    if LINEARIZED:
        new_filename = f"{name}_ph_linearized{ext}"
    else:
        print("WARNING: the normal quadratic PH has not been tested")
        new_filename = f"{name}_ph_quadratic{ext}"
    new_file_path = os.path.join(directory, new_filename)
    return new_file_path


### Generic functions called inside the specific scenario creator
def _add_or_get_set(mi, out_set):
    # Captures the set using data from the out_database. If it hasn't been added yet to the model insatnce it adds it as well
    try:
        return mi.sync_db.add_set(out_set.name, out_set._dim, out_set.text)
    except gams.GamsException:
        return mi.sync_db.get_set(out_set.name)

def pre_instantiation_for_PH(ws, new_file_name, nonants_name_pairs, stoch_param_name_pairs):
    """creates dictionaries indexed over the name of the stochastic parameters or non anticipative variables
    so they can be added to the GamsModifier. Variables are obtained from the initial model database.

    Args:
        ws (GamsWorkspace): the workspace to create the model instance
        new_file_name (str): the gms file in which is created the gams model with the ph_objective
        nonants_name_pairs (list of pairs (str, str)): for each non-anticipative variable, the name of the support set must be given with the name of the paramete
        stoch_param_name_pairs (_type_): for each stochastic parameter, the name of the support set must be given with the name of the variable

    Returns:
        tuple: include everything needed for creating the model instance
        nonant_set_sync_dict gives the name of all the elements of the sets presented as tuples. It is useful if the set is a cartesian set: i,j then the elements
        will be of the shape (element_in_i,element_in_j). Some functions iterate over this set.

    """
    ### First create the model instance
    job = ws.add_job_from_file(new_file_name.replace(".gms",""))
    cp = ws.add_checkpoint()
    mi = cp.add_modelinstance()

    job.run(checkpoint=cp) # at this point the model (with what data?) is solved, it creates the file _gams_py_gjo0.lst

    ### Add to the elements that should be modified the stochastic parameters
    # The parameters don't exist yet in the model instance, so they need to be redefined using the job
    stoch_sets_out_dict = {param_name: [job.out_db.get_set(elementary_set) for elementary_set in set_name.split(",")] for set_name, param_name in stoch_param_name_pairs}
    stoch_sets_sync_dict = {param_name: [_add_or_get_set(mi, out_elementary_set) for out_elementary_set in out_elementary_sets] for param_name, out_elementary_sets in stoch_sets_out_dict.items()}
    glist = [gams.GamsModifier(mi.sync_db.add_parameter_dc(param_name, [sync_elementary_set for sync_elementary_set in sync_elementary_sets])) for param_name, sync_elementary_sets in stoch_sets_sync_dict.items()]

    ph_W_dict = {nonant_variables_name: mi.sync_db.add_parameter_dc(f"ph_W_{nonant_variables_name}", nonants_support_set_name.split(","), "ph weight") for nonants_support_set_name, nonant_variables_name in nonants_name_pairs}
    xbar_dict = {nonant_variables_name: mi.sync_db.add_parameter_dc(f"{nonant_variables_name}bar", nonants_support_set_name.split(","), "xbar") for nonants_support_set_name, nonant_variables_name in nonants_name_pairs}
    rho_dict = {nonant_variables_name: mi.sync_db.add_parameter_dc(f"rho_{nonant_variables_name}", nonants_support_set_name.split(","), "rho") for nonants_support_set_name, nonant_variables_name in nonants_name_pairs}

    # x_out is necessary to add the x variables to the database as we need the type and dimension of x
    x_out_dict = {nonant_variables_name: job.out_db.get_variable(f"{nonant_variables_name}") for _, nonant_variables_name in nonants_name_pairs}
    x_dict = {nonant_variables_name: mi.sync_db.add_variable(f"{nonant_variables_name}", x_out_dict[nonant_variables_name]._dim, x_out_dict[nonant_variables_name].vartype) for _, nonant_variables_name in nonants_name_pairs}
    xlo_dict = {nonant_variables_name: mi.sync_db.add_parameter(f"{nonant_variables_name}lo", x_out_dict[nonant_variables_name]._dim, f"lower bound on {nonant_variables_name}") for _, nonant_variables_name in nonants_name_pairs}
    xup_dict = {nonant_variables_name: mi.sync_db.add_parameter(f"{nonant_variables_name}up", x_out_dict[nonant_variables_name]._dim, f"upper bound on {nonant_variables_name}") for _, nonant_variables_name in nonants_name_pairs}

    W_on = mi.sync_db.add_parameter("W_on", 0, "activate w term")
    prox_on = mi.sync_db.add_parameter("prox_on", 0, "activate prox term")

    glist += [gams.GamsModifier(ph_W_dict[nonants_name_pair[1]]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(xbar_dict[nonants_name_pair[1]]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(rho_dict[nonants_name_pair[1]]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(W_on)] \
        + [gams.GamsModifier(prox_on)] \
        + [gams.GamsModifier(x_dict[nonants_name_pair[1]], gams.UpdateAction.Lower, xlo_dict[nonants_name_pair[1]]) for nonants_name_pair in nonants_name_pairs] \
        + [gams.GamsModifier(x_dict[nonants_name_pair[1]], gams.UpdateAction.Upper, xup_dict[nonants_name_pair[1]]) for nonants_name_pair in nonants_name_pairs]

    all_ph_parameters_dicts = {"ph_W_dict": ph_W_dict, "xbar_dict": xbar_dict, "rho_dict": rho_dict, "W_on": W_on, "prox_on": prox_on}
    return mi, job, glist, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict


def adding_record_for_PH(nonants_name_pairs, cfg, all_ph_parameters_dicts, xlo_dict, xup_dict, x_out_dict, job):
    """This function adds records and repopulates the parameters in the gams instance.

    Args:
        most arguments returned by pre_instanciation

    Returns:
        nonant_set_sync_dict (dict{str: list of str tuples}): given a nonant_support_set_name (key), the value is the list
        of all the elements of the sets presented as tuples. It is useful if the set is a cartesian set: i,j then the elements
        will be of the shape (element_in_i,element_in_j). Some functions iterate over this set.
    """
    
    ### Gather the list of non-anticipative variables and their sets from the job, to modify them and add PH related parameters
    nonant_sets_out_dict = {nonant_set_name: [job.out_db.get_set(elementary_set) for elementary_set in nonant_set_name.split(",")] for nonant_set_name, param_name in nonants_name_pairs}

    nonant_set_sync_dict = {nonant_set_name: [element for element in itertools.product(*[[rec.keys[0] for rec in out_elementary_set] for out_elementary_set in out_elementary_sets])] for nonant_set_name, out_elementary_sets in nonant_sets_out_dict.items()}

    for nonants_name_pair in nonants_name_pairs:
        nonants_set_name, nonant_variables_name = nonants_name_pair

        for c in nonant_set_sync_dict[nonants_set_name]:
            all_ph_parameters_dicts["ph_W_dict"][nonant_variables_name].add_record(c).value = 0
            all_ph_parameters_dicts["xbar_dict"][nonant_variables_name].add_record(c).value = 0
            all_ph_parameters_dicts["rho_dict"][nonant_variables_name].add_record(c).value = cfg.default_rho
            xlo_dict[nonant_variables_name].add_record(c).value = x_out_dict[nonant_variables_name].find_record(c).lower
            xup_dict[nonant_variables_name].add_record(c).value = x_out_dict[nonant_variables_name].find_record(c).upper
    all_ph_parameters_dicts["W_on"].add_record().value = 0
    all_ph_parameters_dicts["prox_on"].add_record().value = 0
    return nonant_set_sync_dict
