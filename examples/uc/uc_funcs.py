# Copyright 2020 by B. Knueven, D. Mildebrath, C. Muir, J-P Watson, and D.L. Woodruff
# This software is distributed under the 3-clause BSD License.
# April 2020: DLW asks: why are we using an environment variable and cb_data?
# (maybe there is a super-computer reason)

import os

from pyomo.dataportal import DataPortal

import mpisppy.scenario_tree as scenario_tree

import pyomo.environ as pyo
import mpisppy.utils.sputils as sputils

import egret.parsers.prescient_dat_parser as pdp
import egret.data.model_data as md
import egret.models.unit_commitment as uc

# As of April 2020 we are using cb_data for this
##UC_NUMSCENS_ENV_VAR = "UC_NUM_SCENS"

def pysp_instance_creation_callback(scenario_name, node_names, cb_data):

    #print("Building instance for scenario =", scenario_name)
    scennum = sputils.extract_num(scenario_name)

    uc_model_params = pdp.get_uc_model()

    # Now using cb_data
    ##path = os.environ[UC_NUMSCENS_ENV_VAR] + "scenarios_r1"
    path = cb_data["path"]
    
    scenario_data = DataPortal(model=uc_model_params)
    scenario_data.load(filename=path+os.sep+"RootNode.dat")
    scenario_data.load(filename=path+os.sep+"Node"+str(scennum)+".dat")

    scenario_params = uc_model_params.create_instance(scenario_data,
                                                      report_timing=False,
                                                      name=scenario_name)

    scenario_md = md.ModelData(pdp.create_model_data_dict_params(scenario_params, keep_names=True))

    ## TODO: use the "power_balance_constraints" for now. In the future, networks should be
    ##       handled with a custom callback -- also consider other base models
    scenario_instance = uc.create_tight_unit_commitment_model(scenario_md,
                                                    network_constraints='power_balance_constraints')

    # hold over string attribute from Egret,
    # causes warning wth LShaped/Benders
    del scenario_instance.objective

    return scenario_instance

def scenario_creator(scenario_name,
                     node_names=None,
                     cb_data=None):

    return pysp2_callback(scenario_name,
                          node_names=node_names,
                          cb_data=cb_data)

def pysp2_callback(scenario_name,
                   node_names=None,
                   cb_data=None):
    ''' The callback needs to create an instance and then attach
        the PySP nodes to it in a list _PySPnode_list ordered by stages.
        Optionally attach _PHrho. Standard (1.0) PySP signature for now...
    '''

    instance = pysp_instance_creation_callback(scenario_name, 
                                               node_names, cb_data)

    # now attach the one and only tree node (ROOT is a reserved word)
    # UnitOn[*,*] is the only set of nonant variables
    """
    instance._PySPnode_list = [scenario_tree.ScenarioNode("ROOT",
                                                          1.0,
                                                          1,
                                                          instance.StageCost["Stage_1"], #"Stage_1" hardcodes the commitments in all time periods
                                                          None,
                                                          [instance.UnitOn],
                                                          instance,
                                                          [instance.UnitStart, instance.UnitStop, instance.StartupIndicator],
                                                          )]
    """
    sputils.attach_root_node(instance,
                             instance.StageCost["Stage_1"],
                             [instance.UnitOn],
                             nonant_ef_suppl_list = [instance.UnitStart,
                                                     instance.UnitStop,
                                                     instance.StartupIndicator])
    return instance

def scenario_denouement(rank, scenario_name, scenario):
#    print("First stage cost for scenario",scenario_name,"is",pyo.value(scenario.StageCost["FirstStage"]))
#    print("Second stage cost for scenario",scenario_name,"is",pyo.value(scenario.StageCost["SecondStage"]))
    pass

def scenario_rhosa(scenario_instance):

    return scenario_rhos(scenario_instance)

def _rho_setter(scenario_instance):

    return scenario_rhos(scenario_instance)

def scenario_rhos(scenario_instance, rho_scale_factor=0.1):
    computed_rhos = []
    for t in scenario_instance.TimePeriods:
        for g in scenario_instance.ThermalGenerators:
            max_capacity = pyo.value(scenario_instance.MaximumPowerOutput[g,t])
            min_power = pyo.value(scenario_instance.MinimumPowerOutput[g,t])
            max_power = pyo.value(scenario_instance.MaximumPowerOutput[g,t])
            avg_power = min_power + ((max_power - min_power) / 2.0)

            min_cost = pyo.value(scenario_instance.MinimumProductionCost[g,t])

            avg_cost = scenario_instance.ComputeProductionCosts(scenario_instance, g, t, avg_power) + min_cost
            #max_cost = scenario_instance.ComputeProductionCosts(scenario_instance, g, t, max_power) + min_cost

            computed_rho = rho_scale_factor * avg_cost
            computed_rhos.append((id(scenario_instance.UnitOn[g,t]), computed_rho))
                             
    return computed_rhos

def scenario_rhos_trial_from_file(scenario_instance, rho_scale_factor=0.01,
                                    fname=None):
    ''' First computes the standard rho values (as computed by scenario_rhos()
        above). Then reads rho values from the specified file (raises error if
        no file specified) which is a csv formatted (var_name,rho_value). If
        the rho_value specified in the file is strictly positive, it replaces
        the value computed by scenario_rhos().

        DTM: I wrote this function to test some specific things--I don't think
        this will have a general purpose use, and can probably be deleted.
    '''
    if (fname is None):
        raise RuntimeError('Please provide an "fname" kwarg to '
                           'the "rho_setter_kwargs" option in PHoptions')
    computed_rhos = scenario_rhos(scenario_instance,
                                    rho_scale_factor=rho_scale_factor)
    try:
        trial_rhos = _get_saved_rhos(fname)
    except:
        raise RuntimeError('Formatting issue in specified rho file ' + fname +
                           '. Format should be (variable_name,rho_value) for '
                           'each row, with no blank lines, and no '
                           'extra/commented lines')
    
    index = 0
    for b in sorted(scenario_instance.Buses):
        for t in sorted(scenario_instance.TimePeriods):
            for g in sorted(scenario_instance.ThermalGeneratorsAtBus[b]):
                var = scenario_instance.UnitOn[g,t]
                computed_rho = computed_rhos[index]
                try:
                    trial_rho = trial_rhos[var.name]
                except KeyError:
                    raise RuntimeError(var.name + ' is missing from '
                                       'the specified rho file ' + fname)
                if (trial_rho >= 1e-14):
                    print('Using a trial rho')
                    computed_rhos[index] = (id(var), trial_rho)
                index += 1
                             
    return computed_rhos

def _get_saved_rhos(fname):
    ''' Return a dict of trial rho values, indexed by variable name.
    '''
    rhos = dict()
    with open(fname, 'r') as f:
        for line in f:
            line = line.split(',')
            vname = ','.join(line[:-1])
            rho = float(line[-1])
            rhos[vname] = rho
    return rhos

def id_fix_list_fct(scenario_instance):
    """ specify tuples used by the fixer.

    Args:
        s (ConcreteModel): the sizes instance.
    Returns:
         i0, ik (tuples): one for iter 0 and other for general iterations.
             Var id,  threshold, nb, lb, ub
             The threshold is on the square root of the xbar squared differnce
             nb, lb an bu an "no bound", "upper" and "lower" and give the numver
                 of iterations or None for ik and for i0 anything other than None
                 or None. In both cases, None indicates don't fix.
    """
    import mpisppy.extensions.fixer as fixer

    iter0tuples = []
    iterktuples = []

    for b in sorted(scenario_instance.Buses):
        for t in sorted(scenario_instance.TimePeriods):
            for g in sorted(scenario_instance.ThermalGeneratorsAtBus[b]):

                iter0tuples.append(fixer.Fixer_tuple(scenario_instance.UnitOn[g,t],
                                                     th=0.01, nb=None, lb=0, ub=None))
                
                iterktuples.append(fixer.Fixer_tuple(scenario_instance.UnitOn[g,t],
                                                     th=0.01, nb=None, lb=6, ub=6))

    return iter0tuples, iterktuples

def write_solution(spcomm, opt_dict, solution_dir):
    from mpisppy.cylinders.xhatshufflelooper_bounder import XhatShuffleInnerBound
    from mpisppy.extensions.xhatclosest import XhatClosest
    from mpisppy.opt.ph import PH

    if spcomm.rank_global == 0:
        if spcomm.last_ib_idx is None:
            best_rank_inter = -1
            print("No incumbent solution to print")
        else:
            best_rank_inter = spcomm.last_ib_idx
    else:
        best_rank_inter = None

    best_rank_inter = spcomm.fullcomm.bcast(best_rank_inter, root=0)

    if spcomm.rank_inter != best_rank_inter:
        # Nothing to do
        return
    ## else this spoke/hub is the winner!

    # do some checks, to make sure the solution we print will be nonantipative
    if best_rank_inter != 0:
        assert opt_dict["spoke_class"] in (XhatShuffleInnerBound, )
    else: # this is the hub, TODO: also could check for XhatSpecific
        assert opt_dict["opt_class"] in (PH, )
        assert XhatClosest in opt_dict["opt_kwargs"]["PH_extension_kwargs"]["ext_classes"]
        assert "keep_solution" in opt_dict["opt_kwargs"]["PHoptions"]["xhat_closest_options"]
        assert opt_dict["opt_kwargs"]["PHoptions"]["xhat_closest_options"]["keep_solution"] is True

    ## if we've passed the above checks, the scenarios should have the tree solution

    ## make solution dir if it doesn't exist,
    ## but only on rank 0
    if spcomm.rank_intra == 0:
        if not os.path.exists(solution_dir):
            os.makedirs(solution_dir)

    spcomm.intracomm.Barrier()

    for sname, s in spcomm.opt.local_scenarios.items():
        file_name = os.path.join(solution_dir, sname+'.json')
        mds = uc._save_uc_results(s, relaxed=False)
        mds.write(file_name)

    return
