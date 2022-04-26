# script to estimate zhat from a given xhat for a given model

import sys
import argparse
import importlib
import numpy as np
import scipy.stats
from mpisppy.utils import config
from mpisppy.confidence_intervals import sample_tree
from mpisppy.utils import sputils
from mpisppy.confidence_intervals import ciutils
from mpisppy.utils.xhat_eval import Xhat_Eval
from mpisppy.utils.sputils import option_string_to_dict


def evaluate_sample_trees(xhat_one, 
                          num_samples,
                          ama_options,  
                          InitSeed=0,  
                          model_module = None):
    """ Create and evaluate multiple sampled trees.
    Args:
        xhat_one : list or np.array of float (*not* a dict)
            A feasible and nonanticipative first stage solution.
        num_samples (int): number of trees to sample
        ama_options (dict): options for the amalgamator
        InitSeed (int): starting seed (but might be used for a scenario name offset)
        model_modules: an imported module with the functions needed by, e.g., amalgamator
    Returns:
        zhats (list as np.array): the objective functions
        seed (int): the updated seed or offset for scenario name sampling        
    """
    ''' creates batch_size sample trees with first-stage solution xhat_one
    using SampleSubtree class from sample_tree
    used to approximate E_{xi_2} phi(x_1, xi_2) for confidence interval coverage experiments
    note: ama_options will include the command line args as 'args'
    '''
    mname = ama_options["args"].model_module_name
    seed = InitSeed
    zhats = list()
    bfs = ama_options["branching_factors"]
    scenario_count = np.prod(bfs)
    solvername = ama_options["EF_solver_name"]
    #sampling_bfs = ciutils.scalable_BFs(batch_size, bfs) # use for variance?
    xhat_eval_options = {"iter0_solver_options": None,
                     "iterk_solver_options": None,
                     "display_timing": False,
                     "solvername": solvername,
                     "verbose": False,
                     "solver_options":{}}

    ### wrestling with options 5 April 2022 - TBD delete this comment
    if 'seed' not in ama_options:
        ama_options['seed'] = seed
    ###scenario_creator_kwargs = model_module.kw_creator(ama_options)
    for j in range(num_samples): # number of sample trees to create
        samp_tree = sample_tree.SampleSubtree(mname,
                                              xhats = [],
                                              root_scen=None,
                                              starting_stage=1, 
                                              branching_factors=bfs,
                                              seed=seed, 
                                              options=ama_options,
                                              solvername=solvername,
                                              solver_options={})
        samp_tree.run()
        ama_object = samp_tree.ama
        ama_options = ama_object.options
        ama_options['verbose'] = False
        scenario_creator_kwargs = ama_object.kwargs
        if len(samp_tree.ef._ef_scenario_names)>1:
            local_scenarios = {sname: getattr(samp_tree.ef, sname)
                               for sname in samp_tree.ef._ef_scenario_names}
        else:
            local_scenarios = {samp_tree.ef._ef_scenario_names[0]:samp_tree.ef}

        xhats, seed = sample_tree.walking_tree_xhats(mname,
                                                     local_scenarios,
                                                     xhat_one,
                                                     bfs,
                                                     seed,
                                                     ama_options,
                                                     solvername=solvername,
                                                     solver_options=None)
        # for Xhat_Eval
        # scenario_creator_kwargs = ama_object.kwargs
        scenario_names = ama_object.scenario_names
        all_nodenames = sputils.create_nodenames_from_branching_factors(bfs)

        # Evaluate objective value of feasible policy for this tree
        ev = Xhat_Eval(xhat_eval_options,
                        scenario_names,
                        ama_object.scenario_creator,
                        model_module.scenario_denouement,
                        scenario_creator_kwargs=scenario_creator_kwargs,
                        all_nodenames=all_nodenames)

        zhats.append(ev.evaluate(xhats))

        seed += scenario_count

    return np.array(zhats), seed

def run_samples(ama_options, args, model_module):
    
    # TBD: This has evolved so there may be overlap between ama_options and args
    #  (some codes assume that ama_options includes "args": args)
    
    # Read xhats from xhatpath
    xhat_one = ciutils.read_xhat(args.xhatpath)["ROOT"]

    num_samples = args.num_samples

    zhats,seed = evaluate_sample_trees(xhat_one, num_samples,
                                       ama_options, InitSeed=0,
                                       model_module=model_module)

    confidence_level = args.confidence_level
    zhatbar = np.mean(zhats)
    s_zhat = np.std(np.array(zhats))
    t_zhat = scipy.stats.t.ppf(confidence_level, len(zhats)-1)
    eps_z = t_zhat*s_zhat/np.sqrt(len(zhats))

    print('zhatbar: ', zhatbar)
    print('estimate: ', [zhatbar-eps_z, zhatbar+eps_z])
    print('confidence_level', confidence_level)

    return zhatbar, eps_z

def _parser_setup():
    # return the parser and set up the config object
    # parsers for the non-model-specific arguments; but the model_module_name will be pulled off first
    config.add_to_config("solver_name",
                         description="solver name (default gurobi_direct)",
                         domain=str,
                         default="gurobi_direct")
    config.branching_factors()
    config.add_to_config("num_samples",
                         description="Number of independent sample trees to construct",
                         domain=int,
                         default=10)
    config.add_to_config("solver_options",
                         description="space separated string of solver options, e.g. 'option1=value1 option2 = value2'",
                         default='')
    
    config.add_to_config("confidence_level",
                         description="one minus alpha (default 0.95)",
                         domain=float,
                         default=0.95)


def _main_body(args, model_module):
    # body of main, pulled out for testing
    solver_options = option_string_to_dict(args.solver_options)

    bfs = args.branching_factors
    solver_name = args.solver_name
    num_samples = args.num_samples

    ama_options = {"EF-mstage": True,
                   "EF_solver_name": solver_name,
                   "EF_solver_options": solver_options,
                   "branching_factors": bfs,
                   "args": args,
                   }

    return run_samples(ama_options, args, model_module)


if __name__ == "__main__":

    parser = _parser_setup()
    parser.add_argument('model_module_name',
                        help="name of model module, must be compatible with amalgamator")
    
    # now get the extra args from the module
    mname = sys.argv[1]  # args.model_module_name eventually
    if mname[-3:] == ".py":
        raise ValueError(f"Module name should end in .py ({mname})")
    try:
        model_module = importlib.import_module(mname)
    except:
        raise RuntimeError(f"Could not import module: {mname}")
    model_module.inparser_adder(parser)
    
    args = parser.parse_args()  # from the command line
    args = config.global_config.import_argparse(args)
    global_config._args = args  # To make it generally available

    zhatbar, eps_z = _main_body(args, model_module)
