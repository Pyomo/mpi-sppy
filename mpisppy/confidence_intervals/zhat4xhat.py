# script to estimate zhat from a given xhat for a given model

import sys
import argparse
import importlib
import numpy as np
import scipy.stats
from mpisppy.confidence_intervals import sample_tree
from mpisppy.utils import sputils
from mpisppy.confidence_intervals import ciutils
from mpisppy.utils.xhat_eval import Xhat_Eval
from mpisppy.utils.sputils import option_string_to_dict
from mpisppy.utils import config

def evaluate_sample_trees(xhat_one, 
                          num_samples,
                          cfg,
                          InitSeed=0,  
                          model_module = None):
    """ Create and evaluate multiple sampled trees.
    Args:
        xhat_one : list or np.array of float (*not* a dict)
            A feasible and nonanticipative first stage solution.
        num_samples (int): number of trees to sample
        cfg (Config): options for/from the amalgamator
        InitSeed (int): starting seed (but might be used for a scenario name offset)
        model_modules: an imported module with the functions needed by, e.g., amalgamator
    Returns:
        zhats (list as np.array): the objective functions
        seed (int): the updated seed or offset for scenario name sampling        
    """
    ''' creates batch_size sample trees with first-stage solution xhat_one
    using SampleSubtree class from sample_tree
    used to approximate E_{xi_2} phi(x_1, xi_2) for confidence interval coverage experiments
    '''
    cfg = cfg()  # so the seed can be ephemeral
    mname = cfg.model_module_name
    seed = InitSeed
    zhats = list()
    bfs = cfg["branching_factors"]
    scenario_count = np.prod(bfs)
    solver_name = cfg["EF_solver_name"]
    #sampling_bfs = ciutils.scalable_BFs(batch_size, bfs) # use for variance?
    xhat_eval_options = {"iter0_solver_options": None,
                     "iterk_solver_options": None,
                     "display_timing": False,
                     "solver_name": solver_name,
                     "verbose": False,
                     "solver_options":{}}

    cfg.dict_assign('seed', 'seed', int, None, seed)

    for j in range(num_samples): # number of sample trees to create
        samp_tree = sample_tree.SampleSubtree(mname,
                                              xhats = [],
                                              root_scen=None,
                                              starting_stage=1, 
                                              branching_factors=bfs,
                                              seed=seed, 
                                              cfg=cfg,
                                              solver_name=solver_name,
                                              solver_options={})
        samp_tree.run()
        ama_object = samp_tree.ama
        cfg = ama_object.cfg
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
                                                     cfg,
                                                     solver_name=solver_name,
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

def run_samples(cfg, model_module):
    # Does all the work for zhat4xhat    

    # Read xhats from xhatpath
    xhat_one = ciutils.read_xhat(cfg.xhatpath)["ROOT"]

    num_samples = cfg.num_samples

    zhats,seed = evaluate_sample_trees(xhat_one, num_samples,
                                       cfg, InitSeed=0,
                                       model_module=model_module)

    confidence_level = cfg.confidence_level
    zhatbar = np.mean(zhats)
    s_zhat = np.std(np.array(zhats))
    t_zhat = scipy.stats.t.ppf(confidence_level, len(zhats)-1)
    eps_z = t_zhat*s_zhat/np.sqrt(len(zhats))

    print('zhatbar: ', zhatbar)
    print('estimate: ', [zhatbar-eps_z, zhatbar+eps_z])
    print('confidence_level', confidence_level)

    return zhatbar, eps_z

def _parser_setup():
    # set up the config object and return it, but don't parse
    # parsers for the non-model-specific arguments; but the model_module_name will be pulled off first
    cfg = config.Config()
    cfg.add_to_config("EF_solver_name",
                         description="solver name (default gurobi_direct)",
                         domain=str,
                         default="gurobi_direct")
    cfg.add_branching_factors()
    cfg.add_to_config("num_samples",
                         description="Number of independent sample trees to construct",
                         domain=int,
                         default=10)
    cfg.add_to_config("solver_options",
                         description="space separated string of solver options, e.g. 'option1=value1 option2 = value2'",
                         domain=str,
                         default='')
    
    cfg.add_to_config("xhatpath",
                         description="path to npy file with xhat",
                         domain=str,
                         default='')
    
    cfg.add_to_config("confidence_level",
                         description="one minus alpha (default 0.95)",
                         domain=float,
                         default=0.95)
    return cfg


def _main_body(model_module, cfg):
    # body of main, pulled out for testing

    lcfg = cfg()  # make a copy because of EF-mstage
    solver_options_dict = option_string_to_dict(cfg.solver_options)
    lcfg.add_and_assign("EF_solver_options", "solver options dict", dict, None, solver_options_dict)
    
    lcfg.quick_assign("EF_mstage", domain=bool, value=True)

    return run_samples(lcfg, model_module)


if __name__ == "__main__":

    cfg = _parser_setup()
    # now get the extra args from the module
    mname = sys.argv[1]  # args.model_module_name eventually
    if mname[-3:] == ".py":
        raise ValueError(f"Module name should not end in .py ({mname})")
    try:
        model_module = importlib.import_module(mname)
    except:
        raise RuntimeError(f"Could not import module: {mname}")
    model_module.inparser_adder(cfg)
    # TBD xxxx the inprser_adder might want num_scens, but zhat4xhat contols the number of scenarios
    # see if this works:
    try:
        del cfg.num_scens
    except:
        pass
    
    parser = cfg.create_parser("zhat4zhat")
    # the module name is very special because it has to be plucked from argv
    parser.add_argument(
            "model_module_name", help="amalgamator compatible module (often read from argv)", type=str,
        )
    cfg.add_to_config("model_module_name",
                         description="amalgamator compatible module",
                         domain=str,
                         default='',
                         argparse=False)
    cfg.model_module_name = mname
    
    args = parser.parse_args()  # from the command line
    args = cfg.import_argparse(args)

    zhatbar, eps_z = _main_body(model_module, cfg)
