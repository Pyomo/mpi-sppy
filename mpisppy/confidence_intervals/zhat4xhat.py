# script to estimate zhat from a given xhat for a given model

####from mpisppy.tests.examples import aircond_submodels 
from mpisppy.confidence_intervals.multi_seqsampling import IndepScens_SeqSampling
from mpisppy.confidence_intervals import mmw_ci
from mpisppy.confidence_intervals import sample_tree
from mpisppy.utils import sputils
from mpisppy.utils.xhat_eval import Xhat_Eval
import numpy as np
import time
import scipy.stats
import argparse


def evaluate_sample_trees(xhat_one, 
                          num_samples,
                          ama_options,  
                          SeedCount=0,  
                          mname='mpisppy.tests.examples.aircond_submodels'):
    ''' creates batch_size sample trees with first-stage solution xhat_one
    using SampleSubtree class from sample_tree
    used to approximate E_{xi_2} phi(x_1, xi_2) for confidence interval coverage experiments
    note: ama_options will include the command line args as 'args'
    '''
    seed = SeedCount
    zhats = list()
    bfs = ama_options["branching_factors"]
    cb_dict = ama_options['cb_dict']
    solvername = ama_options["EF_solver_name"]
    #sampling_bfs = ciutils.scalable_BFs(batch_size, bfs) # use for variance?
    xhat_eval_options = {"iter0_solver_options": None,
                     "iterk_solver_options": None,
                     "display_timing": False,
                     "solvername": solvername,
                     "verbose": False,
                     "solver_options":{}}

    for j in range(num_samples): # number of sample trees to create
        scenario_creator_kwargs={'branching_factors':bfs, 'start_seed':seed,'cb_dict':cb_dict}
        samp_tree = sample_tree.SampleSubtree(mname,
                                              xhats = [],
                                              root_scen=None,
                                              starting_stage=1, 
                                              BFs=bfs,
                                              seed=seed, 
                                              options=scenario_creator_kwargs,
                                              solvername=solvername,
                                              solver_options={})
        samp_tree.run()
        ama_object = samp_tree.ama
        ama_options = ama_object.options
        ama_options['verbose'] = False
        scenario_creator_kwargs = ama_object.kwargs
        if len(samp_tree.ef._ef_scenario_names)>1:
            local_scenarios = {sname: getattr(samp_tree.ef,sname)/
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
        all_nodenames = sputils.create_nodenames_from_BFs(bfs)

        # Evaluate objective value of feasible policy for this tree
        ev = Xhat_Eval(xhat_eval_options,
                        scenario_names,
                        ama_object.scenario_creator,
                        aircond_submodels.scenario_denouement,
                        scenario_creator_kwargs=scenario_creator_kwargs,
                        all_nodenames=all_nodenames)

        zhats.append(ev.evaluate(xhats))

    return np.array(zhats), seed

def run_samples(ama_options, args):
    # TBD: This has evolved so there may be overlap between ama_options and args
    # Read xhats from xhatpath
    xhat = ciutils.read_xhat(args.xhatpath)
    
    zhats,seed = evaluate_sample_trees(xhat_one, num_samples, ama_options, SeedCount=0)

    start_ups = ama_options["cb_dict"]["start_ups"]
    np.savetxt('aircondResults/aircond_start_ups='+str(start_ups)+\
        'xhat_one='+str(xhat_one[0])+'_'+str(xhat_one[1])+\
        '_zhatstars.txt', zhats)        

    confidence_level=.99
    zhatbar = np.mean(zhats)
    s_zhat = np.std(np.array(zhats))
    t_zhat = scipy.stats.t.ppf(confidence_level, len(zhats)-1)
    eps_z = t_zhat*s_zhat/np.sqrt(len(zhats))

    print('zhatbar: ', zhatbar)
    print('estimate: ', [zhatbar-eps_z, zhatbar+eps_z])

if __name__ == "__main__":

    # parsers for the non-model-specific arguments; but the instance will be pull off first
    parser = argparse.ArgumentParser()

    parser.add_argument('instance',
                        help="name of model module, must be compatible with amalgomator")
    parser.add_argument('xhatpath',
                        help="path to .npy file with feasible nonant solution xhat")
    parser.add_argument("--solver-name",
                        help="solver name (default gurobi_direct)",
                        dest='solver_name',
                        default="gurobi_direct")
    parser.add_argument("--branching-factors",
                        help="Spaces delimited branching factors (default 10 10) for two "
                        "stage, just enter one number",
                        dest="BFs",
                        nargs="*",
                        type=int,
                        default=[10,10])
    parser.add_argument("--num-samples",
                        help="Number of independent sample trees to construct",
                        dest="num_samples",
                        type=int,
                        default=10)
    args = parser.parse_args()

    # now get the extra args from the module
    mname = sys.argv[1]  # args.instance eventually
    if mname[-3:] == ".py":
        mname = mname[-3:]
    try:
        model_module = importlib.import_module(mname)
    except:
        raise RuntimeError(f"Could not import module: {mname}")
    model_module.inparser_adder(parser)
    args = parser.parse_args()  

    #parses solver options string
    solver_options = option_string_to_dict(args.solver_options)

    bfs = args.branching_factors
    solver_name = args.solver_name
    num_samples = args.num_samples

    ama_options = {"EF-mstage": True,
                   "EF_solver_name": solver_name,
                   "EF_solver_options": solver_options,
                   "branching_factors": bfs,
                   }

    run_samples(ama_options, args)
