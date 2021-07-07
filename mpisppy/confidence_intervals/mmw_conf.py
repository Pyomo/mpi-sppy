# This software is distributed under the 3-clause BSD License.

# python mmw_conf.py instance.py xhat.npy solver --alpha 0.95 --num-scens 3 --MMW-num-batches 1 --MMW-batch-size 3

# To test: (from confidence_intervals directory) 

# python mmw_conf.py mpisppy/tests/examples/farmer.py ../../examples/farmer/farmer_root_nonants.npy --MMW-num-batches 3 --MMW-batch-size 3 --num-scens 3 --solver-name gurobi 

import sys
import argparse
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.utils.sputils as sputils
import mpisppy.utils.baseparsers as baseparsers
import mpisppy.utils.vanilla as vanilla
import re
from mpisppy.confidence_intervals import mmw_ci
import os
from mpisppy import global_toc

        


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('instance',
                            help="name of model, must be compatible with amalgomator")
    parser.add_argument('xhatpath',
                            help="path to .npy file with feasible nonant solution xhat")
    parser.add_argument('--alpha', 
                            help="defines confidence interval size (default 0.95)", 
                            dest="alpha", 
                            type = float,
                            default=None) #None will set alpha = 0.95 and tell user
    parser.add_argument('--with-objective-gap',
                        help="option to return gap from 0 or around objective value (default True)",
                        dest="objective_gap",
                        action='store_true')
    parser.set_defaults(objective_gap=False)
    parser.add_argument('--num-scens',
                            help="number of scenenarios for EF, should match what is used to solve xhat",
                            type=int,
                            default=None)
    parser.add_argument("--MMW-num-batches",
                            help="number of batches used for MMW confidence interval (default 1)",
                            dest="num_batches",
                            type=int,
                            default=2)
    parser.add_argument("--MMW-batch-size",
                            help="batch size used for MMW confidence interval (default None)",
                            dest="batch_size",
                            type=int,
                            default=None) #None means take batch_size=num_scens
    parser = baseparsers._common_args(parser)

    args = parser.parse_args()
    print(sys.path[0])
    if args.num_scens == None:
        print('\n')
        raise Exception("Please include number of scenes used to compute the candidate solutions xhat.")
        print('\n')

    solver_options = vanilla.shared_options(args)
    # issues w/ gurobi in xhat_eval otherwise...
    solver_options["iter0_solver_options"] = None
    solver_options["iterk_solver_options"] = None

    # convert instance path to module name:
    modelpath = re.sub('/','.', args.instance)
    modelpath = re.sub(r'\.py','', modelpath)

    # Read xhats from xhatpath
    xhat = mmw_ci.read_xhat(args.xhatpath)
   
    if args.batch_size == None:
        args.batch_size = args.num_scens

    refmodel = modelpath #Change this path to use a different model
    
    options = {"EF-2stage": True,# 2stage vs. mstage
               "start": False,
               "EF_solver_name": args.solver_name,
               "EF_solver_options": solver_options,
               "num_scens": args.num_scens}   #Are the scenario shifted by a start arg ?
   

    #should we accept these as arguments?
    num_batches = args.num_batches
    batch_size = args.batch_size

    mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, xhat, num_batches, batch_size=batch_size,
                       verbose=True)

    if args.alpha == None:
        print('\nNo alpha given, defaulting to alpha = 0.95. To provide an alpha try:\n')
        print('python -m mpisppy.confidence_intervals.mmw_conf {} {} {} --alpha 0.97 --solver-name {} --MMW-num-batches {} --MMW-batch-size {} --num-scens {}\
            \n'.format(sys.argv[0], args.instance, args.xhatpath, args.solver_name, 
                args.num_batches, args.batch_size, args.num_scens))
        alpha = 0.95,
    else: 
        alpha = float(args.alpha)

    #r=mmw.run(alpha)
    r = mmw.run(confidence_level=alpha, objective_gap = args.objective_gap)

    global_toc(r)