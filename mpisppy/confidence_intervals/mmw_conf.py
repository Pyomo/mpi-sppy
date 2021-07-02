# This software is distributed under the 3-clause BSD License.

# python mmw_conf.py instance.py xhat.npy solver --alpha 0.95 --MMW-num-batches 1 --MMW-batch-size 3

# To test: (from confidence_intervals directory) 
# python mmw_conf.py mpisppy/tests/examples/farmer.py ../../examples/farmer/farmer_root_nonants.npy gurobi
# python -m mmw_conf farmer.py farmer_root_nonants.npy gurobi
import sys
import argparse
import mpisppy.utils.xhat_eval as xhat_eval
import mpisppy.utils.sputils as sputils
import re
from mpisppy.confidence_intervals import mmw_ci
import os
from mpisppy import global_toc
        


if __name__ == "__main__":

    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument('instance')
    parser.add_argument('xhatpath')
    parser.add_argument('solver_name')
    parser.add_argument('--alpha', 
                            help="defineds confidence interval (default 0.95)", 
                            dest="alpha", 
                            type = float,
                            default=None) #None will set alpha = 0.95 and tell user
    parser.add_argument('--objective-gap',
                        help="option to return gap from 0 or around objective value (default True)",
                        dest="objective_gap",
                        type=int,
                        default=1)
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

    args = parser.parse_args()

    # convert instance path to module name:
    modelpath = re.sub('/','.', args.instance)
    modelpath = re.sub(r'\.py','', modelpath)

    # Read xhats from xhatpath
    xhat = mmw_ci.read_xhat(args.xhatpath)

    num_scens = len(xhat['ROOT'])

    if args.batch_size == None:
        args.batch_size = num_scens

    refmodel = modelpath #Change this path to use a different model
    
    options = {"EF-2stage": True,# 2stage vs. mstage
               "start": False,
               "EF_solver_name": args.solver_name,
               "num_scens": num_scens}   #Are the scenario shifted by a start arg ?
   

    #should we except these as arguments?
    #num_batches = args.num_batches
    num_batches = args.num_batches
    batch_size = args.batch_size

    if args.objective_gap == 1:
        objective_gap = True
    else:
        objective_gap = False
    
    # mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, xhat, num_batches, batch_size=batch_size,
    #                    verbose=True)

    mmwObjective = mmw_ci.MMWConfidenceIntervals(refmodel, options, xhat, num_batches, batch_size=batch_size,
                       verbose=True)

    if args.alpha == None:
        print('\nNo alpha given, defaulting to alpha = 0.95. To provide an alpha try:\n')
        print('python {} {} {} {} --alpha 0.97 --objective-gap {} --MMW-num-batches {} --MMW-batch-size {}\
            \n'.format(sys.argv[0], args.instance, args.xhatpath, args.solver_name, 
                args.objective_gap, args.num_batches, args.batch_size))
        alpha = 0.95,
    else: 
        alpha = float(args.alpha)

    #r=mmw.run(alpha)
    r = mmwObjective.run(confidence_level=alpha, objective_gap = args.objective_gap)

    global_toc(r)