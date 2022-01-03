# This software is distributed under the 3-clause BSD License.

# To test: (from confidence_intervals directory; assumes the npy file is in the farmer directory) 
# python mmw_conf.py mpisppy/tests/examples/farmer.py ../../examples/farmer/farmer_root_nonants.npy gurobi --MMW-num-batches 3 --MMW-batch-size 3 --num-scens 3

import re
import sys
import argparse
from pyomo.common.fileutils import import_file
from mpisppy.utils.sputils import option_string_to_dict
from mpisppy.confidence_intervals import ciutils
from mpisppy.confidence_intervals import mmw_ci
from mpisppy import global_toc



if __name__ == "__main__":

    # parse args for mmw part of things
    parser = argparse.ArgumentParser()
    parser.add_argument('instance',
                            help="name of model module, must be compatible with amalgaator")
    parser.add_argument('xhatpath',
                            help="path to .npy file with feasible nonant solution xhat")
    parser.add_argument('solver_name',
                            help="name of solver to be used",
                            default='gurobi_persistent')
    parser.add_argument('--confidence-level', 
                            help="defines confidence interval size (default 0.95)", 
                            dest="confidence_level", 
                            type = float,
                            default=None) #None will set alpha = 0.95 and tell user
    parser.add_argument('--with-objective-gap',
                        help="option to return gap around objective value (default False)",
                        dest="objective_gap",
                        action='store_true')
    parser.set_defaults(objective_gap=False)
    parser.add_argument('--start-scen',
                            help="starting scenario number,( maybe should match what was used to solve xhat), default 0",
                            dest='start_scen',
                            type=int,
                            default=0)
    parser.add_argument("--MMW-num-batches",
                            help="number of batches used for MMW confidence interval (default 2)",
                            dest="num_batches",
                            type=int,
                            default=2)
    parser.add_argument("--MMW-batch-size",
                            help="batch size used for MMW confidence interval, if None then batch_size = num_scens (default to None)",
                            dest="batch_size",
                            type=int,
                            default=None) #None means take batch_size=num_scens
    parser.add_argument("--solver-options",
                            help="space separated string of solver options, e.g. 'option1=value1 option2 = value2'",
                            default='')

    # now get the extra args from the module
    mname = sys.argv[1]  # args.instance eventually
    try:
        m = import_file(mname)
    except:
        try:
            m = import_file(f"{mname}.py")
        except:
            raise RuntimeError(f"Could not import module: {mname}")
    m.inparser_adder(parser)
    args = parser.parse_args()  

    #parses solver options string
    solver_options = option_string_to_dict(args.solver_options)

    # convert instance path to module name:
    modelpath = re.sub('/','.', args.instance)
    modelpath = re.sub(r'\.py','', modelpath)

    # Read xhats from xhatpath
    xhat = ciutils.read_xhat(args.xhatpath)

    if args.batch_size == None:
        args.batch_size = args.num_scens

    refmodel = modelpath #Change this path to use a different model
    
    options = {"args": args,  # in case of problem specific args
               "EF-2stage": True,  # 2stage vs. mstage
               "EF_solver_name": args.solver_name,
               "EF_solver_options": solver_options,
               "start_scen": args.start_scen}   #Are the scenario shifted by a start arg ?

    num_batches = args.num_batches
    batch_size = args.batch_size

    mmw = mmw_ci.MMWConfidenceIntervals(refmodel, options, xhat, num_batches, batch_size=batch_size, start = args.start_scen,
                       verbose=True)

    cl = float(args.confidence_level)

    r = mmw.run(confidence_level=cl, objective_gap = args.objective_gap)

    global_toc(r)
    
